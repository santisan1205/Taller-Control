import numpy as np
import torch
from setup_entorno import crear_entorno_sumo
from modelo_red_rl import MAPPOActor, MAPPOCritic, RolloutBuffer, MAPPOTrainer, RNDModule, RewardCalculator

def obtener_estado_global(obs_dict, agents):
    """
    Concatena las observaciones locales de todos los agentes para formar el estado global
    que requiere el Crítico centralizado de MAPPO.
    """
    return np.concatenate([obs_dict[agent] for agent in agents])

def train():
    # 1. Inicialización del Entorno
    RED_XML = 'red_urbana.net.xml'
    RUTAS_XML = 'demanda_trafico.rou.xml'
    env = crear_entorno_sumo(RED_XML, RUTAS_XML, use_gui=False)
    
    obs_iniciales, _ = env.reset()
    agentes = env.agents
    num_agentes = len(agentes)
    
    # Dimensiones de observación y acción
    obs_dim = env.observation_space(agentes[0]).shape[0]
    action_dim = env.action_space(agentes[0]).n
    global_obs_dim = obs_dim * num_agentes
    
    # 2. Inicialización de Redes y Módulos
    # En MAPPO con parámetros compartidos, usamos un solo Actor y un solo Crítico para todos los cruces
    actor = MAPPOActor(obs_dim, action_dim)
    critic = MAPPOCritic(global_obs_dim)
    trainer = MAPPOTrainer(actor, critic)
    
    # Un buffer y un módulo RND por agente para rastrear la curiosidad local
    buffers = {agent: RolloutBuffer() for agent in agentes}
    rnd_modules = {agent: RNDModule(obs_dim) for agent in agentes}
    reward_calc = RewardCalculator()
    
    # Hiperparámetros de entrenamiento
    EPISODIOS = 100
    PASOS_POR_EPISODIO = 720 # 3600 segundos / delta_time de 5s
    lambda_t = 1.0
    
    # 3. Bucle Principal de Entrenamiento
    for episodio in range(EPISODIOS):
        obs_dict, _ = env.reset()
        estado_global = obtener_estado_global(obs_dict, agentes)
        
        recompensa_acumulada = 0
        
        for paso in range(PASOS_POR_EPISODIO):
            acciones = {}
            log_probs = {}
            valores_estado = {}
            
            # El Crítico evalúa el estado actual de toda la ciudad
            valor_global = critic(torch.tensor(estado_global, dtype=torch.float32)).item()
            
            # Cada agente decide su acción de forma descentralizada
            for agent in agentes:
                obs_tensor = torch.tensor(obs_dict[agent], dtype=torch.float32)
                accion, log_prob = actor.get_action(obs_tensor)
                
                acciones[agent] = accion
                log_probs[agent] = log_prob
                valores_estado[agent] = valor_global
                
            # Ejecutar acciones en el entorno
            siguientes_obs, r_ext_dict, terminations, truncations, infos = env.step(acciones)
            siguiente_estado_global = obtener_estado_global(siguientes_obs, agentes)
            
            # Calcular recompensas totales (Extrínseca + Intrínseca) y almacenar
            for agent in agentes:
                r_ext = r_ext_dict[agent]
                
                # Recompensa intrínseca (curiosidad)
                r_int = rnd_modules[agent].compute_intrinsic_reward(siguientes_obs[agent])
                rnd_modules[agent].update_predictor(siguientes_obs[agent])
                
                # Recompensa total
                r_total = reward_calc.calculate_total_reward(r_ext, r_int, lambda_t)
                recompensa_acumulada += r_total
                
                # Guardar en la memoria del agente
                done = terminations[agent] or truncations[agent]
                buffers[agent].almacenar(
                    obs_dict[agent], estado_global, acciones[agent], 
                    log_probs[agent], r_total, valores_estado[agent], done
                )
                
            obs_dict = siguientes_obs
            estado_global = siguiente_estado_global
            
            if all(terminations.values()) or all(truncations.values()):
                break
                
        # 4. Fase de Actualización (Optimización PPO al final del episodio)
        print(f"Episodio {episodio + 1}/{EPISODIOS} | Recompensa Total: {recompensa_acumulada:.2f} | Lambda RND: {lambda_t:.2f}")
        
        # Siguiente valor para calcular la ventaja (Bootstrapping)
        next_global_val = critic(torch.tensor(estado_global, dtype=torch.float32)).item()
        
        for agent in agentes:
            # Calcular GAE y extraer tensores del buffer
            buffer_tensors = buffers[agent].calcular_ventajas_gae(next_global_val)
            
            # Actualizar redes Actor y Crítico
            actor_loss, critic_loss = trainer.update(buffer_tensors)
            
            # Vaciar el buffer para la siguiente política (On-Policy)
            buffers[agent].limpiar()
            
        # Decaimiento del coeficiente de exploración RND
        progreso = episodio / EPISODIOS
        lambda_t = max(0.01, 1.0 - progreso)
        
    env.close()
    
    # Guardar los pesos entrenados
    torch.save(actor.state_dict(), 'actor_mappo.pth')
    torch.save(critic.state_dict(), 'critic_mappo.pth')
    print("Entrenamiento completado y modelos guardados.")

if __name__ == '__main__':
    train()