import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import numpy as np
import torch
from setup_entorno import crear_entorno_sumo, CARPETA_SUMO
from modelo_red_rl import MAPPOActor, MAPPOCritic, RolloutBuffer, MAPPOTrainer, RNDModule, RewardCalculator

# 1. Define el dispositivo de cómputo: GPU si está disponible, de lo contrario CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print(f"Usando dispositivo: {device}")

def obtener_estado_global(obs_dict, agents):
    """
    Concatena las observaciones locales de todos los agentes para formar el estado global
    que requiere el Crítico centralizado de MAPPO.
    """
    return np.concatenate([obs_dict[agent] for agent in agents])

def train(episodios=100, pasos_por_episodio=720, use_gui=False, seed=None):
    # Semilla: fija la inicialización de las redes (torch) y la demanda de trafico
    # (sumo_seed), para poder reproducir una corrida y comparar varias semillas.
    # Si no se da semilla, se preserva el comportamiento original (todo aleatorio,
    # rutas de salida sin sufijo) para no romper la corrida base ya guardada.
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        out_csv_name = f'resultados/seed{seed}/mappo_rnd_train'
        sufijo_pesos = f'_seed{seed}'
    else:
        out_csv_name = 'resultados/mappo_rnd_train'
        sufijo_pesos = ''
    sumo_seed = seed if seed is not None else 'random'

    # 1. Inicialización del Entorno con el escenario SUMO real (sector de Bogotá modelado)
    RED_XML = os.path.join(CARPETA_SUMO, 'Config_actualizado.net.xml')
    RUTAS_XML = ','.join([
        os.path.join(CARPETA_SUMO, 'rutas_aleatorias.rou.xml'),
        os.path.join(CARPETA_SUMO, 'flujos_especificos.rou.xml'),
    ])
    env = crear_entorno_sumo(RED_XML, RUTAS_XML, use_gui=use_gui, sumo_seed=sumo_seed, out_csv_name=out_csv_name)

    obs_iniciales, _ = env.reset()
    agentes = env.agents

    # Dimensiones de observación y acción POR AGENTE: los 4 cruces reales tienen
    # distinto número de carriles y de fases, así que no se puede asumir un
    # obs_dim/action_dim único compartido por todos los semáforos.
    obs_dims = {agent: env.observation_space(agent).shape[0] for agent in agentes}
    action_dims = {agent: env.action_space(agent).n for agent in agentes}
    global_obs_dim = sum(obs_dims.values())

    # 2. Inicialización de Redes y Módulos
    # Un Actor independiente por agente (dimensiones heterogéneas impiden compartir pesos)
    # y un único Crítico centralizado que observa el estado global de la red (paradigma CTDE)
    # Se envía a la GPU si está disponible, de lo contrario a la CPU
    actores = {agent: MAPPOActor(obs_dims[agent], action_dims[agent]).to(device) for agent in agentes}
    critic = MAPPOCritic(global_obs_dim).to(device)
    trainers = {agent: MAPPOTrainer(actores[agent], critic) for agent in agentes}

    # Un buffer y un módulo RND por agente para rastrear la curiosidad local
    buffers = {agent: RolloutBuffer() for agent in agentes}
    rnd_modules = {agent: RNDModule(obs_dims[agent]) for agent in agentes}
    reward_calc = RewardCalculator()
    
    # Hiperparámetros de entrenamiento
    EPISODIOS = episodios
    PASOS_POR_EPISODIO = pasos_por_episodio # 720 = 3600 segundos / delta_time de 5s (episodio completo)
    lambda_t = 1.0
    
    # 3. Bucle Principal de Entrenamiento
    for episodio in range(EPISODIOS):
        obs_dict, _ = env.reset()
        estado_global = obtener_estado_global(obs_dict, agentes)
        
        recompensa_acumulada = 0
        
        for paso in range(PASOS_POR_EPISODIO):
            # El Crítico evalúa el estado actual de toda la ciudad en la GPU
            estado_global_tensor = torch.tensor(estado_global, dtype=torch.float32).to(device)
            valor_global = critic(estado_global_tensor).item()
            acciones = {}
            log_probs = {}
            valores_estado = {}
            
            # Cada agente decide su acción de forma descentralizada
            for agent in agentes:
                obs_tensor = torch.tensor(obs_dict[agent], dtype=torch.float32).to(device)
                accion, log_prob = actores[agent].get_action(obs_tensor)
                
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
        estado_global_next = torch.tensor(estado_global, dtype=torch.float32).to(device)
        next_global_val = critic(estado_global_next).item()
        
        for agent in agentes:
            # Calcular GAE y extraer tensores del buffer
            
            buffer_tensors = buffers[agent].calcular_ventajas_gae(next_global_val)
            
            # Actualizar redes Actor y Crítico (cada agente actualiza su propio Actor;
            # el Crítico, al ser compartido, recibe una actualización por agente)
            actor_loss, critic_loss = trainers[agent].update(buffer_tensors)
            
            # Vaciar el buffer para la siguiente política (On-Policy)
            buffers[agent].limpiar()
            
        # Decaimiento del coeficiente de exploración RND
        progreso = episodio / EPISODIOS
        lambda_t = max(0.01, 1.0 - progreso)
        
    # sumo-rl guarda el CSV de métricas de cada episodio en el siguiente reset();
    # el último episodio no dispara ningún reset() posterior, así que hay que
    # forzar su guardado antes de cerrar el entorno.
    entorno_sumo = env.unwrapped.env
    entorno_sumo.save_csv(entorno_sumo.out_csv_name, entorno_sumo.episode)
    env.close()

    # Guardar los pesos entrenados: un Actor por agente (dimensiones heterogéneas)
    # y el Crítico centralizado compartido
    for agent in agentes:
        torch.save(actores[agent].state_dict(), f'actor_mappo_{agent}{sufijo_pesos}.pth')
    torch.save(critic.state_dict(), f'critic_mappo{sufijo_pesos}.pth')
    print("Entrenamiento completado y modelos guardados.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Entrena RND-MAPPO sobre el escenario SUMO de Bogotá.')
    parser.add_argument('--episodios', type=int, default=100)
    parser.add_argument('--pasos', type=int, default=720, help='Pasos por episodio (720 = episodio completo de 1h simulada)')
    parser.add_argument('--gui', action='store_true', help='Muestra la ventana de sumo-gui durante el entrenamiento')
    parser.add_argument('--seed', type=int, default=None, help='Semilla para redes y demanda (reproducibilidad / comparar varias corridas)')
    args = parser.parse_args()

    train(episodios=args.episodios, pasos_por_episodio=args.pasos, use_gui=args.gui, seed=args.seed)
