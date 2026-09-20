import os
import sys
import numpy as np
from sumo_rl import parallel_env

# 1. Definición de la Recompensa Extrínseca Personalizada
def recompensa_delta_colas(traffic_signal):
    """
    Calcula R_ext basándose en la diferencia de colas y tiempos de espera.
    traffic_signal: Objeto de la clase TrafficSignal de sumo-rl.
    """
    # Obtener métricas actuales (t)
    cola_actual = traffic_signal.get_total_queued()
    espera_actual = traffic_signal.get_accumulated_waiting_time_per_lane()
    espera_total_actual = sum(espera_actual.values())
    
    # Obtener métricas del paso anterior (t-1) guardadas en el entorno
    # (Requiere inicializar estas variables personalizadas en el primer paso)
    cola_previa = getattr(traffic_signal, 'cola_previa', cola_actual)
    espera_previa = getattr(traffic_signal, 'espera_previa', espera_total_actual)
    
    # Calcular los deltas (Δ)
    delta_q = cola_actual - cola_previa
    delta_w = espera_total_actual - espera_previa
    
    # Actualizar para el siguiente paso
    traffic_signal.cola_previa = cola_actual
    traffic_signal.espera_previa = espera_total_actual
    
    # Pesos definidos en nuestro modelo
    w1, w2 = 1.0, 0.5 
    
    # Fórmula: R_ext = -(w1*ΔQ + w2*ΔW). Si la cola baja, delta_q es negativo -> recompensa positiva
    reward = -(w1 * delta_q + w2 * delta_w)
    
    return reward

# 2. Configuración del Entorno Multi-Agente
def crear_entorno_sumo(net_file, route_file, use_gui=False):
    """
    Inicializa el entorno PettingZoo con la Selección Directa de Fase.
    """
    env = parallel_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name='resultados/mappo_rnd_train',
        use_gui=use_gui,
        num_seconds=3600,       # Duración de la simulación (1 hora en segundos de SUMO)
        delta_time=5,           # Δt: El agente toma decisiones cada 5 segundos
        yellow_time=3,          # Tiempo amarillo obligatorio (Transición)
        min_green=15,           # Tiempo verde mínimo estricto
        max_green=60,           # Tiempo verde máximo antes de forzar rotación
        reward_fn=recompensa_delta_colas, # Inyección de nuestra R_ext
        observation_class='DefaultObservationFunction' # Vector de colas, densidad y fase one-hot
    )
    return env

# 3. Bucle de Ejecución Paralela (Prueba Aleatoria)
if __name__ == '__main__':
    # Rutas relativas a tus archivos XML de SUMO
    RED_XML = 'red_urbana.net.xml'
    RUTAS_XML = 'demanda_trafico.rou.xml'
    
    env = crear_entorno_sumo(RED_XML, RUTAS_XML, use_gui=True)
    
    # Reset inicializa el estado y devuelve las observaciones de cada semáforo
    observations, infos = env.reset()
    
    print(f"Agentes en la red: {env.agents}")
    print(f"Espacio de Acción (Semáforo 1): {env.action_space(env.agents[0])}")
    
    terminado = False
    
    while env.agents:
        # Aquí eventualmente irá la salida de la política de MAPPO
        # Por ahora, muestreamos acciones aleatorias válidas para cada agente
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # El entorno ejecuta un paso de simulación con las acciones
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Imprimir la recompensa extrínseca generada en este paso
        # print(f"Recompensas: {rewards}")
        
    env.close()
    print("Simulación finalizada.")