import os
import sys
import numpy as np
from sumo_rl import parallel_env
from sumo_rl.environment.observations import DefaultObservationFunction

from modelo_red_rl import RewardCalculator

# Carpeta con el escenario SUMO real (red y demanda del sector de Bogotá modelado)
CARPETA_SUMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2026-09-12-17-03-26')

# Un único calculador de recompensa compartido; la fórmula (pesos w1/w2/w3 y
# penalización por cambio de fase) vive en modelo_red_rl.RewardCalculator
_reward_calc = RewardCalculator()

def _espera_peatonal(traffic_signal):
    """
    Suma el tiempo de espera de los peatones presentes en los edges de entrada
    del semáforo. sumo-rl no trackea peatones de forma nativa, así que se
    consulta TraCI/libsumo directamente a través de traffic_signal.sumo.
    """
    sumo = traffic_signal.sumo
    edges = {sumo.lane.getEdgeID(lane) for lane in traffic_signal.lanes}
    espera = 0.0
    for edge in edges:
        for persona_id in sumo.edge.getLastStepPersonIDs(edge):
            espera += sumo.person.getWaitingTime(persona_id)
    return espera

# 1. Definición de la Recompensa Extrínseca Personalizada
def recompensa_delta_colas(traffic_signal):
    """
    Calcula R_ext basándose en la diferencia de colas y tiempos de espera
    (vehicular y peatonal), delegando la fórmula a RewardCalculator para que
    exista una sola definición de R_ext en todo el proyecto.
    traffic_signal: Objeto de la clase TrafficSignal de sumo-rl.
    """
    # Obtener métricas actuales (t)
    cola_actual = traffic_signal.get_total_queued()
    espera_por_carril = traffic_signal.get_accumulated_waiting_time_per_lane()
    espera_veh_actual = sum(espera_por_carril)
    espera_ped_actual = _espera_peatonal(traffic_signal)
    fase_actual = traffic_signal.green_phase

    # Obtener métricas del paso anterior (t-1) guardadas en el propio objeto
    # (se inicializan la primera vez que se llama para este semáforo)
    cola_previa = getattr(traffic_signal, 'cola_previa', cola_actual)
    espera_veh_previa = getattr(traffic_signal, 'espera_veh_previa', espera_veh_actual)
    espera_ped_previa = getattr(traffic_signal, 'espera_ped_previa', espera_ped_actual)
    fase_previa = getattr(traffic_signal, 'fase_previa', fase_actual)

    # Calcular los deltas (Δ)
    delta_q = cola_actual - cola_previa
    delta_w_veh = espera_veh_actual - espera_veh_previa
    delta_w_ped = espera_ped_actual - espera_ped_previa
    cambio_de_fase = fase_actual != fase_previa

    # Actualizar para el siguiente paso
    traffic_signal.cola_previa = cola_actual
    traffic_signal.espera_veh_previa = espera_veh_actual
    traffic_signal.espera_ped_previa = espera_ped_actual
    traffic_signal.fase_previa = fase_actual

    return _reward_calc.calculate_extrinsic_reward(delta_q, delta_w_veh, delta_w_ped, cambio_de_fase)

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
        observation_class=DefaultObservationFunction # Vector de colas, densidad y fase one-hot
    )
    return env

# 3. Bucle de Ejecución Paralela (Prueba Aleatoria)
if __name__ == '__main__':
    # Rutas al escenario SUMO real del sector de Bogotá modelado
    RED_XML = os.path.join(CARPETA_SUMO, 'Config_actualizado.net.xml')
    RUTAS_XML = ','.join([
        os.path.join(CARPETA_SUMO, 'rutas_aleatorias.rou.xml'),
        os.path.join(CARPETA_SUMO, 'flujos_especificos.rou.xml'),
    ])

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