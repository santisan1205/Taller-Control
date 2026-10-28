import os
import sys
import numpy as np
from sumo_rl import parallel_env
from sumo_rl.environment.observations import DefaultObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal

from modelo_red_rl import RewardCalculator

# Carpeta con el escenario SUMO real (red y demanda del sector de Bogotá modelado)
CARPETA_SUMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2026-09-12-17-03-26')

# Nombres cortos de los 4 cruces semaforizados (los IDs de SUMO son muy largos
# para tablas y gráficos). Se usan en los logs de entrenamiento y en los datos
# exportados para el dashboard.
CRUCES = {
    'cluster_10564135163_13571629282_5146149679': 'C1',
    'cluster_10564135205_11749189839_3266988437_5129187741_#3more': 'C2',
    'cluster_10564135211_11749189838_12136544949_12136544951_#7more': 'C3',
    'cluster_10564135212_13571629279_13571629299': 'C4',
}

def nombre_cruce(agent):
    return CRUCES.get(agent, agent)

MAX_GREEN = 60  # Verde máximo (s) antes de forzar el cambio de fase

# sumo-rl 1.4.5 recibe max_green pero no lo aplica: TrafficSignal.set_next_phase
# solo impone el verde mínimo, así que un agente puede sostener una fase la hora
# entera (pasaba en C4). Se envuelve para que, si el entorno lo pide
# (aplicar_max_green), una fase que ya cumplió MAX_GREEN pase a la siguiente
# aunque el agente pida mantenerla. Como el agente decide cada delta_time, el
# verde puede pasarse hasta delta_time - 1 s de MAX_GREEN. No afecta al tiempo
# fijo: con fixed_ts sumo-rl nunca llama a set_next_phase.
_set_next_phase_sumo_rl = TrafficSignal.set_next_phase

def _set_next_phase_con_verde_maximo(self, new_phase):
    new_phase = int(new_phase)
    if (getattr(self.env, 'aplicar_max_green', False)
            and new_phase == self.green_phase
            and self.time_since_last_phase_change >= self.yellow_time + self.max_green):
        new_phase = (self.green_phase + 1) % self.num_green_phases
    _set_next_phase_sumo_rl(self, new_phase)

TrafficSignal.set_next_phase = _set_next_phase_con_verde_maximo

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
def crear_recompensa(reward_calc):
    """
    Devuelve la función de recompensa que sumo-rl llama por semáforo, usando
    los pesos (w1/w2/w3) y la penalización por cambio de fase de reward_calc.
    Antes se usaba un RewardCalculator() fijo a nivel de módulo, así que los
    pesos que train.py recibía por línea de comandos nunca llegaban al entorno.
    """
    def recompensa_delta_colas(traffic_signal):
        return _recompensa_delta_colas(traffic_signal, reward_calc)
    return recompensa_delta_colas

def _recompensa_delta_colas(traffic_signal, reward_calc):
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

    return reward_calc.calculate_extrinsic_reward(delta_q, delta_w_veh, delta_w_ped, cambio_de_fase)

# 2. Configuración del Entorno Multi-Agente
def crear_entorno_sumo(net_file, route_file, use_gui=False, sumo_seed='random',
                        out_csv_name='resultados/mappo_rnd_train', fixed_ts=False,
                        additional_sumo_cmd=None, reward_calc=None, aplicar_max_green=True):
    """
    Inicializa el entorno PettingZoo con la Selección Directa de Fase.

    sumo_seed: fija la demanda de tráfico (útil para comparar corridas de forma justa
    y para repetir un mismo escenario con distintas semillas de entrenamiento).
    fixed_ts: si es True, ignora las acciones del agente y deja correr los programas
    semafóricos originales del .net.xml (tiempo fijo) — sirve como baseline de comparación.
    additional_sumo_cmd: opciones extra de línea de comandos para SUMO (p.ej.
    --tripinfo-output o --scale), usadas por la evaluación para medir más métricas.
    reward_calc: RewardCalculator con los pesos de R_ext; None = valores por defecto.
    aplicar_max_green: fuerza el cambio de fase tras MAX_GREEN s de verde. Las
    corridas anteriores a esta corrección se entrenaron y evaluaron sin él.
    """
    if reward_calc is None:
        reward_calc = RewardCalculator()
    env = parallel_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv_name,
        use_gui=use_gui,
        num_seconds=3600,       # Duración de la simulación (1 hora en segundos de SUMO)
        delta_time=5,           # Δt: El agente toma decisiones cada 5 segundos
        yellow_time=3,          # Tiempo amarillo obligatorio (Transición)
        min_green=15,           # Tiempo verde mínimo estricto
        max_green=MAX_GREEN,    # Tiempo verde máximo (lo aplica _set_next_phase_con_verde_maximo)
        reward_fn=crear_recompensa(reward_calc), # Inyección de nuestra R_ext
        observation_class=DefaultObservationFunction, # Vector de colas, densidad y fase one-hot
        sumo_seed=sumo_seed,
        fixed_ts=fixed_ts,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    env.unwrapped.env.aplicar_max_green = aplicar_max_green
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