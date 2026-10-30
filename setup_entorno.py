import os
import sys
import numpy as np
from gymnasium import spaces
from sumo_rl import parallel_env
from sumo_rl.environment.observations import DefaultObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal

from modelo_red_rl import RewardCalculator

# Carpeta con el escenario SUMO real (red y demanda del sector de Bogotá modelado)
CARPETA_SUMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2026-09-12-17-03-26')
RED_XML = os.path.join(CARPETA_SUMO, 'Config_actualizado.net.xml')
_RUTAS_VEHICULOS = [
    os.path.join(CARPETA_SUMO, 'rutas_aleatorias.rou.xml'),
    os.path.join(CARPETA_SUMO, 'flujos_especificos.rou.xml'),
]
RUTA_PEATONES = os.path.join(CARPETA_SUMO, 'peatones.rou.xml')  # la genera generar_peatones.py


def rutas_demanda(peatones=False):
    """Archivos de demanda separados por comas, como los espera sumo-rl."""
    return ','.join(_RUTAS_VEHICULOS + ([RUTA_PEATONES] if peatones else []))


# Ventanas de simulación de 1 hora (inicio en s de SUMO). La demanda corre de 0
# a 8000 s y la hora pico (flujos de 5000-7000 s) queda fuera de la ventana de
# entrenamiento original; los 600 s previos a la hora pico sirven de llenado de
# la red, igual que pasa en el valle desde t = 0.
VENTANAS = {'valle': 0, 'pico': 4400}
DURACION_EPISODIO = 3600

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

# Único paso peatonal semaforizado de cada cruce que tiene peatones (C1 y C4;
# C2 y C3 no tienen): el crossing y las dos walkingareas donde los peatones
# esperan el verde. Ver generar_peatones.py.
def _paso_peatonal(cluster):
    return {f':{cluster}_c0', f':{cluster}_w0', f':{cluster}_w1'}

PASOS_PEATONALES = {
    'cluster_10564135163_13571629282_5146149679': _paso_peatonal('cluster_10564135163_13571629282_5146149679'),
    'cluster_10564135212_13571629279_13571629299': _paso_peatonal('cluster_10564135212_13571629279_13571629299'),
}

def _peatones_esperando(traffic_signal):
    """
    Tiempos de espera de los peatones que están en el paso peatonal del
    semáforo (vacío si el cruce no tiene peatones). sumo-rl no trackea
    peatones, así que se consulta TraCI directamente. Se recorre la lista de
    personas en vez de edge.getLastStepPersonIDs, que no devuelve nada en las
    walkingareas (edges internos).
    """
    edges = PASOS_PEATONALES.get(traffic_signal.id)
    if not edges:
        return []
    sumo = traffic_signal.sumo
    return [sumo.person.getWaitingTime(p) for p in sumo.person.getIDList() if sumo.person.getRoadID(p) in edges]

def _espera_peatonal(traffic_signal):
    """Suma el tiempo de espera de los peatones en el paso peatonal del semáforo."""
    return float(sum(_peatones_esperando(traffic_signal)))


class ObservacionConPeatones(DefaultObservationFunction):
    """
    Observación por defecto de sumo-rl (fase one-hot, verde mínimo cumplido,
    densidad y cola por carril) más, solo en los cruces con paso peatonal, dos
    valores en [0, 1]: peatones esperando (saturado en 10) y la espera del
    peatón que más lleva (saturada en 60 s). Sin esto el agente no ve a los
    peatones y w3 solo agrega ruido a la recompensa.
    """
    PEATONES_MAX = 10
    ESPERA_MAX = 60.0

    def _tiene_peatones(self):
        return self.ts.id in PASOS_PEATONALES

    def __call__(self):
        observacion = super().__call__()
        if not self._tiene_peatones():
            return observacion
        esperas = [e for e in _peatones_esperando(self.ts) if e > 0]
        extra = [min(1.0, len(esperas) / self.PEATONES_MAX), min(1.0, max(esperas, default=0.0) / self.ESPERA_MAX)]
        return np.concatenate([observacion, np.array(extra, dtype=np.float32)])

    def observation_space(self):
        espacio = super().observation_space()
        if not self._tiene_peatones():
            return espacio
        n = espacio.shape[0] + 2
        return spaces.Box(low=np.zeros(n, dtype=np.float32), high=np.ones(n, dtype=np.float32))

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
                        additional_sumo_cmd=None, reward_calc=None, aplicar_max_green=True,
                        peatones=False, begin_time=0):
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
    peatones: agrega a la observación de C1 y C4 su paso peatonal (entorno v2).
    La demanda peatonal la decide route_file (ver rutas_demanda). Las políticas
    v1 se entrenaron sin esto y su observación es más corta.
    begin_time: inicio de la ventana en s de SUMO (ver VENTANAS); se puede
    cambiar entre episodios con configurar_episodio.
    """
    if reward_calc is None:
        reward_calc = RewardCalculator()
    env = parallel_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv_name,
        use_gui=use_gui,
        begin_time=begin_time,
        num_seconds=DURACION_EPISODIO,  # Duración de la simulación (1 hora en segundos de SUMO)
        delta_time=5,           # Δt: El agente toma decisiones cada 5 segundos
        yellow_time=3,          # Tiempo amarillo obligatorio (Transición)
        min_green=15,           # Tiempo verde mínimo estricto
        max_green=MAX_GREEN,    # Tiempo verde máximo (lo aplica _set_next_phase_con_verde_maximo)
        reward_fn=crear_recompensa(reward_calc), # Inyección de nuestra R_ext
        # Vector de colas, densidad y fase one-hot (+ paso peatonal en el entorno v2)
        observation_class=ObservacionConPeatones if peatones else DefaultObservationFunction,
        sumo_seed=sumo_seed,
        fixed_ts=fixed_ts,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    env.unwrapped.env.aplicar_max_green = aplicar_max_green
    return env

def configurar_episodio(env, begin_time=None, escala=None):
    """
    Cambia la ventana y/o la escala de la demanda del próximo episodio (se
    aplican en el siguiente env.reset(), que relanza SUMO). sumo-rl calcula
    sim_max_time una sola vez en el constructor, así que hay que moverlo
    junto con begin_time. La escala se pasa como --scale de SUMO y reemplaza
    cualquier --scale previo en additional_sumo_cmd.
    """
    base = env.unwrapped.env
    if begin_time is not None:
        base.begin_time = begin_time
        base.sim_max_time = begin_time + DURACION_EPISODIO
    if escala is not None:
        partes = (base.additional_sumo_cmd or '').split()
        if '--scale' in partes:
            i = partes.index('--scale')
            del partes[i:i + 2]
        base.additional_sumo_cmd = ' '.join(partes + ['--scale', f'{escala:g}'])

# 3. Bucle de Ejecución Paralela (Prueba Aleatoria)
if __name__ == '__main__':
    env = crear_entorno_sumo(RED_XML, rutas_demanda(peatones=True), use_gui=True, peatones=True)
    
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