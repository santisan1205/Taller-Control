"""
Líneas base de control semafórico para comparar con RND-MAPPO, además del
tiempo fijo original:

- actuado: los programas actuados de SUMO (extienden el verde mientras sus
  detectores ven vehículos llegando), con las mismas fases que el tiempo fijo
  y verde entre MIN_GREEN y MAX_GREEN. Se cargan como archivo adicional y se
  corren con fixed_ts=True.
- max-pressure: en cada decisión elige la fase verde con mayor presión (cola
  en los carriles de entrada menos cola en los de salida de los movimientos
  que la fase habilita). Actúa a través de sumo-rl igual que el agente, así
  que respeta el mismo verde mínimo, amarillo y verde máximo.
"""
import os
import xml.etree.ElementTree as ET

from setup_entorno import CRUCES, MAX_GREEN, RED_XML

MIN_GREEN = 15  # igual que min_green en crear_entorno_sumo
PROGRAMA_ACTUADO = 'actuado'
# Fases más cortas que esto (despejes peatonales de 5 s) se dejan fijas
DURACION_MIN_ACTUABLE = 10


def generar_programas_actuados(salida, red=RED_XML):
    """
    Escribe un .add.xml con una copia actuada de cada tlLogic de los 4 cruces.
    Como es el último programa que SUMO carga para cada semáforo, queda activo.
    """
    raiz = ET.Element('additional')
    for tl in ET.parse(red).getroot().iter('tlLogic'):
        if tl.get('id') not in CRUCES:
            continue
        nuevo = ET.SubElement(raiz, 'tlLogic', id=tl.get('id'), type='actuated',
                              programID=PROGRAMA_ACTUADO, offset='0')
        for fase in tl.iter('phase'):
            atributos = {'duration': fase.get('duration'), 'state': fase.get('state')}
            es_verde = 'y' not in fase.get('state').lower()
            if es_verde and float(fase.get('duration')) >= DURACION_MIN_ACTUABLE:
                atributos.update(minDur=str(MIN_GREEN), maxDur=str(MAX_GREEN))
            ET.SubElement(nuevo, 'phase', **atributos)
    ET.indent(raiz)
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    ET.ElementTree(raiz).write(salida, encoding='utf-8', xml_declaration=True)
    return salida


class MaxPresion:
    """
    Política de reglas max-pressure. Se construye después de env.reset(),
    porque sumo-rl recrea los semáforos (y sus fases verdes) en cada reset.
    Solo cuenta vehículos: los carriles de peatones (walkingareas) no entran
    en la presión.
    """

    def __init__(self, env):
        self.env = env
        self.movimientos = {}  # ts_id -> [movimientos (carril_entrada, carril_salida) por fase verde]
        for ts_id, ts in env.unwrapped.env.traffic_signals.items():
            enlaces = ts.sumo.trafficlight.getControlledLinks(ts_id)
            por_fase = []
            for fase in ts.green_phases:
                movimientos = {
                    (entrada, salida)
                    for i, estado in enumerate(fase.state) if estado in 'Gg'
                    for entrada, salida, _ in enlaces[i]
                    if not entrada.startswith(':')
                }
                por_fase.append(sorted(movimientos))
            self.movimientos[ts_id] = por_fase

    def __call__(self, obs_dict=None):
        base = self.env.unwrapped.env
        carril = base.sumo.lane
        acciones = {}
        for ts_id in self.env.agents:
            presiones = [
                sum(carril.getLastStepHaltingNumber(e) - carril.getLastStepHaltingNumber(s) for e, s in movs)
                for movs in self.movimientos[ts_id]
            ]
            actual = base.traffic_signals[ts_id].green_phase
            mejor = max(range(len(presiones)), key=lambda p: presiones[p])
            # Si ninguna fase gana a la actual, se mantiene (evita cambios inútiles)
            acciones[ts_id] = actual if presiones[actual] >= presiones[mejor] else mejor
        return acciones
