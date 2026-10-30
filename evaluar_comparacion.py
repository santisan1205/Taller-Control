import argparse
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from setup_entorno import crear_entorno_sumo, CRUCES, RED_XML, rutas_demanda, VENTANAS
from modelo_red_rl import MAPPOActor
from lineas_base import MaxPresion, generar_programas_actuados

RUTAS_XML = rutas_demanda(peatones=False)  # demanda del entorno v1 (la usa demo_en_vivo.py)

RAIZ = os.path.dirname(os.path.abspath(__file__))
CARPETA_EVAL = os.path.join(RAIZ, 'resultados_eval')
# Un juego de archivos por episodio de evaluación (paso a paso, tripinfo,
# estadísticas de SUMO y control semafórico); exportar_datos.py los consolida
CARPETA_EPISODIOS = os.path.join(CARPETA_EVAL, 'episodios')
# Los episodios con el verde máximo aplicado van aparte: los de 'episodios' son
# los reportados hasta ahora, evaluados sin él (sumo-rl no lo aplicaba)
CARPETA_EPISODIOS_MAX_GREEN = os.path.join(CARPETA_EVAL, 'episodios_max_green')
# Entorno v2 (peatones en C1/C4, ventanas valle y pico): siempre con verde máximo
CARPETA_EPISODIOS_V2 = os.path.join(CARPETA_EVAL, 'episodios_v2')
ARCHIVO_ACTUADO = os.path.join(CARPETA_EVAL, 'actuado.add.xml')


def carpeta_episodios(max_green, entorno='v1'):
    if entorno == 'v2':
        return CARPETA_EPISODIOS_V2
    return CARPETA_EPISODIOS_MAX_GREEN if max_green else CARPETA_EPISODIOS

# Políticas entrenadas disponibles -> sufijo de sus .pth
POLITICAS_V1 = {
    'base': '',
    'seed1': '_seed1',
    'seed2': '_seed2',
    'lr_conservador': '_lr_conservador',
    'penaliza_cambio': '_penaliza_cambio',
    'mas_exploracion': '_mas_exploracion',
    'prioriza_espera': '_prioriza_espera',
}
# Campaña v2: cada configuración con 3 semillas de entrenamiento (train.py --tag <config> --seed <s>)
CONFIGS_V2 = ['v2_base', 'v2_penaliza_cambio', 'v2_prioriza_espera', 'v2_peatones_alto',
              'v2_mas_exploracion', 'v2_sin_rnd']
SEMILLAS_ENTRENAMIENTO_V2 = [0, 1, 2]
POLITICAS_V2 = {f'{c}_seed{s}': f'_{c}_seed{s}' for c in CONFIGS_V2 for s in SEMILLAS_ENTRENAMIENTO_V2}
POLITICAS = {**POLITICAS_V1, **POLITICAS_V2}

# Controladores que no se entrenan: el modo coincide con el nombre de la "política"
LINEAS_BASE = {'tiempo_fijo': 'fijo', 'actuado': 'actuado', 'max_pressure': 'max_pressure'}

DELTA_TIME = 5  # segundos simulados por paso del agente (igual que en setup_entorno)


def _cargar_actores(sufijo_pesos, agentes, obs_dims, action_dims):
    actores = {}
    for agent in agentes:
        actor = MAPPOActor(obs_dims[agent], action_dims[agent])
        ruta = os.path.join(RAIZ, f'actor_mappo_{agent}{sufijo_pesos}.pth')
        actor.load_state_dict(torch.load(ruta, map_location='cpu'))
        actor.eval()
        actores[agent] = actor
    return actores


def prefijo_episodio(politica, modo, semilla, escala, max_green=False, entorno='v1', ventana='valle'):
    # En v1 solo existe la ventana valle y se conserva el nombre de siempre
    # (así sirven los episodios ya simulados); en v2 la ventana va en el nombre
    nombre = f'{politica}__{modo}__' + (f'{ventana}__' if entorno == 'v2' else '') + f's{semilla}__x{escala:g}'
    return os.path.join(carpeta_episodios(max_green, entorno), nombre)


def _correr_episodio(env, obs_dict, pasos, actores=None, determinista=False, al_paso=None, politica_reglas=None):
    """
    Corre un episodio completo a partir de un env ya reseteado (obs_dict es el
    resultado de ese reset). Si actores es None, se pasan acciones dummy
    (fixed_ts=True hace que sumo-rl las ignore y deje correr los programas
    semaforicos originales). Si se da un dict de actores, se actua con la
    politica entrenada; si se da politica_reglas (p.ej. MaxPresion), se actua
    con ella. al_paso(env, paso), si se da, se llama después de cada
    paso (lo usa la demo en vivo para su marcador). Devuelve las métricas paso
    a paso de sumo-rl y un resumen del control semafórico por cruce.
    """
    sumo = env.unwrapped.env.sumo
    # Estado verde actual de cada semáforo (se ignoran los amarillos) para
    # contar cambios de fase y duración media del verde, tanto en tiempo fijo
    # como con el agente
    verde_actual = {}
    pasos_en_verde = {}
    cambios = {}

    for paso in range(pasos):
        acciones = {}
        for agent in env.agents:
            if politica_reglas is not None:
                continue  # decide todos los cruces a la vez, abajo
            if actores is None:
                # fixed_ts=True hace que sumo-rl ignore este valor y deje correr
                # el programa semaforico original; igual hay que mandar un valor
                # valido por agente porque el wrapper de PettingZoo lo exige.
                acciones[agent] = 0
            else:
                obs_tensor = torch.tensor(obs_dict[agent], dtype=torch.float32)
                with torch.no_grad():
                    if determinista:
                        # Greedy: solo como dato de referencia, con 100 episodios
                        # la politica tiende a colapsar en una fase
                        accion = actores[agent](obs_tensor).probs.argmax().item()
                    else:
                        # Se muestrea de la distribucion (igual que durante el entrenamiento,
                        # via actor.get_action) en vez de tomar el argmax: con solo 100
                        # episodios la politica no esta lo bastante afinada, y forzar
                        # determinismo la hace colapsar siempre en la misma fase y
                        # matar de inanicion a una direccion (waiting time se dispara).
                        accion, _ = actores[agent].get_action(obs_tensor)
                acciones[agent] = accion
        if politica_reglas is not None:
            acciones = politica_reglas(obs_dict)
        obs_dict, _, terminations, truncations, _ = env.step(acciones)

        for ts in CRUCES:
            estado = sumo.trafficlight.getRedYellowGreenState(ts)
            if 'y' in estado.lower():
                continue
            if estado != verde_actual.get(ts):
                if ts in verde_actual:
                    cambios[ts] = cambios.get(ts, 0) + 1
                verde_actual[ts] = estado
            pasos_en_verde[ts] = pasos_en_verde.get(ts, 0) + 1

        if al_paso is not None:
            al_paso(env, paso)
        if all(terminations.values()) or all(truncations.values()):
            break

    metricas = pd.DataFrame(env.unwrapped.env.metrics)
    env.close()

    control = []
    for ts, cruce in CRUCES.items():
        n_cambios = cambios.get(ts, 0)
        control.append({
            'cruce': cruce,
            'cambios_fase': n_cambios,
            # Duración media de cada verde = tiempo en verde / número de verdes servidos
            'verde_medio_s': pasos_en_verde.get(ts, 0) * DELTA_TIME / (n_cambios + 1),
        })
    return metricas, pd.DataFrame(control)


def _ruta_para_sumo(ruta):
    """
    sumo-rl separa additional_sumo_cmd por espacios, así que una ruta con
    espacios (p.ej. 'Octavo semestre') se rompe. SUMO corre en la carpeta
    actual, así que basta con pasarle la ruta relativa a ella, que dentro del
    proyecto no tiene espacios.
    """
    if ' ' not in ruta:
        return ruta
    try:
        relativa = os.path.relpath(ruta)
    except ValueError:  # otra unidad de disco
        relativa = ruta
    if ' ' in relativa:
        raise ValueError(f'La ruta tiene espacios y SUMO no la acepta: {ruta}. '
                         f'Corre el comando desde la carpeta del proyecto ({RAIZ}).')
    return relativa


def correr_trabajo(politica, modo, semilla, escala, ventana, pasos, forzar=False, max_green=True, entorno='v1'):
    """
    Corre un episodio de evaluación y deja sus archivos en la carpeta de
    episodios del entorno. Si ya existen (y no se pide forzar) no se vuelve a
    simular, para poder retomar una evaluación larga que se interrumpió.
    """
    torch.set_num_threads(1)  # varios episodios corren en paralelo
    prefijo = prefijo_episodio(politica, modo, semilla, escala, max_green, entorno, ventana)
    archivos = [f'{prefijo}.pasos.csv', f'{prefijo}.tripinfo.xml', f'{prefijo}.stats.xml', f'{prefijo}.control.csv']
    if not forzar and all(os.path.exists(a) for a in archivos):
        return prefijo, 'cache'

    os.makedirs(carpeta_episodios(max_green, entorno), exist_ok=True)
    salida = _ruta_para_sumo(prefijo)
    cmd = (f'--tripinfo-output {salida}.tripinfo.xml --tripinfo-output.write-unfinished '
           f'--statistic-output {salida}.stats.xml --device.emissions.probability 1 '
           f'--scale {escala:g}')
    if modo == 'actuado':
        if not os.path.exists(ARCHIVO_ACTUADO):
            generar_programas_actuados(ARCHIVO_ACTUADO)
        cmd += f' --additional-files {_ruta_para_sumo(ARCHIVO_ACTUADO)}'

    # Tiempo fijo y actuado: SUMO corre sus propios programas, sumo-rl no actúa
    programa_sumo = modo in ('tiempo_fijo', 'actuado')
    v2 = entorno == 'v2'
    env = crear_entorno_sumo(RED_XML, rutas_demanda(peatones=v2), use_gui=False, sumo_seed=semilla,
                             out_csv_name=None, fixed_ts=programa_sumo, additional_sumo_cmd=cmd,
                             aplicar_max_green=max_green, peatones=v2, begin_time=VENTANAS[ventana])
    obs, _ = env.reset()
    actores = politica_reglas = None
    if modo == 'max_pressure':
        politica_reglas = MaxPresion(env)
    elif not programa_sumo:
        agentes = env.agents
        obs_dims = {a: env.observation_space(a).shape[0] for a in agentes}
        action_dims = {a: env.action_space(a).n for a in agentes}
        actores = _cargar_actores(POLITICAS[politica], agentes, obs_dims, action_dims)
        # Fija el muestreo de acciones para que la evaluación sea reproducible
        torch.manual_seed(semilla)
    metricas, control = _correr_episodio(env, obs, pasos, actores=actores, determinista=(modo == 'rnd_mappo_greedy'),
                                         politica_reglas=politica_reglas)
    metricas.to_csv(f'{prefijo}.pasos.csv', index=False)
    control.to_csv(f'{prefijo}.control.csv', index=False)
    return prefijo, 'ok'


def _percentil(valores, q):
    return float(np.percentile(valores, q)) if len(valores) else float('nan')


def resumir_episodio(prefijo, duracion_s=3600):
    """
    Lee los archivos de un episodio y devuelve (fila de métricas del sistema,
    filas por cruce). Las métricas de viaje salen de tripinfo (un registro por
    vehículo); las de sistema, del CSV paso a paso de sumo-rl.
    """
    pasos = pd.read_csv(f'{prefijo}.pasos.csv')
    fila = {
        'espera_media': pasos['system_mean_waiting_time'].mean(),
        'espera_total_media': pasos['system_total_waiting_time'].mean(),
        'vehiculos_detenidos_media': pasos['system_total_stopped'].mean(),
        'vehiculos_detenidos_max': pasos['system_total_stopped'].max(),
        'velocidad_media': pasos['system_mean_speed'].mean(),
    }

    terminados, sin_terminar = [], 0
    co2_mg = combustible_mg = distancia_m = 0.0
    for trip in ET.parse(f'{prefijo}.tripinfo.xml').getroot().iter('tripinfo'):
        a = trip.attrib
        emisiones = trip.find('emissions')
        if emisiones is not None:
            co2_mg += float(emisiones.get('CO2_abs', 0))
            combustible_mg += float(emisiones.get('fuel_abs', 0))
        distancia_m += float(a.get('routeLength', 0))
        if float(a.get('arrival', -1)) < 0:
            sin_terminar += 1
            continue
        terminados.append({k: float(a[k]) for k in
                           ('duration', 'timeLoss', 'waitingTime', 'waitingCount', 'departDelay', 'routeLength')})
    viajes = pd.DataFrame(terminados)
    vacio = viajes.empty

    stats = ET.parse(f'{prefijo}.stats.xml').getroot()
    vehiculos = stats.find('vehicles')
    teleports = stats.find('teleports')

    fila.update({
        'viajes_completados': len(viajes),
        'viajes_sin_terminar': sin_terminar,
        'vehiculos_sin_insertar': int(vehiculos.get('waiting', 0)) if vehiculos is not None else np.nan,
        'vehiculos_por_hora': len(viajes) * 3600 / duracion_s,
        'tiempo_viaje_medio': np.nan if vacio else viajes['duration'].mean(),
        'tiempo_viaje_p95': np.nan if vacio else _percentil(viajes['duration'], 95),
        'perdida_tiempo_media': np.nan if vacio else viajes['timeLoss'].mean(),
        'espera_viaje_media': np.nan if vacio else viajes['waitingTime'].mean(),
        'espera_viaje_p95': np.nan if vacio else _percentil(viajes['waitingTime'], 95),
        'paradas_por_vehiculo': np.nan if vacio else viajes['waitingCount'].mean(),
        'demora_insercion_media': np.nan if vacio else viajes['departDelay'].mean(),
        'teleports': int(teleports.get('total', 0)) if teleports is not None else 0,
        'co2_total_kg': co2_mg / 1e6,
        'combustible_total_kg': combustible_mg / 1e6,
        'co2_g_por_km': (co2_mg / 1e3) / (distancia_m / 1e3) if distancia_m else np.nan,
    })

    # Peatones (solo en el entorno v2): un personinfo por persona, con un único
    # walk que cruza el paso peatonal de C1 o C4
    peatones, peatones_sin_terminar = [], 0
    for persona in ET.parse(f'{prefijo}.tripinfo.xml').getroot().iter('personinfo'):
        caminata = persona.find('walk')
        if caminata is None or float(caminata.get('arrival', -1)) < 0:
            peatones_sin_terminar += 1
            continue
        peatones.append({k: float(caminata.get(k)) for k in ('duration', 'waitingTime', 'timeLoss')})
    if peatones or peatones_sin_terminar:
        caminatas = pd.DataFrame(peatones)
        sin_datos = caminatas.empty
        fila.update({
            'peatones_completados': len(caminatas),
            'peatones_sin_terminar': peatones_sin_terminar,
            'peatones_tiempo_viaje_medio': np.nan if sin_datos else caminatas['duration'].mean(),
            'peatones_espera_media': np.nan if sin_datos else caminatas['waitingTime'].mean(),
            'peatones_espera_p95': np.nan if sin_datos else _percentil(caminatas['waitingTime'], 95),
            'peatones_perdida_tiempo_media': np.nan if sin_datos else caminatas['timeLoss'].mean(),
        })

    control = pd.read_csv(f'{prefijo}.control.csv')
    fila['cambios_fase_total'] = control['cambios_fase'].sum()

    por_cruce = []
    for ts, cruce in CRUCES.items():
        c = control[control['cruce'] == cruce].iloc[0]
        por_cruce.append({
            'cruce': cruce,
            'id_sumo': ts,
            'espera_acumulada_media': pasos[f'{ts}_accumulated_waiting_time'].mean(),
            'vehiculos_detenidos_media': pasos[f'{ts}_stopped'].mean(),
            'vehiculos_detenidos_max': pasos[f'{ts}_stopped'].max(),
            'velocidad_media': pasos[f'{ts}_average_speed'].mean(),
            'cambios_fase': c['cambios_fase'],
            'verde_medio_s': c['verde_medio_s'],
        })
    return fila, por_cruce


def trabajos_evaluacion(politicas, semillas, escalas, determinista=False, lineas_base=('tiempo_fijo',),
                        ventanas=('valle',)):
    """Lista de episodios a correr (politica, modo, semilla, escala, ventana).
    Cada línea base se corre una sola vez por semilla x escala x ventana y
    sirve de referencia para todas las políticas."""
    combinaciones = [(s, e, v) for v in ventanas for e in escalas for s in semillas]
    trabajos = [(LINEAS_BASE[m], m, s, e, v) for m in lineas_base for s, e, v in combinaciones]
    modos = ['rnd_mappo'] + (['rnd_mappo_greedy'] if determinista else [])
    trabajos += [(p, m, s, e, v) for p in politicas for m in modos for s, e, v in combinaciones]
    return trabajos


def evaluar(politicas, semillas=(10, 20, 30, 40, 50), escalas=(0.8, 1.0, 1.2), pasos=720,
            determinista=False, procesos=4, forzar=False, max_green=True, entorno='v1',
            lineas_base=('tiempo_fijo',), ventanas=('valle',)):
    validas = POLITICAS_V2 if entorno == 'v2' else POLITICAS_V1
    invalidas = [p for p in politicas if p not in validas]
    if invalidas:
        # La observación de C1/C4 cambia de tamaño entre v1 y v2
        raise ValueError(f'Estas políticas no son del entorno {entorno}: {invalidas}')
    if entorno == 'v1' and set(ventanas) != {'valle'}:
        raise ValueError('El entorno v1 solo tiene la ventana valle')
    trabajos = trabajos_evaluacion(politicas, semillas, escalas, determinista, lineas_base, ventanas)
    print(f'{len(trabajos)} episodios de evaluación (entorno {entorno}), {procesos} en paralelo, '
          f'verde máximo {"aplicado" if max_green else "sin aplicar"}')
    with ProcessPoolExecutor(max_workers=procesos) as pool:
        futuros = {pool.submit(correr_trabajo, *t, pasos, forzar, max_green, entorno): t for t in trabajos}
        for i, futuro in enumerate(as_completed(futuros), 1):
            prefijo, estado = futuro.result()
            print(f'  [{i}/{len(trabajos)}] {os.path.basename(prefijo)} ({estado})')

    filas = []
    for politica, modo, semilla, escala, ventana in trabajos:
        fila, _ = resumir_episodio(prefijo_episodio(politica, modo, semilla, escala, max_green, entorno, ventana))
        filas.append({'politica': politica, 'modo': modo, 'ventana': ventana, 'semilla': semilla,
                      'escala': escala, **fila})
    # El detalle completo (todas las corridas de evaluación en disco) lo
    # consolida exportar_datos.py en datos_dashboard/
    detalle = pd.DataFrame(filas)
    if entorno == 'v2' or 'tiempo_fijo' not in lineas_base:
        return detalle

    # Resumen con el formato anterior (escala 1.0, 3 métricas) por política,
    # para no romper lo que ya consumía resumen_<etiqueta>.csv
    fijo = detalle[(detalle['modo'] == 'tiempo_fijo') & (detalle['escala'] == 1.0)].assign(modo='tiempo_fijo')
    sufijo = '_max_green' if max_green else ''
    for politica in politicas:
        rl = detalle[(detalle['politica'] == politica) & (detalle['modo'] == 'rnd_mappo') & (detalle['escala'] == 1.0)]
        if rl.empty:
            continue
        resumen = pd.concat([fijo, rl])[['semilla', 'modo', 'espera_media', 'vehiculos_detenidos_media', 'velocidad_media']]
        resumen = resumen.sort_values(['semilla', 'modo'], ascending=[True, False])
        resumen.to_csv(os.path.join(CARPETA_EVAL, f'resumen_{politica}{sufijo}.csv'), index=False)
        graficar(resumen, salida=f'comparacion_fijo_vs_rl_{politica}{sufijo}.png')
    return detalle


def graficar(resumen, salida='comparacion_fijo_vs_rl.png'):
    agregada = resumen.drop(columns='semilla').groupby('modo').agg(['mean', 'std'])
    metricas = ['espera_media', 'vehiculos_detenidos_media', 'velocidad_media']
    etiquetas = ['Espera media (s)', 'Vehiculos detenidos (prom.)', 'Velocidad media (m/s)']

    fig, ejes = plt.subplots(1, 3, figsize=(12, 4.5))
    for eje, metrica, etiqueta in zip(ejes, metricas, etiquetas):
        medias = agregada[(metrica, 'mean')]
        errores = agregada[(metrica, 'std')].fillna(0)
        modos = medias.index.tolist()
        colores = ['tab:red' if m == 'tiempo_fijo' else 'tab:blue' for m in modos]
        eje.bar(modos, medias.values, yerr=errores.values, capsize=5, color=colores)
        eje.set_title(etiqueta)
        eje.tick_params(axis='x', rotation=20)

    fig.suptitle('Tiempo fijo vs. RND-MAPPO entrenado (promedio +/- desv. std entre semillas)')
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    print(f"Grafico guardado en {salida}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compara los agentes entrenados contra las lineas base.')
    parser.add_argument('--entorno', choices=['v1', 'v2'], default='v1',
                        help='v1: sin peatones, solo valle (resultados presentados). v2: peatones en C1/C4 y hora pico')
    parser.add_argument('--politicas', nargs='*', default=None, choices=list(POLITICAS),
                        help='Por defecto, todas las del entorno. Sin valores (--politicas) = solo lineas base')
    parser.add_argument('--lineas-base', nargs='+', default=['tiempo_fijo'], choices=list(LINEAS_BASE))
    parser.add_argument('--ventanas', nargs='+', default=['valle'], choices=sorted(VENTANAS))
    parser.add_argument('--semillas', type=int, nargs='+', default=[10, 20, 30, 40, 50])
    parser.add_argument('--escalas', type=float, nargs='+', default=[0.8, 1.0, 1.2],
                        help='Multiplicador de la demanda de SUMO (--scale): prueba la política con menos/más tráfico')
    parser.add_argument('--pasos', type=int, default=720)
    parser.add_argument('--determinista', action='store_true', help='Evalua tambien la politica greedy (argmax)')
    parser.add_argument('--procesos', type=int, default=4, help='Episodios de SUMO en paralelo')
    parser.add_argument('--forzar', action='store_true', help='Re-simula aunque ya existan los archivos del episodio')
    parser.add_argument('--sin-max-green', action='store_true',
                        help='No forzar el verde maximo (reproduce los resultados reportados antes de la correccion)')
    args = parser.parse_args()

    if args.entorno == 'v2' and args.sin_max_green:
        parser.error('El entorno v2 siempre aplica el verde maximo')
    politicas = args.politicas
    if politicas is None:
        politicas = list(POLITICAS_V2 if args.entorno == 'v2' else POLITICAS_V1)
        # Solo las que ya tienen sus pesos en disco (la campaña v2 se reparte entre PCs)
        politicas = [p for p in politicas if os.path.exists(os.path.join(RAIZ, f'critic_mappo{POLITICAS[p]}.pth'))]

    detalle = evaluar(politicas, semillas=args.semillas, escalas=args.escalas, pasos=args.pasos,
                      determinista=args.determinista, procesos=args.procesos, forzar=args.forzar,
                      max_green=not args.sin_max_green, entorno=args.entorno,
                      lineas_base=args.lineas_base, ventanas=args.ventanas)
    print()
    columnas = ['espera_media', 'tiempo_viaje_medio', 'perdida_tiempo_media', 'viajes_completados', 'co2_total_kg']
    columnas += [c for c in ('peatones_espera_media', 'peatones_tiempo_viaje_medio') if c in detalle]
    print(detalle.groupby(['ventana', 'escala', 'politica', 'modo'])[columnas].mean().round(2).to_string())
