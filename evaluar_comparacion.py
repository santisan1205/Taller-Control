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

from setup_entorno import crear_entorno_sumo, CARPETA_SUMO, CRUCES
from modelo_red_rl import MAPPOActor

RED_XML = os.path.join(CARPETA_SUMO, 'Config_actualizado.net.xml')
RUTAS_XML = ','.join([
    os.path.join(CARPETA_SUMO, 'rutas_aleatorias.rou.xml'),
    os.path.join(CARPETA_SUMO, 'flujos_especificos.rou.xml'),
])

RAIZ = os.path.dirname(os.path.abspath(__file__))
CARPETA_EVAL = os.path.join(RAIZ, 'resultados_eval')
# Un juego de archivos por episodio de evaluación (paso a paso, tripinfo,
# estadísticas de SUMO y control semafórico); exportar_datos.py los consolida
CARPETA_EPISODIOS = os.path.join(CARPETA_EVAL, 'episodios')
# Los episodios con el verde máximo aplicado van aparte: los de 'episodios' son
# los reportados hasta ahora, evaluados sin él (sumo-rl no lo aplicaba)
CARPETA_EPISODIOS_MAX_GREEN = os.path.join(CARPETA_EVAL, 'episodios_max_green')


def carpeta_episodios(max_green):
    return CARPETA_EPISODIOS_MAX_GREEN if max_green else CARPETA_EPISODIOS

# Políticas entrenadas disponibles -> sufijo de sus .pth
POLITICAS = {
    'base': '',
    'seed1': '_seed1',
    'seed2': '_seed2',
    'lr_conservador': '_lr_conservador',
    'penaliza_cambio': '_penaliza_cambio',
    'mas_exploracion': '_mas_exploracion',
    'prioriza_espera': '_prioriza_espera',
}

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


def prefijo_episodio(politica, modo, semilla, escala, max_green=False):
    return os.path.join(carpeta_episodios(max_green), f'{politica}__{modo}__s{semilla}__x{escala:g}')


def _correr_episodio(env, obs_dict, pasos, actores=None, determinista=False, al_paso=None):
    """
    Corre un episodio completo a partir de un env ya reseteado (obs_dict es el
    resultado de ese reset). Si actores es None, se pasan acciones dummy
    (fixed_ts=True hace que sumo-rl las ignore y deje correr los programas
    semaforicos originales). Si se da un dict de actores, se actua con la
    politica entrenada. al_paso(env, paso), si se da, se llama después de cada
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


def correr_trabajo(politica, modo, semilla, escala, pasos, forzar=False, max_green=True):
    """
    Corre un episodio de evaluación y deja sus archivos en CARPETA_EPISODIOS.
    Si ya existen (y no se pide forzar) no se vuelve a simular, para poder
    retomar una evaluación larga que se interrumpió.
    """
    torch.set_num_threads(1)  # varios episodios corren en paralelo
    prefijo = prefijo_episodio(politica, modo, semilla, escala, max_green)
    archivos = [f'{prefijo}.pasos.csv', f'{prefijo}.tripinfo.xml', f'{prefijo}.stats.xml', f'{prefijo}.control.csv']
    if not forzar and all(os.path.exists(a) for a in archivos):
        return prefijo, 'cache'

    os.makedirs(carpeta_episodios(max_green), exist_ok=True)
    if ' ' in prefijo:
        raise ValueError(f'sumo-rl separa additional_sumo_cmd por espacios; la ruta no puede tenerlos: {prefijo}')
    cmd = (f'--tripinfo-output {prefijo}.tripinfo.xml --tripinfo-output.write-unfinished '
           f'--statistic-output {prefijo}.stats.xml --device.emissions.probability 1 '
           f'--scale {escala:g}')

    fijo = modo == 'tiempo_fijo'
    env = crear_entorno_sumo(RED_XML, RUTAS_XML, use_gui=False, sumo_seed=semilla, out_csv_name=None,
                             fixed_ts=fijo, additional_sumo_cmd=cmd, aplicar_max_green=max_green)
    obs, _ = env.reset()
    actores = None
    if not fijo:
        agentes = env.agents
        obs_dims = {a: env.observation_space(a).shape[0] for a in agentes}
        action_dims = {a: env.action_space(a).n for a in agentes}
        actores = _cargar_actores(POLITICAS[politica], agentes, obs_dims, action_dims)
        # Fija el muestreo de acciones para que la evaluación sea reproducible
        torch.manual_seed(semilla)
    metricas, control = _correr_episodio(env, obs, pasos, actores=actores, determinista=(modo == 'rnd_mappo_greedy'))
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


def trabajos_evaluacion(politicas, semillas, escalas, determinista=False):
    """Lista de episodios a correr. El tiempo fijo se corre una sola vez por
    semilla x escala y sirve de referencia para todas las políticas."""
    trabajos = [('fijo', 'tiempo_fijo', s, e) for e in escalas for s in semillas]
    modos = ['rnd_mappo'] + (['rnd_mappo_greedy'] if determinista else [])
    trabajos += [(p, m, s, e) for p in politicas for m in modos for e in escalas for s in semillas]
    return trabajos


def evaluar(politicas, semillas=(10, 20, 30, 40, 50), escalas=(0.8, 1.0, 1.2), pasos=720,
            determinista=False, procesos=4, forzar=False, max_green=True):
    trabajos = trabajos_evaluacion(politicas, semillas, escalas, determinista)
    print(f'{len(trabajos)} episodios de evaluación, {procesos} en paralelo, '
          f'verde máximo {"aplicado" if max_green else "sin aplicar"}')
    with ProcessPoolExecutor(max_workers=procesos) as pool:
        futuros = {pool.submit(correr_trabajo, *t, pasos, forzar, max_green): t for t in trabajos}
        for i, futuro in enumerate(as_completed(futuros), 1):
            prefijo, estado = futuro.result()
            print(f'  [{i}/{len(trabajos)}] {os.path.basename(prefijo)} ({estado})')

    filas = []
    for politica, modo, semilla, escala in trabajos:
        fila, _ = resumir_episodio(prefijo_episodio(politica, modo, semilla, escala, max_green))
        filas.append({'politica': politica, 'modo': modo, 'semilla': semilla, 'escala': escala, **fila})
    # El detalle completo (todas las corridas de evaluación en disco) lo
    # consolida exportar_datos.py en datos_dashboard/
    detalle = pd.DataFrame(filas)

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
    parser = argparse.ArgumentParser(description='Compara los agentes entrenados contra semaforos de tiempo fijo.')
    parser.add_argument('--politicas', nargs='+', default=list(POLITICAS), choices=list(POLITICAS))
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

    detalle = evaluar(args.politicas, semillas=args.semillas, escalas=args.escalas, pasos=args.pasos,
                      determinista=args.determinista, procesos=args.procesos, forzar=args.forzar,
                      max_green=not args.sin_max_green)
    print()
    print(detalle.groupby(['escala', 'politica', 'modo'])[
        ['espera_media', 'tiempo_viaje_medio', 'perdida_tiempo_media', 'viajes_completados', 'co2_total_kg']
    ].mean().round(2).to_string())
