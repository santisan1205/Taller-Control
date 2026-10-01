import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from setup_entorno import crear_entorno_sumo, CARPETA_SUMO
from modelo_red_rl import MAPPOActor

RED_XML = os.path.join(CARPETA_SUMO, 'Config_actualizado.net.xml')
RUTAS_XML = ','.join([
    os.path.join(CARPETA_SUMO, 'rutas_aleatorias.rou.xml'),
    os.path.join(CARPETA_SUMO, 'flujos_especificos.rou.xml'),
])


def _cargar_actores(sufijo_pesos, agentes, obs_dims, action_dims):
    actores = {}
    for agent in agentes:
        actor = MAPPOActor(obs_dims[agent], action_dims[agent])
        ruta = f'actor_mappo_{agent}{sufijo_pesos}.pth'
        actor.load_state_dict(torch.load(ruta, map_location='cpu'))
        actor.eval()
        actores[agent] = actor
    return actores


def _correr_episodio(env, obs_dict, pasos, actores=None, ruta_csv=None):
    """
    Corre un episodio completo a partir de un env ya reseteado (obs_dict es el
    resultado de ese reset). Si actores es None, se pasan acciones dummy
    (fixed_ts=True hace que sumo-rl las ignore y deje correr los programas
    semaforicos originales). Si se da un dict de actores, se actua muestreando
    de la politica entrenada. Como aqui solo corre un episodio por env (no hay
    un reset() posterior que dispare el guardado automatico de sumo-rl), el
    CSV se escribe directamente si se pasa ruta_csv.
    """
    for _ in range(pasos):
        acciones = {}
        for agent in env.agents:
            if actores is None:
                # fixed_ts=True hace que sumo-rl ignore este valor y deje correr
                # el programa semaforico original; igual hay que mandar un valor
                # valido por agente porque el wrapper de PettingZoo lo exige.
                acciones[agent] = 0
            else:
                # Se muestrea de la distribucion (igual que durante el entrenamiento,
                # via actor.get_action) en vez de tomar el argmax: con solo 100
                # episodios la politica no esta lo bastante afinada, y forzar
                # determinismo la hace colapsar siempre en la misma fase y
                # matar de inanicion a una direccion (waiting time se dispara).
                obs_tensor = torch.tensor(obs_dict[agent], dtype=torch.float32)
                with torch.no_grad():
                    accion, _ = actores[agent].get_action(obs_tensor)
                acciones[agent] = accion
        obs_dict, _, terminations, truncations, _ = env.step(acciones)
        if all(terminations.values()) or all(truncations.values()):
            break

    entorno_sumo = env.unwrapped.env
    metricas = pd.DataFrame(entorno_sumo.metrics)
    env.close()
    if ruta_csv is not None:
        os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
        metricas.to_csv(ruta_csv, index=False)
    return metricas


def evaluar(sufijo_pesos='', semillas=(10, 20, 30), pasos=720, use_gui=False):
    # Etiqueta para no pisar los resultados de otra politica (distinto sufijo_pesos)
    # al guardar los CSV de esta evaluacion.
    etiqueta = sufijo_pesos.lstrip('_') or 'base'
    carpeta_eval = f'resultados_eval/{etiqueta}'

    filas = []
    detalle = {}

    for semilla in semillas:
        print(f"--- Semilla de demanda {semilla} ---")

        # Baseline: semaforos de tiempo fijo (los programas originales del .net.xml)
        env_fijo = crear_entorno_sumo(
            RED_XML, RUTAS_XML, use_gui=use_gui, sumo_seed=semilla, out_csv_name=None, fixed_ts=True,
        )
        obs_fijo, _ = env_fijo.reset()
        m_fijo = _correr_episodio(env_fijo, obs_fijo, pasos, actores=None,
                                   ruta_csv=f'{carpeta_eval}/fijo_seed{semilla}.csv')
        print("  Tiempo fijo:    espera media =", round(m_fijo['system_mean_waiting_time'].mean(), 2))

        # Agente entrenado, actuando muestreando de la politica ya aprendida
        env_rl = crear_entorno_sumo(
            RED_XML, RUTAS_XML, use_gui=use_gui, sumo_seed=semilla, out_csv_name=None, fixed_ts=False,
        )
        obs_rl, _ = env_rl.reset()
        agentes = env_rl.agents
        obs_dims = {a: env_rl.observation_space(a).shape[0] for a in agentes}
        action_dims = {a: env_rl.action_space(a).n for a in agentes}
        actores = _cargar_actores(sufijo_pesos, agentes, obs_dims, action_dims)
        m_rl = _correr_episodio(env_rl, obs_rl, pasos, actores=actores,
                                 ruta_csv=f'{carpeta_eval}/entrenado_seed{semilla}.csv')
        print("  Agente RND-MAPPO: espera media =", round(m_rl['system_mean_waiting_time'].mean(), 2))

        detalle[semilla] = {'fijo': m_fijo, 'entrenado': m_rl}
        filas.append({
            'semilla': semilla, 'modo': 'tiempo_fijo',
            'espera_media': m_fijo['system_mean_waiting_time'].mean(),
            'vehiculos_detenidos_media': m_fijo['system_total_stopped'].mean(),
            'velocidad_media': m_fijo['system_mean_speed'].mean(),
        })
        filas.append({
            'semilla': semilla, 'modo': 'rnd_mappo',
            'espera_media': m_rl['system_mean_waiting_time'].mean(),
            'vehiculos_detenidos_media': m_rl['system_total_stopped'].mean(),
            'velocidad_media': m_rl['system_mean_speed'].mean(),
        })

    resumen = pd.DataFrame(filas)
    os.makedirs('resultados_eval', exist_ok=True)
    resumen.to_csv(f'resultados_eval/resumen_{etiqueta}.csv', index=False)
    return resumen, detalle


def graficar(resumen, salida='comparacion_fijo_vs_rl.png'):
    agregada = resumen.groupby('modo').agg(['mean', 'std'])
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
    print(f"Grafico guardado en {salida}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compara el agente entrenado contra semaforos de tiempo fijo.')
    parser.add_argument('--sufijo-pesos', default='', help='Sufijo de los .pth a cargar (p.ej. _seed0). Vacio = corrida original sin semilla.')
    parser.add_argument('--semillas', type=int, nargs='+', default=[10, 20, 30])
    parser.add_argument('--pasos', type=int, default=720)
    args = parser.parse_args()

    resumen, _ = evaluar(sufijo_pesos=args.sufijo_pesos, semillas=args.semillas, pasos=args.pasos)
    print()
    print(resumen.to_string(index=False))
    print()
    print(resumen.groupby('modo')[['espera_media', 'vehiculos_detenidos_media', 'velocidad_media']].mean())
    etiqueta = args.sufijo_pesos.lstrip('_') or 'base'
    graficar(resumen, salida=f'comparacion_fijo_vs_rl_{etiqueta}.png')
