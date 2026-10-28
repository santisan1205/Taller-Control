"""
Consolida los resultados de entrenamiento y de evaluación en CSV planos
(formato largo, una fila por observación) dentro de datos_dashboard/, que es
lo que consume el dashboard. El significado y la unidad de cada columna están
en datos_dashboard/diccionario_datos.md.

Uso: python exportar_datos.py   (después de train.py y evaluar_comparacion.py)
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

from analizar_resultados import CORRIDAS, cargar_todas_las_corridas
from evaluar_comparacion import carpeta_episodios, resumir_episodio
from setup_entorno import CRUCES

CARPETA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datos_dashboard')

# Sentido de mejora de cada métrica de evaluación: 'menor' = menos es mejor
SENTIDO_METRICAS = {
    'espera_media': 'menor',
    'espera_total_media': 'menor',
    'vehiculos_detenidos_media': 'menor',
    'vehiculos_detenidos_max': 'menor',
    'velocidad_media': 'mayor',
    'viajes_completados': 'mayor',
    'viajes_sin_terminar': 'menor',
    'vehiculos_sin_insertar': 'menor',
    'vehiculos_por_hora': 'mayor',
    'tiempo_viaje_medio': 'menor',
    'tiempo_viaje_p95': 'menor',
    'perdida_tiempo_media': 'menor',
    'espera_viaje_media': 'menor',
    'espera_viaje_p95': 'menor',
    'paradas_por_vehiculo': 'menor',
    'demora_insercion_media': 'menor',
    'teleports': 'menor',
    'co2_total_kg': 'menor',
    'combustible_total_kg': 'menor',
    'co2_g_por_km': 'menor',
    'cambios_fase_total': 'menor',
}

# Hiperparámetros de las corridas hechas antes de que train.py guardara
# config.json (sacados de los defaults de train.py y de los mensajes de commit)
HIPERPARAMETROS_DEFAULT = {
    'lr_actor': 3e-4, 'lr_critic': 1e-3, 'clip_ratio': 0.2, 'entropy_coef': 0.01, 'ppo_epochs': 4,
    'gamma': 0.99, 'lam': 0.95, 'rnd_lr': 1e-4, 'w1': 1.0, 'w2': 1.0, 'w3': 2.0, 'p_cambio': 0.5,
}
CORRIDAS_HISTORICAS = {
    'base': {'episodios': 100, 'seed': None, 'notas': 'Corrida original, demanda aleatoria'},
    'seed1': {'episodios': 100, 'seed': 1, 'notas': 'Misma config que base, semilla 1'},
    'seed2': {'episodios': 100, 'seed': 2, 'notas': 'Misma config que base, semilla 2'},
    'lr_conservador': {'episodios': 50, 'seed': None, 'lr_actor': 1e-4, 'lr_critic': 3e-4,
                       'notas': 'Learning rates 3x mas lentos'},
    # train.py no pasaba w1/w2/w3/p_cambio al entorno: estas corridas se
    # entrenaron con los pesos por defecto aunque se pidieran otros
    'penaliza_cambio': {'episodios': 50, 'seed': None,
                        'notas': 'Se pidio p_cambio = 2.0, pero no llego al entorno: entreno con 0.5'},
    'mas_exploracion': {'episodios': 50, 'seed': None, 'entropy_coef': np.nan,
                        'notas': 'Mayor coeficiente de entropia (valor por confirmar)'},
    'prioriza_espera': {'episodios': 50, 'seed': None,
                        'notas': 'Se pidieron otros pesos de recompensa (por confirmar), pero no llegaron '
                                 'al entorno: entreno con los pesos por defecto'},
}


def exportar_corridas():
    filas = []
    for corrida, carpeta in CORRIDAS.items():
        ruta_config = os.path.join(carpeta, 'config.json')
        if os.path.exists(ruta_config):
            with open(ruta_config) as f:
                fila = {'corrida': corrida, **json.load(f), 'notas': ''}
        else:
            fila = {'corrida': corrida, **HIPERPARAMETROS_DEFAULT, **CORRIDAS_HISTORICAS.get(corrida, {})}
        filas.append(fila)
    return pd.DataFrame(filas)


def _episodios_evaluados(carpeta):
    """(politica, modo, semilla, escala, prefijo) de cada episodio de evaluación en disco."""
    episodios = []
    for ruta in sorted(glob.glob(os.path.join(carpeta, '*.pasos.csv'))):
        prefijo = ruta[:-len('.pasos.csv')]
        m = re.fullmatch(r'(.+)__(.+)__s(\d+)__x([\d.]+)', os.path.basename(prefijo))
        if m:
            episodios.append((m.group(1), m.group(2), int(m.group(3)), float(m.group(4)), prefijo))
    return episodios


def exportar_evaluacion(carpeta):
    detalle, por_cruce, series = [], [], []
    for politica, modo, semilla, escala, prefijo in _episodios_evaluados(carpeta):
        claves = {'politica': politica, 'modo': modo, 'semilla': semilla, 'escala': escala}
        fila, cruces = resumir_episodio(prefijo)
        detalle.append({**claves, **fila})
        por_cruce += [{**claves, **c} for c in cruces]
        if escala == 1.0:
            pasos = pd.read_csv(f'{prefijo}.pasos.csv')
            series.append(pd.DataFrame({
                **claves,
                'tiempo_s': pasos['step'],
                'espera_media': pasos['system_mean_waiting_time'],
                'vehiculos_detenidos': pasos['system_total_stopped'],
                'velocidad_media': pasos['system_mean_speed'],
                **{f'{cruce}_detenidos': pasos[f'{ts}_stopped'] for ts, cruce in CRUCES.items()},
            }))
    return pd.DataFrame(detalle), pd.DataFrame(por_cruce), pd.concat(series, ignore_index=True)


def resumir_contra_fijo(detalle):
    """
    Por política x modo x escala x métrica: media, desviación, IC95 y mejora
    porcentual frente a tiempo fijo, con una prueba t pareada por semilla
    (ambos corren exactamente la misma demanda).
    """
    fijo = detalle[detalle['modo'] == 'tiempo_fijo'].set_index(['escala', 'semilla'])
    filas = []
    for (politica, modo, escala), grupo in detalle.groupby(['politica', 'modo', 'escala']):
        grupo = grupo.set_index('semilla')
        for metrica, sentido in SENTIDO_METRICAS.items():
            valores = grupo[metrica].dropna()
            n = len(valores)
            media, std = valores.mean(), valores.std(ddof=1) if n > 1 else np.nan
            margen = stats.t.ppf(0.975, n - 1) * std / np.sqrt(n) if n > 1 else np.nan
            fila = {
                'politica': politica, 'modo': modo, 'escala': escala, 'metrica': metrica,
                'mejor_si': sentido, 'n_semillas': n, 'media': media, 'std': std,
                'ic95_inf': media - margen, 'ic95_sup': media + margen,
                'media_fijo': np.nan, 'mejora_pct': np.nan, 'p_valor': np.nan,
            }
            if modo != 'tiempo_fijo':
                ref = fijo.loc[escala, metrica].reindex(valores.index)
                fila['media_fijo'] = ref.mean()
                if ref.mean():
                    signo = 1 if sentido == 'menor' else -1
                    fila['mejora_pct'] = signo * (ref.mean() - media) / abs(ref.mean()) * 100
                diferencias = (valores - ref).dropna()
                if len(diferencias) > 1 and diferencias.std() > 0:
                    fila['p_valor'] = stats.ttest_rel(valores, ref).pvalue
            filas.append(fila)
    return pd.DataFrame(filas)


def main(max_green=False):
    # Por defecto se exporta la evaluación reportada (sin verde máximo) a
    # datos_dashboard/; con max_green, la evaluación corregida a datos_dashboard/max_green/
    salida = os.path.join(CARPETA_SALIDA, 'max_green') if max_green else CARPETA_SALIDA
    os.makedirs(salida, exist_ok=True)

    entrenamiento = cargar_todas_las_corridas()
    entrenamiento.to_csv(os.path.join(salida, 'entrenamiento_episodios.csv'), index=False)
    exportar_corridas().to_csv(os.path.join(salida, 'corridas.csv'), index=False)

    detalle, por_cruce, series = exportar_evaluacion(carpeta_episodios(max_green))
    detalle.to_csv(os.path.join(salida, 'evaluacion_detalle.csv'), index=False)
    por_cruce.to_csv(os.path.join(salida, 'evaluacion_por_cruce.csv'), index=False)
    series.to_csv(os.path.join(salida, 'evaluacion_series_tiempo.csv'), index=False)
    resumen = resumir_contra_fijo(detalle)
    resumen.to_csv(os.path.join(salida, 'evaluacion_resumen.csv'), index=False)

    pd.DataFrame(
        [{'cruce': cruce, 'id_sumo': ts} for ts, cruce in CRUCES.items()]
    ).to_csv(os.path.join(salida, 'cruces.csv'), index=False)

    print(f'Datos exportados en {salida}:')
    for nombre, df in [('entrenamiento_episodios', entrenamiento), ('evaluacion_detalle', detalle),
                       ('evaluacion_por_cruce', por_cruce), ('evaluacion_series_tiempo', series),
                       ('evaluacion_resumen', resumen)]:
        print(f'  {nombre}.csv: {len(df)} filas')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Consolida los resultados en datos_dashboard/.')
    parser.add_argument('--max-green', action='store_true',
                        help='Exporta la evaluación con verde máximo aplicado (a datos_dashboard/max_green/)')
    main(max_green=parser.parse_args().max_green)
