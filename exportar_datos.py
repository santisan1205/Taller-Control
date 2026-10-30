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

from analizar_resultados import CORRIDAS, cargar_todas_las_corridas, corridas_v2
from evaluar_comparacion import carpeta_episodios, resumir_episodio, LINEAS_BASE
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
    # Solo en el entorno v2 (peatones en C1/C4)
    'peatones_completados': 'mayor',
    'peatones_sin_terminar': 'menor',
    'peatones_tiempo_viaje_medio': 'menor',
    'peatones_espera_media': 'menor',
    'peatones_espera_p95': 'menor',
    'peatones_perdida_tiempo_media': 'menor',
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


def exportar_corridas(corridas=CORRIDAS):
    filas = []
    for corrida, carpeta in corridas.items():
        ruta_config = os.path.join(carpeta, 'config.json')
        if os.path.exists(ruta_config):
            with open(ruta_config) as f:
                fila = {'corrida': corrida, **json.load(f), 'notas': ''}
        else:
            fila = {'corrida': corrida, **HIPERPARAMETROS_DEFAULT, **CORRIDAS_HISTORICAS.get(corrida, {})}
        filas.append(fila)
    return pd.DataFrame(filas)


def _episodios_evaluados(carpeta):
    """(claves, prefijo) de cada episodio de evaluación en disco. Las claves son
    politica, modo, semilla, escala y, en el entorno v2, ventana."""
    episodios = []
    for ruta in sorted(glob.glob(os.path.join(carpeta, '*.pasos.csv'))):
        prefijo = ruta[:-len('.pasos.csv')]
        m = re.fullmatch(r'(?P<politica>.+?)__(?P<modo>.+?)__(?:(?P<ventana>valle|pico)__)?'
                         r's(?P<semilla>\d+)__x(?P<escala>[\d.]+)', os.path.basename(prefijo))
        if m:
            claves = {'politica': m['politica'], 'modo': m['modo'], 'semilla': int(m['semilla']),
                      'escala': float(m['escala'])}
            if m['ventana']:
                claves = {'ventana': m['ventana'], **claves}
            episodios.append((claves, prefijo))
    return episodios


def exportar_evaluacion(carpeta):
    detalle, por_cruce, series = [], [], []
    for claves, prefijo in _episodios_evaluados(carpeta):
        escala = claves['escala']
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


def _media_e_ic95(valores):
    n = len(valores)
    media, std = valores.mean(), valores.std(ddof=1) if n > 1 else np.nan
    margen = stats.t.ppf(0.975, n - 1) * std / np.sqrt(n) if n > 1 else np.nan
    return n, media, std, media - margen, media + margen


def _mejora_pct(media, media_ref, sentido):
    """Mejora frente a la referencia en %: positivo = mejor, según el sentido de la métrica."""
    if not media_ref or pd.isna(media_ref):
        return np.nan
    signo = 1 if sentido == 'menor' else -1
    return signo * (media_ref - media) / abs(media_ref) * 100


def resumir_contra(detalle, modo_ref='tiempo_fijo', grupos=('escala',)):
    """
    Por política x modo x grupos x métrica: media, desviación, IC95 y mejora
    porcentual frente a la línea base modo_ref, con una prueba t pareada por
    semilla (ambos corren exactamente la misma demanda).
    """
    grupos = list(grupos)
    ref_df = detalle[detalle['modo'] == modo_ref].set_index(grupos + ['semilla'])
    metricas = {m: s for m, s in SENTIDO_METRICAS.items() if m in detalle.columns}
    filas = []
    for claves, grupo in detalle.groupby(['politica', 'modo'] + grupos):
        politica, modo, *valores_grupo = claves
        grupo = grupo.set_index('semilla')
        for metrica, sentido in metricas.items():
            valores = grupo[metrica].dropna()
            n, media, std, ic_inf, ic_sup = _media_e_ic95(valores)
            fila = {
                'politica': politica, 'modo': modo, **dict(zip(grupos, valores_grupo)), 'metrica': metrica,
                'mejor_si': sentido, 'n_semillas': n, 'media': media, 'std': std,
                'ic95_inf': ic_inf, 'ic95_sup': ic_sup,
                'media_ref': np.nan, 'mejora_pct': np.nan, 'p_valor': np.nan,
            }
            clave_ref = tuple(valores_grupo) if len(valores_grupo) > 1 else valores_grupo[0]
            if modo != modo_ref and clave_ref in ref_df.index.droplevel('semilla'):
                ref = ref_df.loc[clave_ref, metrica].reindex(valores.index)
                fila['media_ref'] = ref.mean()
                fila['mejora_pct'] = _mejora_pct(media, ref.mean(), sentido)
                diferencias = (valores - ref).dropna()
                if len(diferencias) > 1 and diferencias.std() > 0:
                    fila['p_valor'] = stats.ttest_rel(valores, ref).pvalue
            filas.append(fila)
    return pd.DataFrame(filas)


def resumir_contra_fijo(detalle):
    """Formato de la evaluación v1 (evaluacion_resumen.csv): solo contra tiempo fijo, por escala."""
    return resumir_contra(detalle, 'tiempo_fijo', ['escala']).rename(columns={'media_ref': 'media_fijo'})


def resumir_configs_v2(detalle):
    """
    Una fila por configuración x ventana x escala x métrica x línea base. Cada
    semilla de entrenamiento se promedia primero sobre las semillas de SUMO;
    la media, el IC95 y la prueba t (de una muestra, contra la media de la
    línea base) son entre semillas de entrenamiento, que es la variabilidad
    que importa para decir si la configuración aprende de forma confiable.
    """
    rl = detalle[detalle['modo'] == 'rnd_mappo'].copy()
    if rl.empty:
        return pd.DataFrame()
    rl[['config', 'semilla_entrenamiento']] = rl['politica'].str.extract(r'^(.+)_seed(\d+)$')
    metricas = {m: s for m, s in SENTIDO_METRICAS.items() if m in detalle.columns}
    por_semilla = rl.groupby(['config', 'semilla_entrenamiento', 'ventana', 'escala'])[list(metricas)].mean()
    bases = detalle[detalle['modo'].isin(LINEAS_BASE)].groupby(['modo', 'ventana', 'escala'])[list(metricas)].mean()

    filas = []
    for (config, ventana, escala), grupo in por_semilla.groupby(level=['config', 'ventana', 'escala']):
        for metrica, sentido in metricas.items():
            valores = grupo[metrica].dropna()
            n, media, std, ic_inf, ic_sup = _media_e_ic95(valores)
            for linea_base in bases.index.get_level_values('modo').unique():
                media_ref = bases[metrica].get((linea_base, ventana, escala), np.nan)
                p_valor = np.nan
                if n > 1 and valores.std() > 0 and not pd.isna(media_ref):
                    p_valor = stats.ttest_1samp(valores, media_ref).pvalue
                filas.append({
                    'config': config, 'ventana': ventana, 'escala': escala, 'metrica': metrica,
                    'mejor_si': sentido, 'n_semillas_entrenamiento': n, 'media': media, 'std': std,
                    'ic95_inf': ic_inf, 'ic95_sup': ic_sup, 'referencia': linea_base,
                    'media_ref': media_ref, 'mejora_pct': _mejora_pct(media, media_ref, sentido),
                    'p_valor': p_valor,
                })
    return pd.DataFrame(filas)


def main_v2():
    """Campaña v2 (peatones en C1/C4, valle y hora pico, 3 líneas base) -> datos_dashboard/v2/."""
    salida = os.path.join(CARPETA_SALIDA, 'v2')
    os.makedirs(salida, exist_ok=True)

    corridas = corridas_v2()
    tablas = {}
    if corridas:
        entrenamiento = cargar_todas_las_corridas(corridas)
        entrenamiento[['config', 'semilla_entrenamiento']] = entrenamiento['corrida'].str.extract(r'^(.+)_seed(\d+)$')
        tablas['entrenamiento_episodios'] = entrenamiento
        tablas['corridas'] = exportar_corridas(corridas)

    detalle, por_cruce, series = exportar_evaluacion(carpeta_episodios(True, 'v2'))
    tablas.update({'evaluacion_detalle': detalle, 'evaluacion_por_cruce': por_cruce,
                   'evaluacion_series_tiempo': series})
    referencias = [m for m in LINEAS_BASE if m in set(detalle['modo'])]
    tablas['evaluacion_resumen'] = pd.concat(
        [resumir_contra(detalle, ref, ['ventana', 'escala']).assign(referencia=ref) for ref in referencias],
        ignore_index=True)
    tablas['evaluacion_configs'] = resumir_configs_v2(detalle)
    tablas['cruces'] = pd.DataFrame([{'cruce': cruce, 'id_sumo': ts} for ts, cruce in CRUCES.items()])

    print(f'Datos exportados en {salida}:')
    for nombre, df in tablas.items():
        df.to_csv(os.path.join(salida, f'{nombre}.csv'), index=False)
        print(f'  {nombre}.csv: {len(df)} filas')


def main(max_green=False):
    # Por defecto se exporta la evaluación reportada (sin verde máximo) a
    # datos_dashboard/; con max_green, la evaluación corregida a datos_dashboard/max_green/
    salida = os.path.join(CARPETA_SALIDA, 'max_green') if max_green else CARPETA_SALIDA
    os.makedirs(salida, exist_ok=True)

    entrenamiento = cargar_todas_las_corridas()
    entrenamiento.to_csv(os.path.join(salida, 'entrenamiento_episodios.csv'), index=False)
    exportar_corridas().to_csv(os.path.join(salida, 'corridas.csv'), index=False)

    detalle, por_cruce, series = exportar_evaluacion(carpeta_episodios(max_green))
    detalle = detalle[detalle['modo'].isin(['tiempo_fijo', 'rnd_mappo', 'rnd_mappo_greedy'])]
    por_cruce = por_cruce[por_cruce['modo'].isin(detalle['modo'].unique())]
    series = series[series['modo'].isin(detalle['modo'].unique())]
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
    parser.add_argument('--v2', action='store_true',
                        help='Exporta la campaña v2 (peatones + hora pico + 3 líneas base) a datos_dashboard/v2/')
    args = parser.parse_args()
    if args.v2:
        main_v2()
    else:
        main(max_green=args.max_green)
