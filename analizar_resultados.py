import glob
import os
import re

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from setup_entorno import CRUCES

CARPETA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')

# Corridas de entrenamiento -> carpeta donde sumo-rl dejó sus CSV
CORRIDAS = {
    'base': CARPETA_RESULTADOS,
    'seed1': os.path.join(CARPETA_RESULTADOS, 'seed1'),
    'seed2': os.path.join(CARPETA_RESULTADOS, 'seed2'),
    'lr_conservador': os.path.join(CARPETA_RESULTADOS, 'lr_conservador'),
    'penaliza_cambio': os.path.join(CARPETA_RESULTADOS, 'penaliza_cambio'),
    'mas_exploracion': os.path.join(CARPETA_RESULTADOS, 'mas_exploracion'),
    'prioriza_espera': os.path.join(CARPETA_RESULTADOS, 'prioriza_espera'),
}


def corridas_v2():
    """Corridas de la campaña v2 que hay en disco (train.py --tag v2_<config> --seed <s>)."""
    carpetas = sorted(glob.glob(os.path.join(CARPETA_RESULTADOS, 'v2_*_seed*')))
    return {os.path.basename(c): c for c in carpetas if os.path.isdir(c)}

# Un episodio real tiene ~720 filas; el CSV del reset() inicial tiene solo 1
MIN_FILAS_EPISODIO = 10


def cargar_episodios(carpeta=CARPETA_RESULTADOS):
    """
    Carga todos los CSV que sumo-rl genera (uno por episodio) y arma un
    resumen (promedio por episodio, de sistema y por cruce) mas un diccionario
    con el detalle paso a paso de cada episodio.

    sumo-rl guarda el CSV de un episodio en el reset() siguiente, y el reset()
    inicial de train.py produce un _ep1 espurio (una sola fila en ceros). Por
    eso se descartan los CSV casi vacios y el _epN se renumera como episodio
    de entrenamiento N-1, que es el que imprime train.py y el de log_entrenamiento.csv.
    """
    archivos = glob.glob(os.path.join(carpeta, '*_ep*.csv'))
    if not archivos:
        raise FileNotFoundError(
            f"No hay CSVs en {carpeta}. Corre train.py primero (sumo-rl los genera automaticamente)."
        )

    detalle_por_episodio = {}
    for archivo in archivos:
        m = re.search(r'_ep(\d+)\.csv$', archivo)
        if not m:
            continue
        df = pd.read_csv(archivo)
        if len(df) < MIN_FILAS_EPISODIO:
            continue
        detalle_por_episodio[int(m.group(1)) - 1] = df

    filas_resumen = []
    for episodio in sorted(detalle_por_episodio):
        df = detalle_por_episodio[episodio]
        fila = {
            'episodio': episodio,
            'espera_media': df['system_mean_waiting_time'].mean(),
            'espera_total_media': df['system_total_waiting_time'].mean(),
            'vehiculos_detenidos_media': df['system_total_stopped'].mean(),
            'vehiculos_detenidos_max': df['system_total_stopped'].max(),
            'velocidad_media': df['system_mean_speed'].mean(),
        }
        for ts, cruce in CRUCES.items():
            fila[f'{cruce}_espera_acumulada_media'] = df[f'{ts}_accumulated_waiting_time'].mean()
            fila[f'{cruce}_vehiculos_detenidos_media'] = df[f'{ts}_stopped'].mean()
            fila[f'{cruce}_velocidad_media'] = df[f'{ts}_average_speed'].mean()
        filas_resumen.append(fila)
    resumen = pd.DataFrame(filas_resumen).sort_values('episodio').reset_index(drop=True)

    # Métricas de entrenamiento (pérdidas, entropía, R_int...) si la corrida las registró
    ruta_log = os.path.join(carpeta, 'log_entrenamiento.csv')
    if os.path.exists(ruta_log):
        resumen = resumen.merge(pd.read_csv(ruta_log), on='episodio', how='left')
    return resumen, detalle_por_episodio


def cargar_todas_las_corridas(corridas=CORRIDAS):
    """Resumen por episodio de todas las corridas, en formato largo (columna 'corrida')."""
    resumenes = []
    for corrida, carpeta in corridas.items():
        if not os.path.isdir(carpeta):
            continue
        resumen, _ = cargar_episodios(carpeta)
        resumenes.append(resumen.assign(corrida=corrida))
    todas = pd.concat(resumenes, ignore_index=True)
    return todas[['corrida'] + [c for c in todas.columns if c != 'corrida']]


def graficar_curvas_de_aprendizaje(resumen, salida='curvas_aprendizaje.png'):
    """
    Grafica como evolucionan las metricas clave episodio a episodio
    (deberian mejorar -menos espera, menos colas, mas velocidad- a medida
    que el agente entrena).
    """
    fig, ejes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    ejes[0].plot(resumen['episodio'], resumen['espera_media'], marker='o')
    ejes[0].set_ylabel('Espera media (s)')

    ejes[1].plot(resumen['episodio'], resumen['vehiculos_detenidos_media'], marker='o', color='tab:orange')
    ejes[1].set_ylabel('Vehiculos detenidos (prom.)')

    ejes[2].plot(resumen['episodio'], resumen['velocidad_media'], marker='o', color='tab:green')
    ejes[2].set_ylabel('Velocidad media (m/s)')
    ejes[2].set_xlabel('Episodio')

    fig.suptitle('Curvas de aprendizaje - RND-MAPPO')
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    print(f"Grafico guardado en {salida}")


if __name__ == '__main__':
    resumen, _ = cargar_episodios()
    print(resumen[['episodio', 'espera_media', 'vehiculos_detenidos_media', 'velocidad_media']].to_string(index=False))
    graficar_curvas_de_aprendizaje(resumen)
