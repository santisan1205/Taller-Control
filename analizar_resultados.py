import glob
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

CARPETA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')


def cargar_episodios(carpeta=CARPETA_RESULTADOS):
    """
    Carga todos los CSV que sumo-rl genera (uno por episodio) y arma un
    resumen (promedio por episodio) mas un diccionario con el detalle
    paso a paso de cada episodio.
    """
    archivos = glob.glob(os.path.join(carpeta, '*.csv'))
    if not archivos:
        raise FileNotFoundError(
            f"No hay CSVs en {carpeta}. Corre train.py primero (sumo-rl los genera automaticamente)."
        )

    detalle_por_episodio = {}
    for archivo in archivos:
        m = re.search(r'_ep(\d+)\.csv$', archivo)
        episodio = int(m.group(1)) if m else -1
        detalle_por_episodio[episodio] = pd.read_csv(archivo)

    filas_resumen = []
    for episodio in sorted(detalle_por_episodio):
        df = detalle_por_episodio[episodio]
        filas_resumen.append({
            'episodio': episodio,
            'espera_media': df['system_mean_waiting_time'].mean(),
            'espera_total_media': df['system_total_waiting_time'].mean(),
            'vehiculos_detenidos_media': df['system_total_stopped'].mean(),
            'velocidad_media': df['system_mean_speed'].mean(),
        })
    resumen = pd.DataFrame(filas_resumen).sort_values('episodio').reset_index(drop=True)
    return resumen, detalle_por_episodio


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
    print(f"Grafico guardado en {salida}")


if __name__ == '__main__':
    resumen, _ = cargar_episodios()
    print(resumen.to_string(index=False))
    graficar_curvas_de_aprendizaje(resumen)
