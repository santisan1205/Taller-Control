"""
Construye dashboard/citadel.db (SQLite) a partir de los CSV de la campaña v2
(datos_dashboard/v2/). Nada se edita a mano: si cambian los CSV, se vuelve a
correr este script.

Uso (desde la raíz del proyecto o desde dashboard/):
    python dashboard/construir_bd.py

Esquema:
- Una tabla por CSV, con el mismo nombre y las mismas columnas.
- Tablas auxiliares: `metricas` (etiqueta, unidad y sentido de cada métrica),
  `alertas` (episodios atípicos, p. ej. el gridlock de v2_peatones_alto_seed1)
  y `meta` (fecha de construcción y conteos).
- Vistas para lo que más consulta el dashboard (v_*): agregan las 3 semillas de
  entrenamiento por configuración.
- Índices en las dimensiones de filtro (politica, config, modo, ventana, escala,
  metrica, referencia).
"""
import datetime as dt
import gzip
import os
import sqlite3
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_CSV = os.path.join(RAIZ, 'datos_dashboard', 'v2')
SALIDA = os.path.join(RAIZ, 'dashboard', 'citadel.db')

TABLAS = ['corridas', 'cruces', 'entrenamiento_episodios', 'evaluacion_detalle', 'evaluacion_por_cruce',
          'evaluacion_resumen', 'evaluacion_configs', 'evaluacion_series_tiempo']

# Índices por tabla (solo sobre columnas que existen en el CSV)
INDICES = {
    'corridas': [('corrida',)],
    'entrenamiento_episodios': [('config', 'episodio'), ('corrida',)],
    'evaluacion_detalle': [('politica',), ('modo', 'ventana', 'escala')],
    'evaluacion_por_cruce': [('politica',), ('modo', 'ventana', 'escala', 'cruce')],
    'evaluacion_resumen': [('politica', 'referencia', 'ventana', 'escala'), ('metrica',)],
    'evaluacion_configs': [('config', 'referencia', 'ventana', 'escala'), ('metrica',)],
    'evaluacion_series_tiempo': [('politica', 'ventana', 'escala'), ('modo',)],
}

# Etiqueta, unidad y grupo de cada métrica de evaluación (el sentido sale de los CSV)
METRICAS = {
    'espera_media': ('Espera media vehicular', 's', 'vehiculos'),
    'espera_total_media': ('Espera total en la red', 's', 'vehiculos'),
    'vehiculos_detenidos_media': ('Vehículos detenidos (prom.)', 'veh', 'vehiculos'),
    'vehiculos_detenidos_max': ('Vehículos detenidos (máx.)', 'veh', 'vehiculos'),
    'velocidad_media': ('Velocidad media', 'm/s', 'vehiculos'),
    'viajes_completados': ('Viajes completados', 'veh', 'flujo'),
    'viajes_sin_terminar': ('Viajes sin terminar', 'veh', 'flujo'),
    'vehiculos_sin_insertar': ('Vehículos sin insertar', 'veh', 'flujo'),
    'vehiculos_por_hora': ('Flujo vehicular', 'veh/h', 'flujo'),
    'tiempo_viaje_medio': ('Tiempo de viaje medio', 's', 'viaje'),
    'tiempo_viaje_p95': ('Tiempo de viaje p95', 's', 'viaje'),
    'perdida_tiempo_media': ('Tiempo perdido por viaje', 's', 'viaje'),
    'espera_viaje_media': ('Espera por viaje', 's', 'viaje'),
    'espera_viaje_p95': ('Espera por viaje p95', 's', 'viaje'),
    'paradas_por_vehiculo': ('Paradas por vehículo', '#', 'viaje'),
    'demora_insercion_media': ('Demora de inserción', 's', 'viaje'),
    'teleports': ('Teleports', '#', 'flujo'),
    'co2_total_kg': ('Emisiones CO₂', 'kg/h', 'emisiones'),
    'combustible_total_kg': ('Consumo de combustible', 'kg/h', 'emisiones'),
    'co2_g_por_km': ('CO₂ por km', 'g/km', 'emisiones'),
    'cambios_fase_total': ('Cambios de fase', '#/h', 'control'),
    'peatones_completados': ('Peatones que cruzaron', 'personas', 'peatones'),
    'peatones_sin_terminar': ('Peatones sin terminar', 'personas', 'peatones'),
    'peatones_tiempo_viaje_medio': ('Caminata media (peatones)', 's', 'peatones'),
    'peatones_espera_media': ('Espera peatonal media', 's', 'peatones'),
    'peatones_espera_p95': ('Espera peatonal p95', 's', 'peatones'),
    'peatones_perdida_tiempo_media': ('Tiempo perdido (peatones)', 's', 'peatones'),
}

# Expresión SQL que da la configuración de una fila de evaluación: el nombre de
# la política sin _seedN para los agentes, y el modo para las líneas base
CONFIG_SQL = ("CASE WHEN modo LIKE 'rnd_mappo%' THEN substr(politica, 1, instr(politica, '_seed') - 1) "
              "ELSE modo END")

VISTAS = {
    # Política -> configuración y semilla de entrenamiento (las líneas base son su propia "config")
    'v_politicas': f"""
        SELECT DISTINCT politica, modo, {CONFIG_SQL} AS config,
               CASE WHEN modo LIKE 'rnd_mappo%'
                    THEN CAST(substr(politica, instr(politica, '_seed') + 5) AS INTEGER) END AS semilla_entrenamiento
        FROM evaluacion_detalle""",
    # evaluacion_detalle con la columna config
    'v_detalle': f"SELECT {CONFIG_SQL} AS config, * FROM evaluacion_detalle",
    'v_por_cruce': f"SELECT {CONFIG_SQL} AS config, * FROM evaluacion_por_cruce",
    # Curva de aprendizaje por configuración: media, mínimo y máximo entre las 3 semillas
    'v_entrenamiento_config': """
        SELECT config, episodio, COUNT(*) AS n_semillas,
               AVG(recompensa_ext) AS recompensa_ext, MIN(recompensa_ext) AS recompensa_ext_min,
               MAX(recompensa_ext) AS recompensa_ext_max,
               AVG(espera_media) AS espera_media, MIN(espera_media) AS espera_media_min,
               MAX(espera_media) AS espera_media_max,
               AVG(recompensa_int) AS recompensa_int, MIN(recompensa_int) AS recompensa_int_min,
               MAX(recompensa_int) AS recompensa_int_max,
               AVG(entropia) AS entropia, MIN(entropia) AS entropia_min, MAX(entropia) AS entropia_max,
               AVG(critic_loss) AS critic_loss, MIN(critic_loss) AS critic_loss_min,
               MAX(critic_loss) AS critic_loss_max
        FROM entrenamiento_episodios GROUP BY config, episodio""",
    # Serie de tiempo media por configuración (promedia semillas de entrenamiento y de SUMO)
    'v_series_config': f"""
        SELECT {CONFIG_SQL} AS config, ventana, escala, tiempo_s,
               AVG(espera_media) AS espera_media, AVG(vehiculos_detenidos) AS vehiculos_detenidos,
               AVG(velocidad_media) AS velocidad_media, AVG(C1_detenidos) AS C1_detenidos,
               AVG(C2_detenidos) AS C2_detenidos, AVG(C3_detenidos) AS C3_detenidos,
               AVG(C4_detenidos) AS C4_detenidos, COUNT(*) AS n_episodios
        FROM evaluacion_series_tiempo
        WHERE modo IN ('rnd_mappo', 'tiempo_fijo', 'actuado', 'max_pressure')
        GROUP BY 1, ventana, escala, tiempo_s""",
}


def detectar_alertas(detalle):
    """
    Episodios cuya espera media es muy superior a la de los otros episodios del
    mismo escenario (política x ventana x escala): mas de 5 veces la mediana y
    mas de 20 s. Con los datos actuales sale solo el gridlock conocido de
    v2_peatones_alto_seed1 (pico, x1.2, semilla 50).
    """
    filas = []
    for (politica, modo, ventana, escala), grupo in detalle.groupby(['politica', 'modo', 'ventana', 'escala']):
        mediana = grupo['espera_media'].median()
        for _, f in grupo.iterrows():
            if f['espera_media'] > 20 and f['espera_media'] > 5 * mediana:
                filas.append({
                    'tipo': 'gridlock', 'politica': politica, 'modo': modo, 'ventana': ventana, 'escala': escala,
                    'semilla': int(f['semilla']), 'metrica': 'espera_media', 'valor': f['espera_media'],
                    'mediana_escenario': mediana,
                    'descripcion': (f'{politica}, {ventana} x{escala:g}, semilla {int(f["semilla"])}: espera media '
                                    f'{f["espera_media"]:.1f} s frente a {mediana:.1f} s de mediana en ese escenario. '
                                    'Es un gridlock real (no un error de datos): arrastra la media.'),
                })
    return pd.DataFrame(filas, columns=['tipo', 'politica', 'modo', 'ventana', 'escala', 'semilla', 'metrica',
                                        'valor', 'mediana_escenario', 'descripcion'])


def construir(salida=SALIDA):
    if os.path.exists(salida):
        os.remove(salida)
    con = sqlite3.connect(salida)
    conteos = {}
    datos = {}
    for tabla in TABLAS:
        df = pd.read_csv(os.path.join(CARPETA_CSV, f'{tabla}.csv'))
        # Booleanos de corridas.csv (True/False) como 0/1 para SQLite
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
        df.to_sql(tabla, con, index=False)
        datos[tabla] = df
        conteos[tabla] = len(df)

    sentido = (datos['evaluacion_configs'][['metrica', 'mejor_si']].drop_duplicates().set_index('metrica')['mejor_si'])
    metricas = pd.DataFrame([
        {'metrica': m, 'etiqueta': e, 'unidad': u, 'grupo': g, 'mejor_si': sentido.get(m)}
        for m, (e, u, g) in METRICAS.items()
    ])
    metricas.to_sql('metricas', con, index=False)

    alertas = detectar_alertas(datos['evaluacion_detalle'])
    alertas.to_sql('alertas', con, index=False)

    for tabla, indices in INDICES.items():
        for i, columnas in enumerate(indices):
            con.execute(f'CREATE INDEX idx_{tabla}_{i} ON {tabla} ({", ".join(columnas)})')
    for nombre, sql in VISTAS.items():
        con.execute(f'CREATE VIEW {nombre} AS {sql}')

    meta = {
        'construida': dt.datetime.now().isoformat(timespec='seconds'),
        'fuente': 'datos_dashboard/v2/',
        'campana': 'v2 (peatones en C1/C4, valle y pico, 3 líneas base)',
        **{f'filas_{t}': str(n) for t, n in conteos.items()},
    }
    pd.DataFrame(list(meta.items()), columns=['clave', 'valor']).to_sql('meta', con, index=False)
    con.commit()

    # Verificación: cada tabla tiene las mismas filas que su CSV
    print(f'Base construida en {os.path.relpath(salida, RAIZ)}')
    errores = []
    for tabla in TABLAS:
        en_bd = con.execute(f'SELECT COUNT(*) FROM {tabla}').fetchone()[0]
        estado = 'ok' if en_bd == conteos[tabla] else 'DIFERENTE'
        if estado != 'ok':
            errores.append(tabla)
        print(f'  {tabla:28s} CSV {conteos[tabla]:7d}  BD {en_bd:7d}  {estado}')
    for vista in VISTAS:
        print(f'  {vista:28s} vista con {con.execute(f"SELECT COUNT(*) FROM {vista}").fetchone()[0]} filas')
    print(f'  alertas: {len(alertas)}')
    for d in alertas['descripcion']:
        print(f'    - {d}')
    con.execute('VACUUM')
    con.close()
    print(f'Tamaño: {os.path.getsize(salida) / 1e6:.1f} MB')
    comprimir(salida)
    if errores:
        sys.exit(f'Las tablas {errores} no coinciden con sus CSV')


def comprimir(salida=SALIDA):
    """citadel.db.gz: la que descarga el dashboard (~4 veces más liviana, importa en el celular).
    Es la que se sube al repositorio para GitHub Pages; citadel.db no se sube."""
    with open(salida, 'rb') as f, gzip.GzipFile(salida + '.gz', 'wb', compresslevel=9, mtime=0) as gz:
        gz.write(f.read())
    print(f'Comprimida: {os.path.relpath(salida + ".gz", RAIZ)} ({os.path.getsize(salida + ".gz") / 1e6:.1f} MB)')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # tildes en la consola de Windows
    except (AttributeError, ValueError):
        pass
    construir()
