# Tarea: base de datos + dashboard HTML (C.I.T.A.D.E.L.)

Guía de traspaso para quien continúe el dashboard (Santi, PC4) con Claude Code.
El contexto general del proyecto está en `CLAUDE.md`; aquí solo está lo de esta tarea.

## Cómo empezar (prompt sugerido para Claude Code)
> Lee `CLAUDE.md`, `TAREA_DASHBOARD.md` y `datos_dashboard/diccionario_datos.md`. Mira el prototipo
> `presentacion/dashboard.jpg`. Luego propón el esquema de la base de datos y arma el dashboard
> siguiendo el plan de `TAREA_DASHBOARD.md`. Trabaja en una carpeta `dashboard/`.

Antes de empezar: `git pull` y `pip install -r requirements.txt`. **Para esta tarea no hace falta SUMO**:
todos los datos ya están calculados en CSV.

## Objetivo
1. **Base de datos**: cargar los CSV de `datos_dashboard/v2/` en una base SQLite (`dashboard/citadel.db`)
   generada por un script reproducible (`dashboard/construir_bd.py`). Nada se edita a mano: si cambian
   los CSV, se vuelve a correr el script.
2. **Dashboard HTML** (`dashboard/index.html`) que lea esa base y muestre los resultados de la campaña v2.
   Se presentó en clase como "Dashboard de seguimiento del modelo" con tres vistas:
   **evaluación, entrenamiento y simulación** (ver `presentacion/dashboard.jpg`; los números de esa imagen
   son ilustrativos, no reales).

## Datos de entrada: `datos_dashboard/v2/` (versión final)
| CSV | Filas | Qué tiene |
|---|---|---|
| `corridas.csv` | 18 | Una fila por política entrenada con todos sus hiperparámetros |
| `cruces.csv` | 4 | Los 4 cruces (C1–C4): id SUMO y nombre |
| `entrenamiento_episodios.csv` | 2 700 | Curva de aprendizaje: una fila por corrida × episodio (con `config`, `semilla_entrenamiento`, `ventana`, `escala` sorteadas) |
| `evaluacion_detalle.csv` | 630 | Una fila por episodio de evaluación (política × ventana × escala × semilla SUMO), con métricas vehiculares y peatonales |
| `evaluacion_por_cruce.csv` | 2 520 | Lo mismo, desglosado por cruce |
| `evaluacion_resumen.csv` | 10 206 | Política vs. cada línea base (`referencia`): media, IC95, `mejora_pct`, `p_valor` |
| `evaluacion_configs.csv` | 2 916 | **Tabla principal del informe**: por configuración (promedia las 3 semillas de entrenamiento) × ventana × escala × métrica × referencia |
| `evaluacion_series_tiempo.csv` | 151 410 | Series dentro de la hora simulada (para la vista "simulación") |

Unidades, convenciones y significado de cada columna: **`datos_dashboard/diccionario_datos.md`**
(sección `datos_dashboard/v2/`). Leerlo antes de diseñar el esquema.

Dimensiones para filtrar:
- **config** (6): `v2_base`, `v2_sin_rnd`, `v2_mas_exploracion`, `v2_prioriza_espera`, `v2_penaliza_cambio`, `v2_peatones_alto`. Cada una × 3 semillas (`seed0..2`).
- **líneas base** (3): `tiempo_fijo`, `actuado`, `max_pressure`.
- **ventana**: `valle` / `pico`. **escala** de demanda: 0.8 / 1.0 / 1.2. **semilla SUMO**: 10–50.
- **cruce**: C1–C4 (peatones solo en C1 y C4).

## Plan sugerido
1. **Esquema SQLite**: una tabla por CSV (mismos nombres de columna) + índices en las dimensiones de filtro
   (`politica`, `config`, `ventana`, `escala`, `metrica`, `referencia`). Opcional: vistas SQL para lo más usado.
2. **`construir_bd.py`**: lee los CSV con pandas, crea `citadel.db`, imprime un conteo de filas por tabla
   y verifica que coincide con los CSV.
3. **Dashboard**: un HTML que cargue `citadel.db` en el navegador con **sql.js** (SQLite en WebAssembly,
   desde CDN) y grafique con Chart.js o Plotly. Como el navegador no deja leer archivos con `file://`,
   se abre con un servidor local:
   ```powershell
   cd dashboard; python -m http.server 8000   # y abrir http://localhost:8000
   ```
   (Alternativa si se prefiere un solo archivo: que `construir_bd.py` también exporte un JSON resumido y
   lo incruste en el HTML. Las series de tiempo pesan, así que conviene agregarlas antes.)
4. **Vistas**:
   - **Evaluación**: tarjetas KPI (espera media, cola, tiempo de viaje, peatones) del agente vs. las 3 líneas base;
     barras con IC95 por config, filtros de ventana/escala; tabla con `mejora_pct` y `p_valor`; desglose por cruce.
   - **Entrenamiento**: curvas de recompensa por episodio por config (media de 3 semillas ± banda), con opción de ver cada semilla.
   - **Simulación**: series de tiempo dentro de la hora para una política vs. línea base elegida.
   - **Peatones**: espera media/p95 de peatones en C1 y C4 (el compromiso vehículos ↔ peatones es un mensaje clave).
5. Probar abriendo el dashboard y revisar que los números coincidan con los del póster
   (`python poster/generar_graficas.py` imprime los KPI del póster).
6. Commit y push (rama `metricas-dashboard`, y llevar `main` al mismo commit).

## Mensajes que el dashboard debe dejar ver
- El agente logra un compromiso vehículos/peatones frente a max-pressure (que no ve a los peatones).
- RND (curiosidad) no aporta: `v2_sin_rnd` ≈ `v2_base`.
- Robustez: cómo se degrada cada política al subir la demanda (escala 1.2) y en hora pico.

## Advertencias (no omitir en el dashboard)
- **Gridlock real**: `v2_peatones_alto_seed1`, pico, escala 1.2, semilla 50 → `espera_media` ≈ 94 s
  (los otros 4 episodios de ese escenario: 2–4 s). No filtrarlo; marcarlo en los gráficos porque arrastra la media.
- `v2_mas_exploracion` se **entrenó** con SUMO 1.28.0 (sus curvas de entrenamiento vienen de ahí); su
  **evaluación** se rehízo con 1.27.1 como las demás, así que es comparable.
- `mejora_pct` positivo = el agente es mejor (ya corrige el sentido de cada métrica).
- Las carpetas `datos_dashboard/` (raíz) y `datos_dashboard/max_green/` son la campaña v1 (la de la presentación).
  El dashboard nuevo usa **v2**. Si se quiere mostrar v1, que sea en una pestaña aparte y rotulada.

## Estilo visual
Coherente con el póster y la presentación (colores Uniandes): amarillo `#FFD100`, azul oscuro `#0C1829`,
gris `#334155`, fondo claro `#FDFDFB`. Textos en español. Nombre del proyecto: **C.I.T.A.D.E.L.**
(Control Inteligente de Tráfico Adaptativo con Distribución y Exploración Local).

## Fuera de alcance
- No reentrenar ni reevaluar (no hace falta SUMO).
- No modificar `datos_dashboard/` a mano: si algo está mal en los datos, se corrige en `exportar_datos.py`.
