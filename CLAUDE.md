# C.I.T.A.D.E.L. — Control semafórico multiagente con RND-MAPPO

Taller de Control (Ing. Eléctrica y Electrónica, Uniandes). Aprendizaje por refuerzo multiagente
(MAPPO + curiosidad RND) para controlar 4 intersecciones semaforizadas de Bogotá simuladas en SUMO.
Equipo: Tomás Castro, Andrés Felipe Barbosa, Mariana Vanoy, Santiago Sánchez.
Profesor: Jorge Alfredo López Jiménez. **Responder siempre en español.**

**Tarea pendiente actual:** base de datos + dashboard HTML → ver `TAREA_DASHBOARD.md`.

## Entorno
- Python con `pip install -r requirements.txt` (versiones fijadas; `sumo-rl==1.4.5`).
- SUMO **1.27.1** instalado y `SUMO_HOME` definido.
- Correr todo **desde la raíz del proyecto** (los scripts usan rutas relativas a esa carpeta).
- En Windows, si existe `.venv`: `.\.venv\Scripts\python.exe <script>.py`.

## Mapa del código
| Archivo | Qué hace |
|---|---|
| `setup_entorno.py` | Crea el entorno sumo-rl (`crear_entorno_sumo`), los 4 cruces (`CRUCES`), ventanas valle/pico (`VENTANAS`), observación con peatones en C1/C4 |
| `modelo_red_rl.py` | Redes: `MAPPOActor`, `MAPPOCritic`, `RNDModule`, `RolloutBuffer`, `MAPPOTrainer`, `RewardCalculator` (w1 cola, w2 espera, w3 peatones, p_cambio) |
| `train.py` | Entrena. `python train.py --help`. Flags v2: `--peatones --demanda-variable`, `--tag`, `--seed`, `--sin-rnd` |
| `campana.py` | Campaña v2 repartida en 4 PCs (`--pc N`, `--listar`, `--prueba`). Salta corridas ya terminadas |
| `lineas_base.py` | Líneas base: semáforo actuado de SUMO y max-pressure |
| `evaluar_comparacion.py` | Evalúa políticas vs. líneas base (`--entorno v1/v2`, `--ventanas`, `--escalas`, `--semillas`) |
| `exportar_datos.py` | Consolida todo en CSV planos en `datos_dashboard/` (columnas en `datos_dashboard/diccionario_datos.md`) |
| `analizar_resultados.py` | Gráficas de curvas de aprendizaje (v1) |
| `demo_en_vivo.py` | Demo: dos sumo-gui lado a lado (tiempo fijo vs. agente). Ver `presentacion/GUIA_DEMO.md` |
| `generar_peatones.py` | Genera la demanda peatonal de `2026-09-12-17-03-26/peatones.rou.xml` |

## Carpetas
- `2026-09-12-17-03-26/`: escenario SUMO (red `Config_actualizado.net.xml`, rutas, peatones).
- `*.pth` en la raíz: pesos entrenados. Actor por cruce (`actor_mappo_cluster_<id>_<tag>.pth`) + un `critic_mappo_<tag>.pth`.
  Sin sufijo `v2_` = modelos v1 (los de la presentación). Los v1 **no cargan** en v2 (cambió la observación de C1/C4).
  En el repo están los v2 de `base`, `mas_exploracion`, `prioriza_espera` y `sin_rnd`; los de `penaliza_cambio`
  y `peatones_alto` se entrenaron en otros PCs y sus resultados ya están consolidados en `datos_dashboard/v2/`.
- `resultados/<corrida>/`: `config.json` y `log_entrenamiento.csv` de cada corrida.
- `resultados_eval/`: evaluaciones (los episodios crudos no se suben, ver `.gitignore`).
- `datos_dashboard/`: datos para el dashboard. **`datos_dashboard/v2/` es la versión final** (18 políticas + 3 líneas base, 630 episodios).
- `presentacion/`: presentación (`presentacion.html`/`.pdf`, fuente en `presentacion/fuente/`), guion, diagrama del proyecto.
- `poster/`: póster 70×100 cm para la Semana de la Innovación.
  - `python poster/generar_graficas.py` → SVG en `poster/graficas/` + KPIs por consola.
  - `poster/poster.html` se exporta a PDF/PNG con Chrome headless.
  - `python poster/exportar_pptx.py` → `.pptx` editable sobre la plantilla (usa Verdana; requiere Chrome).
  - Letra grande a propósito (cuerpo ~25 pt): no reducirla.

## Entorno v1 vs. v2
- **v1**: sin peatones, solo ventana valle (0–3600 s). Resultados de la presentación.
- **v2**: peatones en los cruces C1 y C4, ventana pico (4400–8000 s), demanda variable (escala 0,7–1,3),
  verde máximo de 60 s también en entrenamiento, líneas base actuado y max-pressure. Todo lleva prefijo `v2_`.
  Detalles y procedimiento de la campaña en `GUIA_CAMPANA.md`.

## Versión del algoritmo (`modelo_red_rl.VERSION_ALGORITMO`, se guarda en `config.json`)
- **1** (campaña v1 y campaña v2 consolidada): crítico con una sola salida, actualizado por turnos con la
  recompensa de cada agente; RND normalizando cada observación con su propia media → R_int < 1 % de |R_ext|.
- **2** (corridas nuevas): crítico con una salida V_i(s) por agente + ValueNorm y un solo optimizador;
  RND con media/desviación acumuladas de las observaciones y R_int normalizada por su desviación acumulada.
  `critic_loss` queda en unidades normalizadas (no comparable con la versión 1). Los `critic_mappo_*.pth`
  cambian de forma; la evaluación solo carga los actores, así que los modelos viejos se siguen evaluando.
- Hay que reentrenar para medir el efecto: los resultados actuales (incluido "RND no aporta") son de la versión 1.
- `evaluar_comparacion.py --pasos N` (N ≠ 720) guarda los episodios como `__p<N>` y `exportar_datos.py` los ignora.

## Mensajes clave de los resultados (v2)
- El agente logra un compromiso vehículos/peatones frente a max-pressure.
- RND (curiosidad) no aporta: `v2_sin_rnd` rinde parecido a `v2_base`.
- `v2_peatones_alto_seed1` cae en gridlock en hora pico con demanda ×1,2.

## Convenciones
- Ramas: el trabajo vive en `metricas-dashboard`; `main` se mantiene al día con ella.
- Mensajes de commit en español, en imperativo y cortos (ver `git log`).
- No subir CSV por paso de entrenamiento ni episodios crudos de evaluación (pesan cientos de MB).
