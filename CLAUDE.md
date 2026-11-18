# C.I.T.A.D.E.L. — Control semafórico multiagente con RND-MAPPO

Taller de Control (Ing. Eléctrica y Electrónica, Uniandes). Aprendizaje por refuerzo multiagente
(MAPPO + curiosidad RND) para controlar 4 intersecciones semaforizadas de Bogotá simuladas en SUMO.
Equipo: Tomás Castro, Andrés Felipe Barbosa, Mariana Vanoy, Santiago Sánchez.
Profesor: Jorge Alfredo López Jiménez. **Responder siempre en español.**

**Dashboard:** terminado en `dashboard/` (uso y esquema de la base en `dashboard/README.md`).
`TAREA_DASHBOARD.md` queda como referencia del encargo original.

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
| `campana.py` | Campaña v2 repartida en 4 PCs (`--pc N`, `--listar`, `--prueba`). Salta corridas ya terminadas. Usa el prefijo del algoritmo actual (`v2a2_`) |
| `lineas_base.py` | Líneas base: semáforo actuado de SUMO y max-pressure |
| `evaluar_comparacion.py` | Evalúa políticas vs. líneas base (`--entorno v1/v2`, `--ventanas`, `--escalas`, `--semillas`) |
| `exportar_datos.py` | Consolida todo en CSV planos en `datos_dashboard/` (columnas en `datos_dashboard/diccionario_datos.md`) |
| `analizar_resultados.py` | Gráficas de curvas de aprendizaje (v1) |
| `demo_en_vivo.py` | Demo: dos sumo-gui lado a lado (tiempo fijo vs. agente). Ver `presentacion/GUIA_DEMO.md` |
| `generar_peatones.py` | Genera la demanda peatonal de `2026-09-12-17-03-26/peatones.rou.xml` |
| `dashboard/construir_bd.py` | Carga `datos_dashboard/v2/` en `dashboard/citadel.db` (SQLite; no se sube, se regenera en segundos) y la comprime en `citadel.db.gz` (~11 MB, **sí se sube**: es la que descarga el dashboard y la que usa GitHub Pages) |
| `dashboard/generar_animacion.py` | Graba con SUMO 10 min de hora pico de cada método (3 sin IA + versiones de la IA) en `dashboard/animacion/`; en el dashboard se eligen dos para compararlos |
| `dashboard/index.html` | Dashboard. Abrir con `cd dashboard; python -m http.server 8000`. Pensado para alguien que no conoce el proyecto. Funciona en celular (menú abajo, ≤ 640 px) y tiene botón "Abrir en el celular" con QR |
| `dashboard/servir_celular.py` | Sirve el dashboard en la red Wi-Fi (IP local) y muestra el QR: el celular debe estar en la misma red |
| `dashboard/generar_qr.py` | QR fijo del sitio publicado (`https://santisan1205.github.io/Taller-Control/dashboard/`) en `dashboard/img/qr_dashboard.png/.svg`; lo usa la presentación final |

## Carpetas
- `2026-09-12-17-03-26/`: escenario SUMO (red `Config_actualizado.net.xml`, rutas, peatones).
- `*.pth` en la raíz: pesos entrenados. Actor por cruce (`actor_mappo_cluster_<id>_<tag>.pth`) + un `critic_mappo_<tag>.pth`.
  Sin sufijo `v2_`/`v2a2_` = modelos v1 (los de la presentación). Los v1 **no cargan** en v2 (cambió la observación de C1/C4).
  En el repo están los `v2_` (algoritmo 1) de `base`, `mas_exploracion`, `prioriza_espera` y `sin_rnd` (los de
  `penaliza_cambio` y `peatones_alto` quedaron en otros PCs) y **los 18 `v2a2_` (algoritmo 2) completos**.
- `resultados/<corrida>/`: `config.json` y `log_entrenamiento.csv` de cada corrida. Los CSV por episodio de sumo-rl
  no se suben; `exportar_datos.py` reutiliza las curvas ya consolidadas de las corridas que no los tengan en el PC.
- `resultados_eval/`: evaluaciones (los episodios crudos no se suben, ver `.gitignore`).
- `datos_dashboard/`: datos para el dashboard. **`datos_dashboard/v2/` es la versión final**: las dos campañas del
  entorno v2, 36 corridas (18 `v2_` con el algoritmo 1 + 18 `v2a2_` con el 2) + 3 líneas base, 1170 episodios.
- `presentacion/`: presentación del 28 de octubre, de v1 (`presentacion.html`/`.pdf`, fuente en `presentacion/fuente/`, `GUION.pdf`),
  y **presentación final** de 8–10 min con las dos versiones del algoritmo: `python presentacion/generar_final.py` genera
  `presentacion_final.html` y `guion_final.html` desde `datos_dashboard/v2/` (los PDF se exportan con Chrome headless).
  Expositores: Tomy 1–3, Andrés 4–6, Santi 7–9, Mariana 10–13.
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
- **1** (campaña v1 y campaña `v2_*`): crítico con una sola salida, actualizado por turnos con la
  recompensa de cada agente; RND normalizando cada observación con su propia media → R_int < 1 % de |R_ext|.
- **2** (campaña `v2a2_*`, la que entrena el código actual): crítico con una salida V_i(s) por agente + ValueNorm
  y un solo optimizador; RND con media/desviación acumuladas de las observaciones y R_int normalizada por su
  desviación acumulada (pesa de verdad: al inicio del orden de |R_ext|). `critic_loss` queda en unidades
  normalizadas (no comparable con la versión 1). Los `critic_mappo_*.pth` cambian de forma; la evaluación solo
  carga los actores, así que los modelos de las dos versiones se evalúan igual.
- La campaña `v2a2_*` se corrió completa (mismas 6 configuraciones × 3 semillas, 150 episodios, commit `aa9403d`,
  SUMO 1.27.1 verificado re-simulando episodios de cada PC). El código actual ya no puede entrenar con la versión 1.
- `evaluar_comparacion.py --pasos N` (N ≠ 720) guarda los episodios como `__p<N>` y `exportar_datos.py` los ignora.

## Mensajes clave de los resultados (v2)
- Las dos versiones del algoritmo reducen la espera vehicular entre 64 % y 73 % frente al tiempo fijo en hora pico
  (demanda nominal).
- El agente logra un compromiso vehículos/peatones frente a max-pressure: max-pressure deja menos espera
  vehicular (~1,3 s en pico) pero hace esperar a los peatones ~35 s, contra ~8–11 s con el agente.
- RND (curiosidad) no aporta: `sin_rnd` rinde igual que `base` en las dos versiones. Con la versión 2 la
  conclusión es sólida (R_int sí pesa); con la 1 era casi trivial (R_int ≈ 0).
- Algoritmo 2 frente a 1 (18 políticas de cada uno, promedio de las 3 escalas): la espera vehicular no cambia
  (pico 2,66 → 2,59 s, p = 0,84; valle 3,21 → 3,29 s, p = 0,23), la espera peatonal empeora ~1 s (pico
  8,31 → 9,52 s, p = 0,001), la variación entre semillas baja (desv. 0,33 → 0,19 s) y ningún episodio se tranca.
  Lectura: el método es correcto y más estable, pero con 150 episodios no mejora el desempeño; probable causa,
  la curiosidad ahora explora mucho al inicio (λ = 1). Siguiente experimento natural: λ inicial menor o más episodios.
- `v2_peatones_alto_seed1` (algoritmo 1) cae en gridlock en hora pico con demanda ×1,2 (espera 94 s);
  es la única alerta de la base del dashboard.

## Publicación del dashboard (GitHub Pages)
- Sitio: `https://santisan1205.github.io/Taller-Control/` (el `index.html` de la raíz redirige a `dashboard/`;
  `.nojekyll` evita el procesamiento de Jekyll). Pages se publica desde `main`, carpeta raíz.
- Necesita en el repo `dashboard/citadel.db.gz` y `dashboard/animacion/`. Si cambian los resultados: correr
  `construir_bd.py` (regenera el `.gz`), commit y push a `main`.
- Si cambia la dirección pública: actualizar `URL_PUBLICA` en `dashboard/index.html` y `dashboard/generar_qr.py`,
  y volver a correr `generar_qr.py` y `presentacion/generar_final.py`.
- iCloud a veces renombra archivos con sufijo " 2" (`exportar_datos 2.py`): revisar antes de hacer commit.

## Convenciones
- Ramas: el trabajo vive en `metricas-dashboard`; `main` se mantiene al día con ella.
- Mensajes de commit en español, en imperativo y cortos (ver `git log`).
- No subir CSV por paso de entrenamiento ni episodios crudos de evaluación (pesan cientos de MB).
