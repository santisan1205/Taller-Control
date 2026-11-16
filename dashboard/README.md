# Dashboard C.I.T.A.D.E.L. (campaña v2)

## Uso
```powershell
python dashboard/construir_bd.py          # CSV de datos_dashboard/v2/ -> dashboard/citadel.db
cd dashboard; python -m http.server 8000  # abrir http://localhost:8000
```
No hace falta SUMO. El navegador carga `citadel.db` con sql.js (SQLite en WebAssembly, desde cdnjs), así que
necesita internet la primera vez y **no funciona abriendo el HTML con doble clic** (`file://`).

Si cambian los CSV (`python exportar_datos.py --v2`), se vuelve a correr `construir_bd.py`. La base no se edita a mano.

## Base de datos (`citadel.db`)
| Objeto | Tipo | Contenido |
|---|---|---|
| `corridas`, `cruces`, `entrenamiento_episodios`, `evaluacion_detalle`, `evaluacion_por_cruce`, `evaluacion_resumen`, `evaluacion_configs`, `evaluacion_series_tiempo` | tablas | Una por CSV de `datos_dashboard/v2/`, mismas columnas (ver `datos_dashboard/diccionario_datos.md`) |
| `metricas` | tabla | Etiqueta, unidad, grupo y sentido (`mejor_si`) de cada métrica |
| `alertas` | tabla | Episodios atípicos detectados al construir (espera > 5× la mediana del escenario y > 20 s): hoy, el gridlock de `v2_peatones_alto_seed1` |
| `meta` | tabla | Fecha de construcción, fuente y filas por tabla |
| `v_politicas` | vista | Política → `config`, `semilla_entrenamiento` |
| `v_detalle`, `v_por_cruce` | vistas | Las tablas de evaluación con la columna `config` (para líneas base, `config` = `modo`) |
| `v_entrenamiento_config` | vista | Curva por configuración y episodio: media, mínimo y máximo entre las 3 semillas |
| `v_series_config` | vista | Serie de tiempo media por configuración (o línea base), ventana y escala |

Índices sobre `politica`, `config`, `modo`, `ventana`, `escala`, `metrica`, `referencia` y `cruce`.

## Animación de los carros (`animacion/`)
La página **Red de Tráfico** muestra dos ventanas con 10 minutos reales de SUMO (hora pico, tráfico normal,
semilla 10). Cada ventana tiene un selector para elegir el método que se compara: los 3 semáforos sin IA
(fijo, con sensores y regla de colas) y las 6 versiones de la IA de cada algoritmo (semilla de entrenamiento 0).
Todos los métodos ven exactamente los mismos carros, así que cualquier pareja se puede comparar.
Se generan con SUMO instalado:
```powershell
python dashboard/generar_animacion.py   # todos los métodos con pesos en el PC; los ya generados se saltan
```
SUMO graba la posición de cada vehículo y peatón cada segundo (`--fcd-output`) y el estado de cada semáforo
(`SaveTLSStates`). El script lo recorta a la zona de los 4 cruces: `animacion/red.json` (geometría, escenario y
lista de métodos) y un `animacion/<metodo>.json` por método (~1 MB cada uno; el navegador solo descarga los que
se eligen). Quien abre el dashboard no necesita SUMO. `--help` muestra cómo cambiar el escenario o los métodos.
`v2_peatones_alto_seed0` (algoritmo 1) no está: sus pesos quedaron en el PC4 y nunca se subieron.

## Logo (`img/`)
`citadel_logo_original.jpg` es el logo original (fondo negro). A partir de él se generaron, con fondo transparente
(el alfa sale del brillo de cada píxel, así se conservan el degradado y el brillo): `citadel_logo.png` (logo
completo, en la pantalla de inicio), `citadel_emblema.png` (solo el símbolo, en el menú lateral y en Comunidad) y
`favicon.png` (ícono de la pestaña).

## Qué muestra cada parte de la interfaz
Está pensada para **alguien que no conoce el proyecto**: nombres sencillos ("IA estándar", "Semáforo fijo",
"Regla de colas"), sin p-valores ni intervalos a la vista (salen al pasar el mouse) y un glosario en Soporte.
El diseño es el del prototipo (`dashboard_interactivo.html`).
Los filtros de arriba (versión de la IA, hora, tráfico y punto de comparación) aplican a todas las páginas.

- **Dashboard**, con los tres modos:
  - *Evaluación*: cuánto mejora la IA elegida frente al punto de comparación, la espera en cada cruce y un
    ranking de todas las opciones de mejor a peor, con la pregunta que responde ("¿Cuánto esperan los carros?").
  - *Entrenamiento*: cómo cambió la IA del inicio al final del entrenamiento y su curva de aprendizaje.
  - *Simulación*: espera, detenidos o velocidad minuto a minuto durante una hora (solo con tráfico normal).
  - *Explorar Insights IA*: mensajes en palabras calculados con los datos (peatones, curiosidad, más tráfico,
    mejor versión y el episodio en que el tráfico se trancó).
- **Red de Tráfico**: la animación de SUMO de los dos métodos elegidos, con tres encuadres (C1/C3/C4, C2, toda la zona) y velocidades.
- **Entrenamiento**: curvas de las 6 versiones de la IA y qué hace distinta a cada una.
- **Comunidad**: enlace al repositorio en GitHub (con el comando para clonarlo) y el equipo.
- **Configuración**: qué contiene la base y cómo actualizarla. **Soporte**: glosario en lenguaje sencillo.

Los filtros de arriba solo se muestran en el Dashboard y en Entrenamiento, donde tienen efecto. En pantallas
angostas el menú lateral queda solo con íconos y las tarjetas pasan a una columna.
