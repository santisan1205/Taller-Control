# Guía: campaña de entrenamiento v2 (4 PCs)

## Qué cambia en el entorno v2
- **Peatones** solo en dos pasos peatonales semaforizados: `:cluster_10564135163_13571629282_5146149679_c0` (C1) y `:cluster_10564135212_13571629279_13571629299_c0` (C4). La demanda la genera `python generar_peatones.py` (120 personas/h por sentido; 240/h en la hora pico).
  - C1 y C4 ven dos valores más en su observación: peatones esperando y la espera máxima.
  - `w3` ahora sí tiene efecto en esos dos cruces.
- **Hora pico:** además del valle (0–3600 s), hay una ventana pico (4400–8000 s) que incluye los flujos de 5000–7000 s.
- **Demanda variable:** con `--demanda-variable`, cada episodio de entrenamiento sortea la ventana (valle/pico) y una escala de demanda entre 0,7 y 1,3.
- **Verde máximo** de 60 s, ahora también durante el entrenamiento.
- **Líneas base nuevas:** semáforos **actuados** de SUMO y **max-pressure** (`lineas_base.py`).

Los modelos v1 (los de la presentación) no cargan en v2 porque la observación de C1/C4 cambió. Por eso todo lo v2 lleva el prefijo `v2_` y se guarda aparte.

## 0. Preparar cada PC (una sola vez)
1. Mismo código en los 4 PCs: `git pull` del mismo commit. Nadie cambia el entorno durante la campaña.
2. Mismas versiones: SUMO **1.27.1**, `sumo-rl==1.4.5` y `pip install -r requirements.txt`.
3. Corran todos los comandos **desde la carpeta del proyecto** (`cd` a `Taller-Control`). Si la ruta tiene espacios (p.ej. `Octavo semestre`), esto es obligatorio: a SUMO se le pasan rutas relativas a esa carpeta.
4. Comprobación de que todos los PCs simulan igual: corran **exactamente** este comando (con `--semillas 10 --escalas 1.0`; sin eso promedia 5 semillas y 3 escalas y no se puede comparar). La tabla debe dar exactamente lo mismo en los 4:
   ```
   python evaluar_comparacion.py --entorno v2 --politicas --lineas-base tiempo_fijo --semillas 10 --escalas 1.0
   ```
   Resultado esperado (PC1): `valle 1.0 fijo tiempo_fijo 5.63 134.16 53.12 1375.0 345.03 11.87 32.27`

## 1. Prueba corta (la primera noche o una tarde)
```
python campana.py --pc N --prueba
```
Entrena 20 episodios y estima cuánto dura una corrida de 150. En el PC donde se desarrolló: ~34 s por episodio, o sea ~1,5 h por corrida con una sola a la vez. Al final borren `resultados/prueba_*` y los `*prueba_*.pth`.

## 2. Campaña
| PC | Corridas (150 episodios c/u) |
|---|---|
| PC1 | v2_base ×3 semillas, v2_sin_rnd semilla 0 |
| PC2 | v2_penaliza_cambio ×3 (P = 2,0), v2_sin_rnd semilla 1 |
| PC3 | v2_prioriza_espera ×3 (w2 = 2,0), v2_sin_rnd semilla 2 |
| PC4 | v2_peatones_alto ×3 (w3 = 5,0), v2_mas_exploracion ×3 (entropía 0,05) |

```
python campana.py --pc N              # 3 corridas a la vez (ajustable con --paralelo)
```
- La consola de cada corrida queda en `resultados/<corrida>/consola.log`.
- Si algo se corta, vuelvan a lanzar el mismo comando: las corridas con pesos guardados se saltan.
- `python campana.py --listar` muestra el reparto.

## 3. Evaluación (cada PC evalúa lo que entrenó)
```
python evaluar_comparacion.py --entorno v2 --ventanas valle pico --procesos 8
```
Sin `--politicas`, evalúa todas las políticas v2 que tengan pesos en ese PC. **Un solo PC** (PC1) corre además las líneas base:
```
python evaluar_comparacion.py --entorno v2 --politicas --lineas-base tiempo_fijo actuado max_pressure --ventanas valle pico --procesos 8
```
Matriz: 5 semillas de SUMO × escalas 0,8 / 1,0 / 1,2 × 2 ventanas.

## 4. Juntar resultados en un PC
Cada PC copia a la carpeta compartida (o sube a su rama `resultados-pcN`):
- `resultados/v2_*_seed*/` (curvas de entrenamiento y `config.json`)
- `actor_mappo_*_v2_*.pth` y `critic_mappo_v2_*.pth`
- `resultados_eval/episodios_v2/` (episodios de evaluación; está en `.gitignore` porque pesa)

En el PC que consolida:
```
python exportar_datos.py --v2         # -> datos_dashboard/v2/
```
`evaluacion_configs.csv` es la tabla principal para el informe. Trae cada configuración contra cada línea base, con IC95 entre las 3 semillas de entrenamiento.

## Hallazgo previo, para tenerlo en cuenta
En una prueba con la semilla 10, **max-pressure le gana al agente v1** en espera vehicular (1,70 s frente a 2,66 s), pero hace esperar a los peatones unos 35 s, contra unos 12 s con tiempo fijo. El argumento fuerte de la campaña v2 es mostrar que RND-MAPPO logra un mejor compromiso entre vehículos y peatones.
