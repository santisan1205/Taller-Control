# Diccionario de datos: `datos_dashboard/`

Todos los CSV se generan con `python exportar_datos.py` (después de `train.py` y `evaluar_comparacion.py`). No se editan a mano: si cambian los resultados, se vuelve a correr el script.

Formato: separador `,`, decimales con `.`, UTF-8, una fila por observación (formato largo, listo para filtrar y agrupar).

**Dos versiones de la evaluación:**
- **`datos_dashboard/`** (`python exportar_datos.py`): los resultados de la presentación, evaluados **sin** verde máximo. sumo-rl 1.4.5 no aplicaba el límite de 60 s.
- **`datos_dashboard/max_green/`** (`python exportar_datos.py --max-green`): los mismos archivos, con las mismas columnas, evaluados **con** el verde máximo de 60 s aplicado, que es el comportamiento corregido.

Los archivos de entrenamiento (`entrenamiento_episodios.csv`, `corridas.csv`) son iguales en las dos carpetas.

## Conceptos

| Término | Significado |
|---|---|
| **corrida** | Un entrenamiento completo con una configuración de hiperparámetros (`base`, `seed1`, `seed2`, `lr_conservador`, `penaliza_cambio`, `mas_exploracion`, `prioriza_espera`). Ver `corridas.csv`. Ojo: en `penaliza_cambio` y `prioriza_espera` los pesos de recompensa pedidos no llegaron al entorno (error ya corregido), así que entrenaron con los pesos por defecto. |
| **politica** | El modelo entrenado que se evalúa (mismo nombre que la corrida que lo produjo). `fijo` = semáforos de tiempo fijo originales (la línea base). |
| **modo** | `tiempo_fijo` (línea base), `rnd_mappo` (agente, acciones muestreadas de la política: es el modo oficial) o `rnd_mappo_greedy` (agente, siempre la acción más probable: solo como referencia). |
| **semilla** | Semilla de SUMO de la evaluación (10, 20, 30, 40, 50). Cambia la aleatoriedad de los conductores; la demanda (rutas) es la misma. Agente y tiempo fijo con la misma semilla y escala ven exactamente el mismo tráfico, así que son comparaciones **pareadas**. |
| **escala** | Multiplicador de la demanda (`--scale` de SUMO): 0.8 = 20 % menos carros, 1.0 = demanda normal (la de entrenamiento), 1.2 = 20 % más. |
| **cruce** | Intersección semaforizada, C1–C4. Ver `cruces.csv` para el ID de SUMO. |
| **episodio** | 1 hora simulada (3600 s, 720 decisiones cada 5 s). |

## `entrenamiento_episodios.csv`: curvas de aprendizaje
Una fila por corrida × episodio de entrenamiento.

| Columna | Unidad | Descripción |
|---|---|---|
| corrida | — | Nombre de la corrida |
| episodio | # | Episodio de entrenamiento (1…N) |
| espera_media | s | Promedio en el episodio del tiempo de espera medio de los vehículos en la red |
| espera_total_media | s | Promedio de la suma de esperas de todos los vehículos en la red |
| vehiculos_detenidos_media | veh | Promedio de vehículos detenidos (velocidad < 0,1 m/s) |
| vehiculos_detenidos_max | veh | Pico de vehículos detenidos en el episodio |
| velocidad_media | m/s | Promedio de la velocidad media de la red |
| C1…C4_espera_acumulada_media | s | Espera acumulada en los carriles de entrada de ese cruce (promedio del episodio) |
| C1…C4_vehiculos_detenidos_media | veh | Vehículos detenidos en ese cruce |
| C1…C4_velocidad_media | 0–1 | Velocidad media en ese cruce, normalizada por la velocidad máxima de la vía |

**Solo en corridas nuevas** (las 5 corridas actuales no las tienen; aparecen vacías hasta re-entrenar):

| Columna | Unidad | Descripción |
|---|---|---|
| recompensa_total | — | Suma de recompensas de los 4 agentes en el episodio (extrínseca + λ·intrínseca) |
| recompensa_ext | — | Parte extrínseca (reducción de colas y esperas, menos la penalización por cambio de fase) |
| recompensa_int | — | Parte intrínseca (curiosidad RND), antes de multiplicar por λ |
| lambda_rnd | 0–1 | Peso de la curiosidad en ese episodio (decae de 1 a 0) |
| rnd_loss | — | Error de la red predictora de RND (baja a medida que los estados se vuelven conocidos) |
| actor_loss / critic_loss | — | Pérdidas PPO promedio (4 agentes × épocas PPO) |
| entropia | nats | Entropía de la política: alta = explora, baja = decidida |
| kl_aprox | — | Cuánto cambió la política en la actualización |
| clip_frac | 0–1 | Fracción de muestras recortadas por el clip de PPO |
| duracion_s | s | Tiempo real que tardó el episodio |
| C1…C4_cambios_fase | # | Cambios de fase verde en el cruce durante el episodio |
| C1…C4_frac_accion_dominante | 0–1 | Fracción de decisiones que eligieron la fase más frecuente (1 = política colapsada en una fase) |

## `corridas.csv`: configuración de cada corrida
Una fila por corrida: `episodios`, `seed`, `lr_actor`, `lr_critic`, `clip_ratio`, `entropy_coef`, `ppo_epochs`, `gamma`, `lam`, `rnd_lr`, `w1` (peso de Δcola), `w2` (peso de Δespera vehicular), `w3` (peso de Δespera peatonal, sin efecto: no hay peatones), `p_cambio` (penalización por cambio de fase) y `notas`.

## `evaluacion_detalle.csv`: un episodio de evaluación por fila
Clave: `politica, modo, semilla, escala`.

| Columna | Unidad | Mejor si | Descripción |
|---|---|---|---|
| espera_media | s | ↓ | Igual que en entrenamiento (métrica histórica del proyecto) |
| espera_total_media | s | ↓ | Igual que en entrenamiento |
| vehiculos_detenidos_media / _max | veh | ↓ | Promedio y pico de vehículos detenidos |
| velocidad_media | m/s | ↑ | Velocidad media de la red |
| viajes_completados | veh | ↑ | Vehículos que llegaron a su destino dentro de la hora (**throughput**) |
| viajes_sin_terminar | veh | ↓ | Vehículos que seguían en la red al terminar la hora |
| vehiculos_sin_insertar | veh | ↓ | Vehículos que no pudieron entrar a la red por congestión |
| vehiculos_por_hora | veh/h | ↑ | Viajes completados por hora |
| tiempo_viaje_medio / _p95 | s | ↓ | Duración de los viajes completados: media y percentil 95 (el 5 % peor) |
| perdida_tiempo_media | s | ↓ | Tiempo perdido por viaje frente a ir a velocidad libre (`timeLoss`) |
| espera_viaje_media / _p95 | s | ↓ | Tiempo total detenido por viaje: media y percentil 95 |
| paradas_por_vehiculo | # | ↓ | Número medio de detenciones por viaje |
| demora_insercion_media | s | ↓ | Espera media para poder entrar a la red |
| teleports | # | ↓ | Vehículos teletransportados por SUMO (debería ser 0: está desactivado) |
| co2_total_kg | kg | ↓ | CO₂ emitido por todos los vehículos en la hora |
| combustible_total_kg | kg | ↓ | Combustible consumido en la hora |
| co2_g_por_km | g/km | ↓ | CO₂ por kilómetro recorrido |
| cambios_fase_total | # | — | Cambios de fase verde sumando los 4 cruces |

Las métricas de viaje (tiempo de viaje, pérdida de tiempo, espera por viaje, paradas, demora de inserción) se calculan solo sobre viajes completados.

## `evaluacion_resumen.csv`: agente vs. tiempo fijo (para KPIs)
Una fila por `politica × modo × escala × metrica`.

| Columna | Descripción |
|---|---|
| metrica | Nombre de la columna de `evaluacion_detalle.csv` |
| mejor_si | `menor` o `mayor` |
| n_semillas | Número de semillas promediadas |
| media, std | Media y desviación estándar entre semillas |
| ic95_inf, ic95_sup | Intervalo de confianza del 95 % de la media (t de Student) |
| media_fijo | Media del tiempo fijo en las mismas semillas y escala |
| mejora_pct | Mejora frente a tiempo fijo en %: **positivo = el agente es mejor**, ya corregido por el sentido de la métrica |
| p_valor | Prueba t pareada contra tiempo fijo. < 0,05 = diferencia estadísticamente significativa. Vacío para la propia línea base. |

## `evaluacion_por_cruce.csv`: desempeño por intersección
Clave: `politica, modo, semilla, escala, cruce`.

| Columna | Unidad | Descripción |
|---|---|---|
| id_sumo | — | ID del cruce en la red SUMO |
| espera_acumulada_media | s | Espera acumulada en los carriles de entrada (promedio de la hora) |
| vehiculos_detenidos_media / _max | veh | Promedio y pico de vehículos detenidos en el cruce |
| velocidad_media | 0–1 | Velocidad normalizada en el cruce |
| cambios_fase | # | Cambios de fase verde en la hora |
| verde_medio_s | s | Duración media de cada verde (3600 = nunca cambió de fase) |

## `evaluacion_series_tiempo.csv`: evolución dentro de la hora (escala 1.0)
Una fila cada 5 s simulados por episodio de evaluación: `politica, modo, semilla, escala, tiempo_s, espera_media` (s), `vehiculos_detenidos` (veh), `velocidad_media` (m/s), `C1…C4_detenidos` (veh). Sirve para gráficos de línea agente vs. tiempo fijo.

## `cruces.csv`
Mapeo `cruce` (C1–C4) → `id_sumo`.

## Advertencias para interpretar
- La demanda de las rutas es fija; las semillas solo cambian el comportamiento de los conductores, por eso la variación entre semillas es pequeña. La robustez se mide mejor comparando escalas.
- El episodio dura 3600 s, pero la demanda está definida hasta ~8000 s y los flujos de "hora pico" empiezan en 5000 s, así que **la hora pico no está en estos datos**.
- No hay peatones en el escenario.
