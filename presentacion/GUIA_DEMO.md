# Guía: demo en vivo (tiempo fijo vs. agente)

La demo abre **dos ventanas de SUMO lado a lado con exactamente el mismo tráfico**:

- **Izquierda:** los semáforos de tiempo fijo, como funcionan hoy.
- **Derecha:** el agente RND-MAPPO entrenado.

Mientras corre, la terminal muestra un marcador con los vehículos detenidos, la espera media y los vehículos que llegaron a su destino.

---

## Opción A: en el computador donde ya está todo instalado

1. Abre VS Code en la carpeta `Taller-Control`.
2. Abre una terminal (**Ctrl + ñ**). Debe estar en la carpeta del proyecto.
3. Verifica que estás en la rama correcta:
   ```powershell
   git checkout metricas-dashboard
   git pull
   ```
4. Haz una prueba rápida (unos 30 s; las ventanas se abren y se cierran solas):
   ```powershell
   .\.venv\Scripts\python.exe demo_en_vivo.py --minutos 3 --delay 0
   ```
5. Corre la versión para presentar:
   ```powershell
   .\.venv\Scripts\python.exe demo_en_vivo.py --minutos 10 --delay 300 --escala 1.2
   ```

## Opción B: en otro computador (preparar el día anterior)

Hay que instalar tres cosas: SUMO, el código y las librerías de Python.

1. **Instalar SUMO:** descarga el instalador para Windows desde <https://eclipse.dev/sumo/> e instálalo con las opciones por defecto.
   - Cierra y vuelve a abrir la terminal. Luego verifica:
     ```powershell
     echo $env:SUMO_HOME
     ```
     Debe mostrar algo como `C:\Program Files (x86)\Eclipse\Sumo\`.
   - Si sale vacío, agrega la variable de entorno `SUMO_HOME` con esa ruta (Configuración de Windows → "Editar las variables de entorno del sistema") y reabre la terminal.
2. **Descargar el código:**
   ```powershell
   git clone https://github.com/santisan1205/Taller-Control.git
   cd Taller-Control
   git checkout metricas-dashboard
   ```
3. **Crear el entorno de Python** (el proyecto usa Python 3.14; con una versión muy vieja las librerías pueden no instalar):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   Si `torch` tarda mucho en instalar, es normal: pesa bastante.
4. Sigue desde el paso 4 de la opción A.

---

## Parámetros útiles

| Parámetro | Qué hace | Recomendado para presentar |
|---|---|---|
| `--minutos` | Minutos de tráfico simulado (máximo 60) | `10` |
| `--delay` | Pausa en milisegundos por segundo simulado. Más alto = más lento | `300` (prueba entre 200 y 500) |
| `--escala` | Multiplica la cantidad de carros (1.2 = 20 % más) | `1.2`: la diferencia se nota más |
| `--semilla` | Fija el azar del tráfico, para repetir la misma demo | `10` (por defecto) |
| `--politica` | Qué modelo entrenado usar | `base` (por defecto, el mejor) |
| `--reporte-cada` | Cada cuántos minutos simulados se imprime el marcador | `2` para una demo de 10 min |

Duración aproximada: `--minutos 10 --delay 300` dura unos 3 minutos reales. Con `--delay 100` va unas 3 veces más rápido. Confírmalo en el ensayo, porque depende del computador.

## Mientras corre

- **Rueda del mouse:** zoom. **Arrastrar:** mover la vista. La vista arranca centrada en los 4 cruces.
- **Campo "Delay"** (barra superior de cada ventana): cambia la velocidad en vivo.
- **Pasar el mouse sobre un carro:** SUMO muestra su información.
- **Ctrl + C** en la terminal: detiene la demo.
- Las líneas `Step #0.00` y `Retrying in 1 seconds` del arranque son normales.

## Qué decir durante la demo

1. **Antes de arrancar:** "Las dos ventanas tienen exactamente los mismos carros, que salen a la misma hora hacia el mismo destino. Lo único distinto es quién controla los semáforos: a la izquierda el ciclo fijo, a la derecha nuestro agente."
2. **Mientras corre:** muestren cómo en la izquierda se arman colas aunque la otra calle esté vacía, mientras en la derecha el verde va hacia donde están los carros. Miren sobre todo los dos cruces grandes.
3. **Cada vez que salga una línea del marcador:** léanla. "Minuto 6: con tiempo fijo hay 24 carros detenidos; con el agente, 7."
4. **Al final:** lean la última línea, la del % de mejora.

Resultado de referencia (semilla 10, demanda normal, 15 minutos): **48 % menos vehículos detenidos y 57 % menos espera media**.

## Plan B: grabar un video de respaldo

Si el computador de la presentación falla, pongan un video grabado antes.

1. Corre la demo con los parámetros de la presentación.
2. Presiona **Win + Alt + R** para empezar a grabar (barra de juegos de Windows) y otra vez para parar.
3. El video queda en `Videos\Capturas`. Si la grabadora no se activa, usen OBS Studio.

## Si algo falla

| Síntoma | Solución |
|---|---|
| `No module named ...` | Falta instalar las librerías: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| No se abre ninguna ventana | Revisa `echo $env:SUMO_HOME`. Si está vacío, configúralo (opción B, paso 1). |
| `FileNotFoundError` con un `.pth` | No estás en la carpeta del proyecto, o no estás en la rama `metricas-dashboard`. |
| Una ventana queda encima de la otra | Muévelas con el mouse: siguen simulando sin problema. |
| Va muy rápido o muy lento | Ajusta `--delay`, o cámbialo en vivo en el campo "Delay" de SUMO. |
| Error de puerto o de conexión (TraCI) | Cierra todas las ventanas de SUMO que hayan quedado abiertas y vuelve a correr. |

---

## La presentación

- **Online:** https://claude.ai/artifact/RKxpbNoWQnMpv6dp5TB4vs (es privada: hay que compartirla desde el menú Share).
- **`presentacion.html`:** ábrela con doble clic en el navegador.
  - **→ / ←:** avanzar y retroceder.
  - **N:** muestra las notas del presentador (el guion de cada diapositiva).
  - **F:** pantalla completa.
  - Necesita internet solo para cargar las tipografías.
- **`presentacion.pdf`:** la misma presentación en PDF, sin notas.
- **`fuente/`:** los archivos originales de las diapositivas.
