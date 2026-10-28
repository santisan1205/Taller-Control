"""
Demo para presentar: abre dos ventanas de sumo-gui lado a lado con exactamente
el mismo tráfico, a la izquierda los semáforos de tiempo fijo y a la derecha
el agente RND-MAPPO, e imprime en la terminal un marcador en vivo.

Para que la diferencia se vea a simple vista, los vehículos detenidos se pintan
de rojo y los que avanzan de verde, y la cámara se acerca al grupo C3/C1/C4,
donde más mejora el agente.

Uso:  python demo_en_vivo.py                     (10 min, demanda x1,2: lo recomendado)
      python demo_en_vivo.py --delay 80           (más rápido)
      python demo_en_vivo.py --vista c2           (el otro cruce congestionado)
      python demo_en_vivo.py --vista todos        (los 4 cruces)
La velocidad también se puede cambiar en vivo con el campo "Delay" de sumo-gui.
"""
import argparse
import ctypes
import multiprocessing as mp
import os
import queue
import sys
import warnings

import torch

from evaluar_comparacion import POLITICAS, RED_XML, RUTAS_XML, DELTA_TIME, _cargar_actores, _correr_episodio
from setup_entorno import crear_entorno_sumo, CRUCES

VISTA = 'View #0'
ROJO = (220, 38, 38, 255)     # detenido
VERDE = (34, 197, 94, 255)    # en movimiento
VELOCIDAD_DETENIDO = 0.1      # m/s, el mismo umbral que usa sumo-rl para "stopped"

# Encuadres de la cámara: cruces a mostrar y margen (m) alrededor. C2 queda
# solo a la izquierda de la red; C3 está pegado a C1 y C4 (a menos de 50 m).
# C2 y C3 concentran la congestión (ver evaluacion_por_cruce.csv).
_ID = {cruce: ts for ts, cruce in CRUCES.items()}
VISTAS = {
    'c3': ([_ID['C3'], _ID['C1'], _ID['C4']], 70),
    'c2': ([_ID['C2']], 90),
    'congestion': ([_ID['C2'], _ID['C3']], 60),
    'todos': (list(CRUCES), 120),
}

# Colores ANSI para el marcador (la terminal de VS Code y Windows Terminal los soportan)
NARANJA, AZUL, GRIS, NEGRITA, FIN = '\033[38;5;208m', '\033[38;5;39m', '\033[90m', '\033[1m', '\033[0m'
ANCHO_BARRA = 40


def _tamano_pantalla():
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def _enfocar(sumo, cruces, margen):
    """Centra la vista de sumo-gui en los cruces dados."""
    try:
        xs, ys = zip(*(sumo.junction.getPosition(ts) for ts in cruces))
        sumo.gui.setBoundary(VISTA, min(xs) - margen, min(ys) - margen, max(xs) + margen, max(ys) + margen)
    except Exception:
        pass  # si el ID del cruce no es un junction, se deja la vista por defecto


def _colorear(sumo, colores):
    """Pinta de rojo los vehículos detenidos y de verde los que avanzan (solo
    llama a setColor cuando el color cambia, para no frenar la simulación)."""
    ids = sumo.vehicle.getIDList()
    for v in ids:
        color = ROJO if sumo.vehicle.getSpeed(v) < VELOCIDAD_DETENIDO else VERDE
        if colores.get(v) != color:
            sumo.vehicle.setColor(v, color)
            colores[v] = color
    return ids


def _simular(modo, politica, semilla, escala, pasos, delay, ventana, cola, cada_pasos, max_green, vista,
             carpeta_capturas):
    x, y, ancho, alto = ventana
    warnings.filterwarnings('ignore')  # avisos de PettingZoo que ensucian el marcador
    cmd = (f'--window-pos {x},{y} --window-size {ancho},{alto} --delay {delay} --scale {escala:g} '
           f'--no-step-log --no-warnings')
    fijo = modo == 'fijo'
    env = crear_entorno_sumo(RED_XML, RUTAS_XML, use_gui=True, sumo_seed=semilla, out_csv_name=None,
                             fixed_ts=fijo, additional_sumo_cmd=cmd, aplicar_max_green=max_green)
    obs, _ = env.reset()
    sumo = env.unwrapped.env.sumo
    _enfocar(sumo, *VISTAS[vista])

    actores = None
    if not fijo:
        agentes = env.agents
        actores = _cargar_actores(POLITICAS[politica], agentes,
                                  {a: env.observation_space(a).shape[0] for a in agentes},
                                  {a: env.action_space(a).n for a in agentes})
        torch.manual_seed(semilla)

    # Cada paso del agente avanza 5 pasos de SUMO, así que getArrivedNumber()
    # (solo el último paso) se queda corto: los que llegaron = vistos - en la red
    vistos = set()
    colores = {}
    total_llegados = [0]

    def al_paso(env, paso):
        en_red = _colorear(sumo, colores)
        vistos.update(en_red)
        total_llegados[0] = len(vistos) - len(en_red)  # _correr_episodio cierra el env al terminar
        if (paso + 1) % cada_pasos == 0:
            m = env.unwrapped.env.metrics[-1]
            cola.put((modo, paso + 1, m['system_total_stopped'], m['system_mean_waiting_time'], total_llegados[0]))
            if carpeta_capturas:
                minuto = (paso + 1) * DELTA_TIME // 60
                sumo.gui.screenshot(VISTA, os.path.join(carpeta_capturas, f'{modo}_min{minuto:02d}.png'))

    try:
        metricas, _ = _correr_episodio(env, obs, pasos, actores=actores, al_paso=al_paso)
        cola.put((modo, 'fin', metricas['system_total_stopped'].mean(),
                  metricas['system_mean_waiting_time'].mean(), total_llegados[0]))
    except Exception as e:  # p.ej. si se cierra la ventana de sumo-gui a mitad de la demo
        cola.put((modo, 'error', str(e), None, None))


def _mejora(fijo, rl):
    return (fijo - rl) / fijo * 100 if fijo else 0.0


def _barra(valor, maximo, color):
    n = round(ANCHO_BARRA * valor / maximo) if maximo else 0
    return f'{color}{"█" * n}{FIN}{" " * (ANCHO_BARRA - n)}'


def main():
    parser = argparse.ArgumentParser(description='Demo lado a lado: tiempo fijo vs. RND-MAPPO en sumo-gui.')
    parser.add_argument('--politica', default='base', choices=list(POLITICAS))
    parser.add_argument('--semilla', type=int, default=10)
    parser.add_argument('--escala', type=float, default=1.2, help='Multiplicador de la demanda (1.2 = 20 %% más carros)')
    parser.add_argument('--minutos', type=float, default=10, help='Minutos simulados (máx. 60)')
    parser.add_argument('--delay', type=int, default=120, help='ms de pausa por segundo simulado (0 = máxima velocidad)')
    parser.add_argument('--reporte-cada', type=float, default=1, help='Minutos simulados entre líneas del marcador')
    parser.add_argument('--vista', choices=list(VISTAS), default='c3',
                        help='c3 = C3 con C1 y C4 (derecha); c2 = C2 (izquierda); congestion = C2 y C3; '
                             'todos = los 4 cruces')
    parser.add_argument('--capturas', default=None, help='Carpeta donde guardar una captura de cada ventana por reporte')
    parser.add_argument('--sin-max-green', action='store_true',
                        help='No forzar el verde máximo (como en los resultados de la presentación)')
    args = parser.parse_args()

    os.system('')  # activa los colores ANSI en la consola de Windows
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # las barras (█) no existen en cp1252
    except (AttributeError, ValueError):
        pass
    if args.capturas:
        os.makedirs(args.capturas, exist_ok=True)
    pasos = int(min(args.minutos, 60) * 60 / DELTA_TIME)
    cada_pasos = max(1, int(args.reporte_cada * 60 / DELTA_TIME))
    ancho, alto = _tamano_pantalla()
    mitad = ancho // 2
    ventanas = {'fijo': (0, 0, mitad, alto - 80), 'rl': (mitad, 0, mitad, alto - 80)}

    cola = mp.Queue()
    capturas = os.path.abspath(args.capturas) if args.capturas else None
    procesos = [mp.Process(target=_simular, args=(modo, args.politica, args.semilla, args.escala, pasos,
                                                  args.delay, ventanas[modo], cola, cada_pasos,
                                                  not args.sin_max_green, args.vista, capturas))
                for modo in ('fijo', 'rl')]
    for p in procesos:
        p.start()

    print(f'\n{NEGRITA}Izquierda: {NARANJA}TIEMPO FIJO{FIN}{NEGRITA}   |   Derecha: {AZUL}AGENTE RND-MAPPO{FIN}'
          f'   {GRIS}semilla {args.semilla}, demanda x{args.escala:g}{FIN}')
    print(f'{GRIS}En las ventanas: rojo = vehículo detenido, verde = en movimiento{FIN}\n')

    reportes = {}
    terminados = set()
    maximo = 1.0  # escala de las barras: el mayor número de detenidos visto hasta ahora
    while len(terminados) < 2:
        try:
            modo, t, detenidos, espera, llegados = cola.get(timeout=1)
        except queue.Empty:
            if not any(p.is_alive() for p in procesos):
                break
            continue
        if t == 'error':
            print(f'[{modo}] la simulación se detuvo: {detenidos}')
            terminados.add(modo)
            continue
        if t == 'fin':
            terminados.add(modo)
        reportes.setdefault(t, {})[modo] = (detenidos, espera, llegados)
        fila = reportes[t]
        if len(fila) < 2:
            continue
        (df, ef, lf), (dr, er, lr) = fila['fijo'], fila['rl']
        if t == 'fin':
            print(f'\n{NEGRITA}RESULTADO (promedio de toda la simulación){FIN}')
            print(f'  Vehículos detenidos:  {NARANJA}fijo {df:5.1f}{FIN}   {AZUL}agente {dr:5.1f}{FIN}   '
                  f'{NEGRITA}→ {_mejora(df, dr):.0f} % menos{FIN}')
            print(f'  Espera media:         {NARANJA}fijo {ef:5.1f} s{FIN} {AZUL}agente {er:5.1f} s{FIN} '
                  f'{NEGRITA}→ {_mejora(ef, er):.0f} % menos{FIN}')
            print(f'  Llegaron a destino:   {NARANJA}fijo {lf:5d}{FIN}   {AZUL}agente {lr:5d}{FIN}\n')
            continue
        maximo = max(maximo, df, dr)
        minuto = t * DELTA_TIME / 60
        print(f'{minuto:3.0f} min  {NARANJA}FIJO  {FIN} {_barra(df, maximo, NARANJA)} {df:3.0f} detenidos · '
              f'espera {ef:5.1f} s')
        variacion = f'{-_mejora(df, dr):+.0f} % detenidos'.replace('-', '−') if df else ''
        print(f'         {AZUL}AGENTE{FIN} {_barra(dr, maximo, AZUL)} {dr:3.0f} detenidos · '
              f'espera {er:5.1f} s   {NEGRITA}{variacion}{FIN}\n')

    for p in procesos:
        p.join(timeout=10)


if __name__ == '__main__':
    main()
