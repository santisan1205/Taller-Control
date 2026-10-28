"""
Demo para presentar: abre dos ventanas de sumo-gui lado a lado con exactamente
el mismo tráfico, a la izquierda los semáforos de tiempo fijo y a la derecha
el agente RND-MAPPO, e imprime en la terminal un marcador en vivo.

Uso:  python demo_en_vivo.py                       (hora completa, política base)
      python demo_en_vivo.py --minutos 15 --delay 10
      python demo_en_vivo.py --escala 1.2           (20 % más tráfico)
La velocidad también se puede cambiar en vivo con el campo "Delay" de sumo-gui.
"""
import argparse
import ctypes
import multiprocessing as mp
import queue
import warnings

import torch

from evaluar_comparacion import POLITICAS, RED_XML, RUTAS_XML, DELTA_TIME, _cargar_actores, _correr_episodio
from setup_entorno import crear_entorno_sumo, CRUCES

VISTA = 'View #0'


def _tamano_pantalla():
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def _enfocar_cruces(sumo, margen=120):
    """Centra la vista de sumo-gui en los 4 cruces semaforizados."""
    try:
        xs, ys = zip(*(sumo.junction.getPosition(ts) for ts in CRUCES))
        sumo.gui.setBoundary(VISTA, min(xs) - margen, min(ys) - margen, max(xs) + margen, max(ys) + margen)
    except Exception:
        pass  # si el ID del cruce no es un junction, se deja la vista por defecto


def _simular(modo, politica, semilla, escala, pasos, delay, ventana, cola, cada_pasos):
    x, y, ancho, alto = ventana
    warnings.filterwarnings('ignore')  # avisos de PettingZoo que ensucian el marcador
    cmd = (f'--window-pos {x},{y} --window-size {ancho},{alto} --delay {delay} --scale {escala:g} '
           f'--no-step-log --no-warnings')
    fijo = modo == 'fijo'
    env = crear_entorno_sumo(RED_XML, RUTAS_XML, use_gui=True, sumo_seed=semilla, out_csv_name=None,
                             fixed_ts=fijo, additional_sumo_cmd=cmd)
    obs, _ = env.reset()
    _enfocar_cruces(env.unwrapped.env.sumo)

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

    def llegados():
        return len(vistos) - env.unwrapped.env.sumo.vehicle.getIDCount()

    def al_paso(env, paso):
        vistos.update(env.unwrapped.env.sumo.vehicle.getIDList())
        if (paso + 1) % cada_pasos == 0:
            m = env.unwrapped.env.metrics[-1]
            cola.put((modo, paso + 1, m['system_total_stopped'], m['system_mean_waiting_time'], llegados()))

    total_llegados = [0]

    def al_paso_final(env, paso):
        al_paso(env, paso)
        total_llegados[0] = llegados()  # _correr_episodio cierra el env al terminar

    try:
        metricas, _ = _correr_episodio(env, obs, pasos, actores=actores, al_paso=al_paso_final)
        cola.put((modo, 'fin', metricas['system_total_stopped'].mean(),
                  metricas['system_mean_waiting_time'].mean(), total_llegados[0]))
    except Exception as e:  # p.ej. si se cierra la ventana de sumo-gui a mitad de la demo
        cola.put((modo, 'error', str(e), None, None))


def _mejora(fijo, rl):
    return (fijo - rl) / fijo * 100 if fijo else 0.0


def main():
    parser = argparse.ArgumentParser(description='Demo lado a lado: tiempo fijo vs. RND-MAPPO en sumo-gui.')
    parser.add_argument('--politica', default='base', choices=list(POLITICAS))
    parser.add_argument('--semilla', type=int, default=10)
    parser.add_argument('--escala', type=float, default=1.0, help='Multiplicador de la demanda')
    parser.add_argument('--minutos', type=float, default=60, help='Minutos simulados (máx. 60)')
    parser.add_argument('--delay', type=int, default=20, help='ms de pausa por paso de SUMO en la GUI (0 = máxima velocidad)')
    parser.add_argument('--reporte-cada', type=float, default=5, help='Minutos simulados entre líneas del marcador')
    args = parser.parse_args()

    pasos = int(min(args.minutos, 60) * 60 / DELTA_TIME)
    cada_pasos = max(1, int(args.reporte_cada * 60 / DELTA_TIME))
    ancho, alto = _tamano_pantalla()
    mitad = ancho // 2
    ventanas = {'fijo': (0, 0, mitad, alto - 80), 'rl': (mitad, 0, mitad, alto - 80)}

    cola = mp.Queue()
    procesos = [mp.Process(target=_simular, args=(modo, args.politica, args.semilla, args.escala, pasos,
                                                  args.delay, ventanas[modo], cola, cada_pasos))
                for modo in ('fijo', 'rl')]
    for p in procesos:
        p.start()

    print(f'\nIzquierda: TIEMPO FIJO   |   Derecha: RND-MAPPO ({args.politica})   '
          f'semilla {args.semilla}, demanda x{args.escala:g}\n')
    print(f'{"min":>5} | {"detenidos fijo":>14} {"detenidos RL":>12} | {"espera fijo":>11} {"espera RL":>9} | '
          f'{"llegaron fijo":>13} {"llegaron RL":>11}')
    print('-' * 95)

    reportes = {}
    terminados = set()
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
        if len(fila) == 2:
            (df, ef, lf), (dr, er, lr) = fila['fijo'], fila['rl']
            etiqueta = 'TOTAL' if t == 'fin' else f'{t * DELTA_TIME / 60:5.0f}'
            if t == 'fin':
                print('-' * 95)
            print(f'{etiqueta:>5} | {df:14.1f} {dr:12.1f} | {ef:10.1f}s {er:8.1f}s | {lf:13d} {lr:11d}')
            if t == 'fin':
                print(f'\nCon el agente: {_mejora(df, dr):.0f} % menos vehículos detenidos y '
                      f'{_mejora(ef, er):.0f} % menos espera media que con tiempo fijo '
                      f'(promedio de toda la simulación).')

    for p in procesos:
        p.join(timeout=10)


if __name__ == '__main__':
    main()
