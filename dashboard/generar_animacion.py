"""
Genera dashboard/animacion.json: un tramo de simulación de SUMO para ver en el
dashboard los vehículos moviéndose, con los semáforos de tiempo fijo y con el
agente RND-MAPPO, con exactamente la misma demanda (misma semilla de SUMO).

SUMO graba la posición de cada vehículo y peatón cada segundo (--fcd-output)
y el estado de cada semáforo (SaveTLSStates). Aquí se recorta a la zona de los
4 cruces y se guarda en un JSON compacto que el navegador reproduce, así que
para ver la animación no hace falta tener SUMO instalado.

Uso (desde la raíz del proyecto; requiere SUMO):
    python dashboard/generar_animacion.py
    python dashboard/generar_animacion.py --ventana valle --escala 1.2 --minutos 5
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from setup_entorno import crear_entorno_sumo, CRUCES, RED_XML, rutas_demanda, VENTANAS  # noqa: E402
from evaluar_comparacion import (POLITICAS, DELTA_TIME, _cargar_actores, _correr_episodio,  # noqa: E402
                                 _ruta_para_sumo)

SALIDA = os.path.join(RAIZ, 'dashboard', 'animacion.json')
MARGEN = 70          # m alrededor de los cruces
VELOCIDAD_DETENIDO = 0.1  # m/s, el mismo umbral que usa sumo-rl para "detenido"
# Tamaño (largo, ancho) en m según el tipo de vehículo de SUMO
TIPOS = {'motorcycle': 1, 'bus': 2}  # 0 = carro


def geometria_red(red=RED_XML):
    """Carriles dentro de la zona de los cruces, posición de los cruces y la línea
    de pare de cada enlace semaforizado (para dibujar el color del semáforo)."""
    raiz = ET.parse(red).getroot()
    cruces = {CRUCES[j.get('id')]: (float(j.get('x')), float(j.get('y')))
              for j in raiz.iter('junction') if j.get('id') in CRUCES}
    xs, ys = [p[0] for p in cruces.values()], [p[1] for p in cruces.values()]
    caja = (min(xs) - MARGEN, min(ys) - MARGEN, max(xs) + MARGEN, max(ys) + MARGEN)
    dentro = lambda x, y: caja[0] <= x <= caja[2] and caja[1] <= y <= caja[3]

    carriles, forma_carril = [], {}
    for edge in raiz.iter('edge'):
        funcion = edge.get('function')
        if funcion in ('internal', 'walkingarea'):
            continue
        for lane in edge.iter('lane'):
            puntos = [tuple(map(float, p.split(','))) for p in lane.get('shape').split()]
            forma_carril[lane.get('id')] = puntos
            if not any(dentro(x, y) for x, y in puntos):
                continue
            tipo = 'cruce_peatonal' if funcion == 'crossing' else (
                'anden' if lane.get('allow') == 'pedestrian' else 'via')
            carriles.append({'t': tipo, 'a': round(float(lane.get('width', 3.2)), 1),
                             'p': [[round(x - caja[0], 1), round(y - caja[1], 1)] for x, y in puntos]})

    # Área interior de cada intersección (sin ella las calles se ven cortadas en los cruces)
    intersecciones = []
    for j in raiz.iter('junction'):
        if j.get('type') == 'internal' or not j.get('shape'):
            continue
        puntos = [tuple(map(float, p.split(','))) for p in j.get('shape').split()]
        if len(puntos) >= 3 and any(dentro(x, y) for x, y in puntos):
            intersecciones.append([[round(x - caja[0], 1), round(y - caja[1], 1)] for x, y in puntos])

    # Línea de pare: final del carril de entrada de cada enlace (linkIndex) de cada semáforo
    lineas_pare = {}
    for c in raiz.iter('connection'):
        tl = c.get('tl')
        if tl not in CRUCES or c.get('linkIndex') is None:
            continue
        carril = f'{c.get("from")}_{c.get("fromLane")}'
        if carril not in forma_carril:
            continue
        (x0, y0), (x, y) = forma_carril[carril][-2], forma_carril[carril][-1]
        lineas_pare.setdefault(tl, {}).setdefault(carril, {
            'x': round(x - caja[0], 1), 'y': round(y - caja[1], 1),
            # dirección del carril al llegar al cruce (para dibujar la barra del semáforo atravesada)
            'ang': round(math.degrees(math.atan2(y - y0, x - x0))),
            'enlaces': []})['enlaces'].append(int(c.get('linkIndex')))
    semaforos = [{'cruce': CRUCES[tl], 'pare': list(pares.values())} for tl, pares in lineas_pare.items()]
    cruces_rel = {c: [round(x - caja[0], 1), round(y - caja[1], 1)] for c, (x, y) in cruces.items()}
    return caja, carriles, intersecciones, cruces_rel, semaforos


def simular(modo, politica, ventana, escala, semilla, segundos, carpeta):
    """Corre la simulación y deja fcd.xml y tls.xml en carpeta."""
    fcd = os.path.join(carpeta, f'{modo}_fcd.xml')
    tls_add = os.path.join(carpeta, f'{modo}_tls.add.xml')
    tls_out = os.path.join(carpeta, f'{modo}_tls.xml')
    with open(tls_add, 'w', encoding='utf-8') as f:
        f.write('<additional>\n')
        for ts in CRUCES:
            f.write(f'  <timedEvent type="SaveTLSStates" source="{ts}" dest="{_ruta_para_sumo(tls_out)}"/>\n')
        f.write('</additional>\n')
    cmd = (f'--fcd-output {_ruta_para_sumo(fcd)} --scale {escala:g} '
           f'--additional-files {_ruta_para_sumo(tls_add)} --no-step-log --no-warnings')
    fijo = modo == 'fijo'
    env = crear_entorno_sumo(RED_XML, rutas_demanda(peatones=True), use_gui=False, sumo_seed=semilla,
                             out_csv_name=None, fixed_ts=fijo, additional_sumo_cmd=cmd,
                             peatones=True, begin_time=VENTANAS[ventana])
    obs, _ = env.reset()
    actores = None
    if not fijo:
        agentes = env.agents
        actores = _cargar_actores(POLITICAS[politica], agentes,
                                  {a: env.observation_space(a).shape[0] for a in agentes},
                                  {a: env.action_space(a).n for a in agentes})
        torch.manual_seed(semilla)
    _correr_episodio(env, obs, segundos // DELTA_TIME, actores=actores)  # cierra el entorno
    return fcd, tls_out


def leer_fcd(fcd, caja, t_ini, t_fin):
    """
    Un cuadro por segundo: vehículos [id, x, y, ángulo, detenido, tipo] y peatones [x, y],
    con posiciones en decímetros relativas a la esquina de la caja. También cuenta los
    vehículos detenidos en la zona (lo que muestra el marcador).
    """
    ids, cuadros = {}, []
    for _, ts in ET.iterparse(fcd, events=('end',)):
        if ts.tag != 'timestep':
            continue
        t = float(ts.get('time'))
        if t_ini <= t < t_fin:
            vehiculos, peatones, detenidos = [], [], 0
            for v in ts:
                x, y = float(v.get('x')), float(v.get('y'))
                if not (caja[0] <= x <= caja[2] and caja[1] <= y <= caja[3]):
                    continue
                xr, yr = round((x - caja[0]) * 10), round((y - caja[1]) * 10)
                if v.tag == 'person':
                    peatones.append([xr, yr])
                    continue
                parado = float(v.get('speed')) < VELOCIDAD_DETENIDO
                detenidos += parado
                tipo = next((n for clave, n in TIPOS.items() if clave in (v.get('type') or '')), 0)
                vehiculos.append([ids.setdefault(v.get('id'), len(ids)), xr, yr, round(float(v.get('angle'))),
                                  int(parado), tipo])
            cuadros.append({'v': vehiculos, 'p': peatones, 'd': detenidos})
        ts.clear()
    return cuadros


def leer_tls(tls_out, t_ini, t_fin):
    """Estado de cada semáforo segundo a segundo: {cruce: [estado por segundo]}."""
    estados = {c: {} for c in CRUCES.values()}
    for e in ET.parse(tls_out).getroot().iter('tlsState'):
        t = int(float(e.get('time')))
        if t_ini <= t < t_fin and e.get('id') in CRUCES:
            estados[CRUCES[e.get('id')]][t - int(t_ini)] = e.get('state')
    n = int(t_fin - t_ini)
    salida = {}
    for cruce, por_t in estados.items():
        serie, ultimo = [], None
        for t in range(n):  # se rellena con el último estado conocido
            ultimo = por_t.get(t, ultimo)
            serie.append(ultimo)
        salida[cruce] = serie
    return salida


def main():
    parser = argparse.ArgumentParser(description='Genera la animación de SUMO para el dashboard.')
    parser.add_argument('--politica', default='v2_base_seed0', help='Política del entorno v2 (con sus pesos en disco)')
    parser.add_argument('--ventana', default='pico', choices=sorted(VENTANAS))
    parser.add_argument('--escala', type=float, default=1.0)
    parser.add_argument('--semilla', type=int, default=10)
    parser.add_argument('--desde', type=int, default=600,
                        help='Segundos desde el inicio de la ventana en que empieza la grabación '
                             '(los primeros minutos la red se está llenando)')
    parser.add_argument('--minutos', type=float, default=10)
    args = parser.parse_args()

    caja, carriles, intersecciones, cruces, semaforos = geometria_red()
    t_ini = VENTANAS[args.ventana] + args.desde
    t_fin = t_ini + int(args.minutos * 60)
    carpeta = tempfile.mkdtemp(prefix='animacion_')
    try:
        datos = {}
        for modo in ('fijo', 'rl'):
            print(f'Simulando {modo}...')
            fcd, tls = simular(modo, args.politica, args.ventana, args.escala, args.semilla,
                               t_fin - VENTANAS[args.ventana], carpeta)
            datos[modo] = {'cuadros': leer_fcd(fcd, caja, t_ini, t_fin), 'semaforos': leer_tls(tls, t_ini, t_fin)}
            print(f'  {len(datos[modo]["cuadros"])} segundos grabados')
    finally:
        shutil.rmtree(carpeta, ignore_errors=True)

    animacion = {
        'escenario': {'politica': args.politica, 'ventana': args.ventana, 'escala': args.escala,
                      'semilla': args.semilla, 't_inicio': t_ini, 'segundos': t_fin - t_ini},
        'ancho': round(caja[2] - caja[0], 1), 'alto': round(caja[3] - caja[1], 1),
        'carriles': carriles, 'intersecciones': intersecciones, 'cruces': cruces, 'semaforos': semaforos, **datos,
    }
    with open(SALIDA, 'w', encoding='utf-8') as f:
        json.dump(animacion, f, separators=(',', ':'))
    detenidos = {m: sum(c['d'] for c in datos[m]['cuadros']) / max(1, len(datos[m]['cuadros'])) for m in datos}
    print(f'Animación guardada en {os.path.relpath(SALIDA, RAIZ)} ({os.path.getsize(SALIDA) / 1e6:.1f} MB). '
          f'Detenidos promedio en la zona: fijo {detenidos["fijo"]:.1f}, agente {detenidos["rl"]:.1f}')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # tildes en la consola de Windows
    except (AttributeError, ValueError):
        pass
    main()
