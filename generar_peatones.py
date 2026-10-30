"""
Genera 2026-09-12-17-03-26/peatones.rou.xml: la demanda peatonal del escenario.

Solo hay peatones en dos pasos peatonales semaforizados (verificado con la
salida FCD de SUMO: ninguna persona usa otro crossing):
  C4: :cluster_10564135212_13571629279_13571629299_c0 (cruza 1169160817#0, linkIndex 6)
  C1: :cluster_10564135163_13571629282_5146149679_c0  (cruza 1133034637#3, linkIndex 10)

Se escriben personas individuales con su hora de salida en vez de <personFlow>:
con --begin > 0 (ventana de hora pico) SUMO no insertaba las personas de un
flujo que empezaba antes de --begin.
"""
import argparse
import os
import random

from setup_entorno import CARPETA_SUMO

# Paso peatonal -> (acera de un lado, acera del otro). Se generan ambos sentidos.
RUTAS_PEATONALES = {
    'C4': ('1374486328#1', '1133034629#1'),
    'C1': ('1374843820#0', '1374843820#1'),
}
FIN = 8000                 # igual que la demanda vehicular
PICO = (5000, 7000)        # igual que los flujos de hora pico de flujos_especificos.rou.xml


def generar(salida, por_hora_valle=120, por_hora_pico=240, seed=0):
    """por_hora_*: personas por hora y por sentido en cada paso peatonal.
    Las llegadas son de Poisson, con la semilla fija para que todos los PCs
    tengan la misma demanda."""
    rng = random.Random(seed)
    personas = []
    for cruce, (lado_a, lado_b) in RUTAS_PEATONALES.items():
        for sentido, (origen, destino) in (('ida', (lado_a, lado_b)), ('vuelta', (lado_b, lado_a))):
            t = 0.0
            while True:
                tasa = por_hora_pico if PICO[0] <= t < PICO[1] else por_hora_valle
                t += rng.expovariate(tasa / 3600)
                if t >= FIN:
                    break
                personas.append((round(t, 2), f'peaton_{cruce}_{sentido}', origen, destino))

    personas.sort()
    with open(salida, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!-- Generado por generar_peatones.py: no editar a mano -->\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        for i, (depart, prefijo, origen, destino) in enumerate(personas):
            f.write(f'    <person id="{prefijo}_{i}" depart="{depart:.2f}">'
                    f'<walk from="{origen}" to="{destino}"/></person>\n')
        f.write('</routes>\n')
    print(f'{len(personas)} personas escritas en {salida}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera la demanda peatonal de C1 y C4.')
    parser.add_argument('--valle', type=float, default=120, help='Personas/h por sentido fuera de la hora pico')
    parser.add_argument('--pico', type=float, default=240, help='Personas/h por sentido en la hora pico')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--salida', default=os.path.join(CARPETA_SUMO, 'peatones.rou.xml'))
    args = parser.parse_args()
    generar(args.salida, args.valle, args.pico, args.seed)
