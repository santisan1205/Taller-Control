"""
Campaña de entrenamiento v2 repartida entre los 4 PCs del equipo.

Cada PC corre su lista fija de corridas (ASIGNACION), varias a la vez. Todas
usan el entorno v2 (peatones en C1/C4, demanda variable con valle y hora pico,
verde máximo) y 150 episodios. La salida de consola de cada corrida queda en
resultados/<corrida>/consola.log.

Uso:
  python campana.py --pc 1             # corre las corridas del PC 1, 3 a la vez
  python campana.py --pc 1 --prueba    # 1 corrida corta (20 episodios) para medir tiempos
  python campana.py --listar           # muestra el reparto completo

Una corrida que ya tiene su critic_mappo_<corrida>.pth se salta, así que si
algo se interrumpe basta con volver a lanzar el mismo comando.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from evaluar_comparacion import CONFIGS_V2, SEMILLAS_ENTRENAMIENTO_V2

RAIZ = os.path.dirname(os.path.abspath(__file__))
EPISODIOS = 150
ARGS_COMUNES = ['--peatones', '--demanda-variable']

# Lo que cambia cada configuración respecto a los defaults de train.py
ARGS_CONFIG = {
    'v2_base': [],
    'v2_penaliza_cambio': ['--p-cambio', '2.0'],
    'v2_prioriza_espera': ['--w2', '2.0'],
    'v2_peatones_alto': ['--w3', '5.0'],
    'v2_mas_exploracion': ['--entropy-coef', '0.05'],
    'v2_sin_rnd': ['--sin-rnd'],
}
assert set(ARGS_CONFIG) == set(CONFIGS_V2), 'ARGS_CONFIG y evaluar_comparacion.CONFIGS_V2 deben coincidir'

# PC -> [(configuración, semilla)]. 18 corridas en total, repartidas según la
# prueba de 20 episodios (h por corrida de 150, una a la vez): PC1 2,0 h,
# PC2 4,1 h, PC3 6,1 h, PC4 3,8 h. El PC1 lleva 3 tandas de 3 y los demás una,
# para que todos terminen en ~7-9 h. El PC3, el más lento, lleva
# v2_mas_exploracion, que es la primera configuración que se recorta si no alcanza.
def _tres_semillas(config):
    return [(config, s) for s in SEMILLAS_ENTRENAMIENTO_V2]

ASIGNACION = {
    1: _tres_semillas('v2_base') + _tres_semillas('v2_sin_rnd') + _tres_semillas('v2_prioriza_espera'),
    2: _tres_semillas('v2_penaliza_cambio'),
    3: _tres_semillas('v2_mas_exploracion'),
    4: _tres_semillas('v2_peatones_alto'),
}


def nombre_corrida(config, semilla):
    return f'{config}_seed{semilla}'  # mismo orden que usa train.py: <tag>_seed<seed>


def comando(config, semilla, episodios):
    return [sys.executable, os.path.join(RAIZ, 'train.py'), '--tag', config, '--seed', str(semilla),
            '--episodios', str(episodios), *ARGS_COMUNES, *ARGS_CONFIG[config]]


def correr(config, semilla, episodios, hilos_torch):
    corrida = nombre_corrida(config, semilla)
    if os.path.exists(os.path.join(RAIZ, f'critic_mappo_{corrida}.pth')):
        print(f'[saltada] {corrida}: ya tiene pesos guardados')
        return corrida, 0, 0.0
    carpeta = os.path.join(RAIZ, 'resultados', corrida)
    os.makedirs(carpeta, exist_ok=True)
    # Varias corridas a la vez: se limitan los hilos de torch para que no se peleen la CPU
    entorno = {**os.environ, 'OMP_NUM_THREADS': str(hilos_torch), 'MKL_NUM_THREADS': str(hilos_torch)}
    inicio = time.time()
    print(f'[inicio] {corrida}')
    with open(os.path.join(carpeta, 'consola.log'), 'w', encoding='utf-8') as log:
        codigo = subprocess.call(comando(config, semilla, episodios), cwd=RAIZ, stdout=log,
                                 stderr=subprocess.STDOUT, env=entorno)
    horas = (time.time() - inicio) / 3600
    print(f'[{"fin" if codigo == 0 else "ERROR"}] {corrida} en {horas:.1f} h'
          + ('' if codigo == 0 else f' (codigo {codigo}, ver {carpeta}/consola.log)'))
    return corrida, codigo, horas


def main():
    parser = argparse.ArgumentParser(description='Campaña de entrenamiento v2 repartida entre 4 PCs.')
    parser.add_argument('--pc', type=int, choices=sorted(ASIGNACION))
    parser.add_argument('--paralelo', type=int, default=3, help='Corridas simultaneas en este PC')
    parser.add_argument('--hilos-torch', type=int, default=2, help='Hilos de CPU de torch por corrida')
    parser.add_argument('--prueba', action='store_true',
                        help='Solo la primera corrida del PC, con 20 episodios y tag prueba_* (no cuenta para la campaña)')
    parser.add_argument('--listar', action='store_true', help='Muestra el reparto de corridas y sale')
    args = parser.parse_args()

    if args.listar:
        for pc, corridas in ASIGNACION.items():
            print(f'PC{pc}: ' + ', '.join(nombre_corrida(c, s) for c, s in corridas))
        return
    if args.pc is None:
        parser.error('falta --pc (o --listar)')

    if args.prueba:
        config, semilla = ASIGNACION[args.pc][0]
        corrida = f'prueba_{nombre_corrida(config, semilla)}'
        print(f'Prueba: {corrida}, 20 episodios. Los resultados quedan en resultados/{corrida}/ y se pueden borrar.')
        cmd = comando(config, semilla, 20)
        cmd[cmd.index('--tag') + 1] = f'prueba_{config}'
        inicio = time.time()
        subprocess.check_call(cmd, cwd=RAIZ)
        print(f'Duracion: {(time.time() - inicio) / 60:.1f} min -> ~{(time.time() - inicio) / 20 * EPISODIOS / 3600:.1f} h '
              f'por corrida de {EPISODIOS} episodios (con 1 corrida a la vez)')
        return

    corridas = ASIGNACION[args.pc]
    print(f'PC{args.pc}: {len(corridas)} corridas de {EPISODIOS} episodios, {args.paralelo} a la vez')
    with ThreadPoolExecutor(max_workers=args.paralelo) as pool:
        resultados = list(pool.map(lambda cs: correr(*cs, EPISODIOS, args.hilos_torch), corridas))
    fallidas = [c for c, codigo, _ in resultados if codigo != 0]
    print('Listo.' if not fallidas else f'Fallaron: {fallidas}')
    sys.exit(1 if fallidas else 0)


if __name__ == '__main__':
    main()
