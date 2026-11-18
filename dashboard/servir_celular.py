"""Sirve el dashboard en la red Wi-Fi para abrirlo desde el celular con un código QR.

Uso (desde la raíz del proyecto):
    python dashboard/servir_celular.py            # puerto 8000
    python dashboard/servir_celular.py --puerto 8080

Abre el dashboard en este computador con la IP de la red (no con localhost) y muestra el
código QR en pantalla y en la consola. El celular debe estar en la misma red Wi-Fi. Si la
red no deja que los equipos se vean entre sí (pasa en algunas redes de universidad), hay
que conectar el computador al hotspot del celular, o usar el enlace público de GitHub Pages.

La primera vez Windows puede preguntar si deja que Python reciba conexiones: hay que
permitirlo en redes privadas.
"""
import argparse
import functools
import http.server
import socket
import sys
import threading
import webbrowser
from pathlib import Path

CARPETA = Path(__file__).resolve().parent


def ip_local():
    """IP de este computador en la red (no envía nada: solo pregunta qué interfaz usaría)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(('10.255.255.255', 1))
            return s.getsockname()[0]
        except OSError:
            return '127.0.0.1'


class Manejador(http.server.SimpleHTTPRequestHandler):
    def log_message(self, formato, *args):  # consola limpia: solo errores
        if len(args) > 1 and str(args[1]).startswith(('4', '5')) and 'citadel.db.gz' not in str(args[0]):
            super().log_message(formato, *args)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--puerto', type=int, default=8000)
    p.add_argument('--sin-navegador', action='store_true', help='No abre el navegador en este computador')
    a = p.parse_args()

    if not (CARPETA / 'citadel.db.gz').exists() and not (CARPETA / 'citadel.db').exists():
        sys.exit('Falta la base: corre python dashboard/construir_bd.py')

    ip = ip_local()
    url = f'http://{ip}:{a.puerto}/'
    servidor = http.server.ThreadingHTTPServer(('0.0.0.0', a.puerto),
                                               functools.partial(Manejador, directory=str(CARPETA)))

    print(f'\nDashboard en la red: {url}')
    if ip == '127.0.0.1':
        print('No se encontró una red: conecta el computador al Wi-Fi o al hotspot del celular.')
    try:
        import segno
        print('Escanea con la cámara del celular (misma red Wi-Fi):\n')
        segno.make(url, error='m').terminal(compact=True)
    except ImportError:
        print('(Para ver el QR en la consola: pip install segno. El dashboard también lo muestra.)')
    print('\nCtrl+C para detener.\n')

    if not a.sin_navegador:
        threading.Timer(0.8, webbrowser.open, args=(url + '?qr=1',)).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print('Servidor detenido.')


if __name__ == '__main__':
    main()
