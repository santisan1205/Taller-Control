"""Código QR fijo del dashboard publicado en GitHub Pages (para la presentación y el póster).

Uso (desde la raíz del proyecto):
    python dashboard/generar_qr.py
Genera dashboard/img/qr_dashboard.png (para imprimir) y dashboard/img/qr_dashboard.svg.
Si cambia la dirección pública, cambiarla aquí y en URL_PUBLICA de dashboard/index.html.
"""
from pathlib import Path

import segno

URL_PUBLICA = 'https://santisan1205.github.io/Taller-Control/dashboard/'
SALIDA = Path(__file__).resolve().parent / 'img'
OSCURO = '#1C1924'  # el morado oscuro del dashboard: contrasta como el negro

qr = segno.make(URL_PUBLICA, error='q')
qr.save(SALIDA / 'qr_dashboard.png', scale=24, border=4, dark=OSCURO, light='#FFFFFF')
qr.save(SALIDA / 'qr_dashboard.svg', scale=10, border=4, dark=OSCURO, light='#FFFFFF')
print(f'{URL_PUBLICA}\n-> {SALIDA / "qr_dashboard.png"}\n-> {SALIDA / "qr_dashboard.svg"}')
