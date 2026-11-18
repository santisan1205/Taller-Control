"""Genera 4 stickers con el emblema de C.I.T.A.D.E.L. y el nombre de cada integrante.

Uso (desde la raíz del proyecto):  python stickers/generar_stickers.py
Salida: stickers/sticker_<nombre>.png (1200x1200, esquinas redondeadas transparentes)
        stickers/hoja_stickers.png (los 4 juntos, para revisar)
"""
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont

RAIZ = Path(__file__).resolve().parent
LOGO = RAIZ.parent / "dashboard" / "img" / "citadel_logo_original.jpg"
FUENTE = "C:/Windows/Fonts/GOTHICB.TTF"  # Century Gothic negrita, parecida a la del logo
NOMBRES = ["Tomy", "Santi", "Andrés", "Mari"]
LADO = 1200
FONDO = (10, 8, 14)
MORADO, ROSA = (154, 114, 196), (240, 140, 186)  # extremos del degradado del logo


def emblema():
    """Recorta el símbolo del logo original (fondo negro, se conserva el brillo)."""
    im = Image.open(LOGO).convert("RGB")
    cx, cy, r = 1385, 598, 345
    return im.crop((cx - r, cy - r, cx + r, cy + r))


def texto_degradado(nombre, ancho_max):
    """Devuelve (imagen, desplazamiento_y_respecto_a_la_línea_base)."""
    fs = 250
    while True:
        f = ImageFont.truetype(FUENTE, fs)
        x0, y0, x1, y1 = f.getbbox(nombre, anchor="ls")
        if x1 - x0 <= ancho_max or fs < 60:
            break
        fs -= 4
    mascara = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(mascara).text((-x0, -y0), nombre, font=f, fill=255, anchor="ls")
    w, h = mascara.size
    grad = Image.new("RGB", (w, h))
    gd = ImageDraw.Draw(grad)
    for x in range(w):
        t = x / max(w - 1, 1)
        gd.line([(x, 0), (x, h)], fill=tuple(round(a + (b - a) * t) for a, b in zip(MORADO, ROSA)))
    grad.putalpha(mascara)
    return grad, y0


def sticker(nombre, emb):
    base = Image.new("RGB", (LADO, LADO), FONDO)
    # borde fino con degradado
    borde = Image.new("RGB", (LADO, LADO))
    bd = ImageDraw.Draw(borde)
    for x in range(LADO):
        t = x / (LADO - 1)
        bd.line([(x, 0), (x, LADO)], fill=tuple(round(a + (b - a) * t) for a, b in zip(MORADO, ROSA)))
    m_ext = Image.new("L", (LADO, LADO), 0)
    ImageDraw.Draw(m_ext).rounded_rectangle((0, 0, LADO - 1, LADO - 1), 170, fill=255)
    m_int = Image.new("L", (LADO, LADO), 0)
    ImageDraw.Draw(m_int).rounded_rectangle((22, 22, LADO - 23, LADO - 23), 150, fill=255)
    lienzo = Image.composite(base, borde, m_int)

    e = emb.resize((640, 640), Image.LANCZOS)
    # "screen": el negro del recorte desaparece y solo queda el brillo del emblema
    x, y = (LADO - 640) // 2, 80
    zona = lienzo.crop((x, y, x + 640, y + 640))
    lienzo.paste(ImageChops.screen(zona, e), (x, y))
    t, y0 = texto_degradado(nombre, 900)
    LINEA_BASE = 1000
    lienzo.paste(t, ((LADO - t.width) // 2, LINEA_BASE + y0), t)

    salida = lienzo.convert("RGBA")
    salida.putalpha(m_ext)
    return salida


if __name__ == "__main__":
    emb = emblema()
    hoja = Image.new("RGBA", (LADO * 2 + 60, LADO * 2 + 60), (255, 255, 255, 255))
    for i, n in enumerate(NOMBRES):
        s = sticker(n, emb)
        s.save(RAIZ / f"sticker_{n.lower()}.png")
        hoja.paste(s, (20 + (i % 2) * (LADO + 20), 20 + (i // 2) * (LADO + 20)), s)
        print("listo:", n)
    hoja.resize((hoja.width // 2, hoja.height // 2), Image.LANCZOS).save(RAIZ / "hoja_stickers.png")
