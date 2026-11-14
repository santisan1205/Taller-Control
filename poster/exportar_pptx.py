"""Convierte poster/poster.html en un .pptx editable sobre la plantilla vertical de la Semana de la Innovación.

Textos, tarjetas, cifras y la línea de ruta quedan como formas editables de PowerPoint (en Verdana,
la fuente de la plantilla). Logo, esquemas de la metodología, gráficas y QR van como imágenes PNG
en alta resolución (las gráficas se regeneran con poster/generar_graficas.py).

Uso (desde la carpeta del proyecto, con Chrome instalado):
    pip install python-pptx Pillow
    python poster/exportar_pptx.py
Sale poster/Poster_CITADEL_70x100.pptx
"""
import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Emu, Pt

AQUI = Path(__file__).resolve().parent
HTML = AQUI / "poster.html"
PLANTILLA = AQUI / "assets" / "plantilla_vertical.pptx"
SALIDA = AQUI / "Poster_CITADEL_70x100.pptx"
CHROME = next((p for p in [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
] if p.exists()), None)

ANCHO_PX, ALTO_PX = 2646, 3780     # lienzo del HTML = 70 x 100 cm
ESCALA_IMG = 3                     # resolución de los recortes PNG (3x ~ 290 ppp a tamaño real)
FUENTE = "Verdana"
Image.MAX_IMAGE_PIXELS = None      # la captura a 3x (~90 Mpx) la genera este mismo script


def chrome(*args):
    subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={ANCHO_PX},{ALTO_PX}", "--virtual-time-budget=10000", *args],
                   check=True, capture_output=True)


def url(consulta):
    return HTML.as_uri() + "?" + consulta


def leer_geometria():
    r = subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", f"--window-size={ANCHO_PX},{ALTO_PX}",
                        "--virtual-time-budget=10000", "--dump-dom", url("ppt")],
                       check=True, capture_output=True, text=True, encoding="utf8")
    m = re.search(r'data-ppt="([^"]*)"', r.stdout)
    if not m:
        raise SystemExit("No se encontró data-ppt: ¿cargó poster.html?")
    datos = json.loads(html.unescape(m.group(1)))
    if datos["desborde"] > 1:
        raise SystemExit(f"El contenido se desborda {datos['desborde']} px en modo Verdana: ajusta poster.html")
    return datos


def color(css):
    """'rgb(1, 2, 3)' / 'rgba(1, 2, 3, a)' -> (RGBColor, alfa) o None si es transparente."""
    v = [float(x) for x in re.findall(r"[\d.]+", css or "")]
    if len(v) < 3 or (len(v) == 4 and v[3] == 0):
        return None
    return RGBColor(*(int(round(c)) for c in v[:3])), (v[3] if len(v) == 4 else 1.0)


E = 25199975 / ANCHO_PX  # EMU por px


def emu(v):
    return Emu(int(round(v * E)))


def sin_sombra(forma):
    forma.shadow.inherit = False


def rect(slide, x, y, w, h, radio, relleno=None, borde=None, ancho_borde=0, nombre=""):
    if radio <= 0.5:
        tipo = MSO_SHAPE.RECTANGLE
    elif abs(w - h) < 2 and radio >= w / 2 - 1:
        tipo = MSO_SHAPE.OVAL
    else:
        tipo = MSO_SHAPE.ROUNDED_RECTANGLE
    f = slide.shapes.add_shape(tipo, emu(x), emu(y), emu(w), emu(h))
    if tipo == MSO_SHAPE.ROUNDED_RECTANGLE:
        f.adjustments[0] = min(radio / min(w, h), 0.5)
    if relleno:
        f.fill.solid()
        f.fill.fore_color.rgb = relleno[0]
    else:
        f.fill.background()
    if borde and ancho_borde > 0:
        f.line.color.rgb = borde[0]
        f.line.width = Pt(ancho_borde * 0.75)
    else:
        f.line.fill.background()
    sin_sombra(f)
    if nombre:
        f.name = nombre
    return f


def dibujar_caja(slide, c):
    x, y, w, h = c["r"]
    if c["cls"] == "linea-ruta":
        ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, emu(x), emu(y + h / 2), emu(x + w), emu(y + h / 2))
        ln.line.color.rgb = RGBColor(0x5C, 0xB7, 0x7C)
        ln.line.width = Pt(h * 0.75)
        ln.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        ln.name = "linea ruta"
        return
    fondo = color(c["fondo"])
    bt, bl = c["borde"]
    nombre = c["cls"]
    if bl > 0 and bt == 0:
        # tarjeta con borde solo a la izquierda (KPI): forma de color debajo y la tarjeta encima, corrida
        rect(slide, x, y, w, h, c["radio"], relleno=color(c["bordeIzqColor"]), nombre=nombre + " borde")
        rect(slide, x + bl, y, w - bl, h, c["radio"], relleno=fondo, nombre=nombre)
        return
    if bt > 0:  # el borde CSS va por dentro; el de PowerPoint va centrado en el contorno
        x, y, w, h = x + bt / 2, y + bt / 2, w - bt, h - bt
    rect(slide, x, y, w, h, c["radio"], relleno=fondo, borde=color(c["bordeColor"]), ancho_borde=bt, nombre=nombre)


def dibujar_texto(slide, t):
    x, y, w, h = t["r"]
    pars = t["parrafos"]
    if not pars:
        return
    # cada párrafo ocupa una sola línea en el HTML (títulos, líneas separadas con <br>): en PowerPoint no se parten
    alto_lineas = sum(max(r["lh"] for r in runs) for runs in pars)
    una_linea = h <= alto_lineas * 1.3
    alinea = {"center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}.get(t["alinea"], PP_ALIGN.LEFT)
    # holgura para que PowerPoint no parta distinto: amplia en líneas fijas o textos centrados
    holgura = w * (0.04 if (una_linea or t["alinea"] == "center") else 0.005)
    if alinea == PP_ALIGN.CENTER:
        x -= holgura / 2
    w += holgura
    caja = slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    caja.name = "texto " + t["cls"]
    tf = caja.text_frame
    tf.word_wrap = not una_linea
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE if t["vcentro"] else MSO_ANCHOR.TOP
    for i, runs in enumerate(pars):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = alinea
        p.line_spacing = Pt(max(r["lh"] for r in runs) * 0.75)
        antes = t.get("antes") or []
        if i < len(antes) and antes[i]:
            p.space_before = Pt(antes[i] * 0.75)
        for r in runs:
            if not r["t"]:
                continue
            run = p.add_run()
            run.text = r["t"]
            fnt = run.font
            fnt.name = FUENTE
            fnt.size = Pt(round(r["px"] * 0.75 * 2) / 2)
            fnt.bold = r["b"]
            fnt.italic = r["i"]
            c = color(r["color"])
            if c:
                fnt.color.rgb = c[0]


def recortar_imagenes(datos, carpeta):
    captura = carpeta / "graficos.png"
    chrome(f"--force-device-scale-factor={ESCALA_IMG}", "--default-background-color=00000000",
           f"--screenshot={captura}", url("ppt&graficos"))
    lienzo = Image.open(captura)
    rutas = {}
    for im in datos["imagenes"]:
        x, y, w, h = (v * ESCALA_IMG for v in im["r"])
        pieza = lienzo.crop((int(x), int(y), int(round(x + w)), int(round(y + h))))
        rutas[im["id"]] = carpeta / f"{im['id']}.png"
        pieza.save(rutas[im["id"]])
    return rutas


def main():
    if CHROME is None:
        raise SystemExit("No encontré Chrome ni Edge para renderizar el HTML")
    datos = leer_geometria()
    # la línea punteada de la ruta va detrás de los círculos de las etapas
    cajas = datos["cajas"]
    linea = [c for c in cajas if c["cls"] == "linea-ruta"]
    cajas = [c for c in cajas if c["cls"] != "linea-ruta"]
    i = next((k for k, c in enumerate(cajas) if c["cls"] == "p"), len(cajas))
    cajas[i:i] = linea

    with tempfile.TemporaryDirectory() as tmp:
        rutas = recortar_imagenes(datos, Path(tmp))
        prs = Presentation(str(PLANTILLA))
        assert prs.slide_width == 25199975, "La plantilla no es la vertical de 70 x 100 cm"
        slide = prs.slides[0]
        # quita los marcadores de la plantilla; el encabezado con los logos es el fondo de la diapositiva
        for forma in list(slide.shapes):
            forma._element.getparent().remove(forma._element)
        for rid, rel in list(slide.part.rels.items()):
            if rel.reltype.endswith("/image") and rid not in slide._element.xml:
                slide.part.drop_rel(rid)

        for c in cajas:
            dibujar_caja(slide, c)
        for im in datos["imagenes"]:
            x, y, w, h = im["r"]
            pic = slide.shapes.add_picture(str(rutas[im["id"]]), emu(x), emu(y), emu(w), emu(h))
            pic.name = "imagen " + im["cls"]
            pic._element.nvPicPr.cNvPr.set("descr", im["alt"] or im["cls"])
        for t in datos["textos"]:
            dibujar_texto(slide, t)
        prs.save(str(SALIDA))
    print("Listo:", SALIDA)


if __name__ == "__main__":
    main()
