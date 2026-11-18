"""Presentación final (8–10 min) en dos estilos y su guion, a partir de datos_dashboard/v2.

Uso (desde la raíz del proyecto):
    python presentacion/generar_final.py
Genera en presentacion/:
    presentacion_final_amarilla.html  estilo Uniandes (amarillo)
    presentacion_final_citadel.html   colores del equipo (los del dashboard y el logo)
    guion_final.html                  guion por expositor
Los PDF se exportan con Chrome headless (--print-to-pdf).

Misma estructura que la presentación del 28 de octubre, con los resultados finales (entorno v2,
dos versiones del algoritmo). Expositores: Tomy (1–4), Andrés (5–8), Santi (9–11), Mariana (12–15).
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos_dashboard" / "v2"
SALIDA = Path(__file__).resolve().parent

# Imágenes que usa la presentación (se copian junto a ella)
URL_DASHBOARD = "https://santisan1205.github.io/Taller-Control/dashboard/"
for nombre in ["qr_dashboard.png", "citadel_emblema.png", "citadel_logo.png"]:
    shutil.copy(RAIZ / "dashboard" / "img" / nombre, SALIDA / nombre)

# ================================================================== datos
d = pd.read_csv(DATOS / "evaluacion_detalle.csv")
d["config"] = d.politica.str.replace(r"_seed\d", "", regex=True)
e = pd.read_csv(DATOS / "entrenamiento_episodios.csv")


def media(ventana, escala, config, metrica):
    s = d[(d.ventana == ventana) & (d.escala == escala) & (d.config == config)]
    return s[metrica].mean()


def pct(a, b):
    return (a / b - 1) * 100


def p_pareado(ventana, escala, config, metrica):
    s = d[(d.ventana == ventana) & (d.escala == escala)]
    f = s[s.config == "fijo"].groupby("semilla")[metrica].mean().sort_index()
    x = s[s.config == config].groupby("semilla")[metrica].mean().sort_index()
    return stats.ttest_rel(x, f).pvalue


def n(x, dec=1):
    """Número con coma decimal y punto de miles."""
    s = f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s.replace("-", "−")


def pc(x, dec=0):
    return ("+" if x > 0 else "") + n(x, dec) + " %"


METODOS = [("fijo", "Tiempo fijo"), ("actuado", "Actuado"), ("max_pressure", "Max-pressure"),
           ("v2_base", "IA · alg. 1"), ("v2a2_base", "IA · alg. 2")]
M = {c: {m: media("pico", 1.0, c, m) for m in d.columns if d[m].dtype != object} for c, _ in METODOS}
fijo, a1, a2, act, mp = M["fijo"], M["v2_base"], M["v2a2_base"], M["actuado"], M["max_pressure"]

# Algoritmo 1 frente a 2: 18 políticas de cada uno, pareadas, promedio de las 3 escalas
r = d[d.modo == "rnd_mappo"].copy()
r["alg"] = np.where(r.politica.str.startswith("v2a2"), 2, 1)
r["pol"] = r.politica.str.replace("v2a2_", "").str.replace("v2_", "")
r["cfg"] = r.config.str.replace("v2a2_", "").str.replace("v2_", "")
ALG = {}
for v in ["pico", "valle"]:
    p = r[r.ventana == v].groupby(["alg", "pol"])[["espera_media", "peatones_espera_media"]].mean()
    x1, x2 = p.loc[1], p.loc[2].loc[p.loc[1].index]
    for m in ["espera_media", "peatones_espera_media"]:
        ALG[(v, m)] = (x1[m].mean(), x2[m].mean(), stats.ttest_rel(x1[m], x2[m]).pvalue)
    sd = (r[r.ventana == v].groupby(["alg", "cfg", "pol"]).espera_media.mean()
          .groupby(["alg", "cfg"]).std().groupby("alg").mean())
    ALG[(v, "sd")] = (sd[1], sd[2])
RND = {a: r[(r.alg == a) & (r.ventana == "pico") & r.cfg.isin(["base", "sin_rnd"])]
       .groupby("cfg").espera_media.mean() for a in (1, 2)}
TRANCADOS = {a: int(((r.alg == a) & (r.espera_media > 20)).sum()) for a in (1, 2)}

# Peso de la curiosidad: λ·R_int frente a |R_ext| por episodio, primeros 20 episodios
e["alg"] = np.where(e.corrida.str.startswith("v2a2"), 2, 1)
ec = e[~e.corrida.str.contains("sin_rnd") & (e.episodio <= 20)]
PESO_RND = {a: ((ec[ec.alg == a].lambda_rnd * ec[ec.alg == a].recompensa_int).mean()
                / ec[ec.alg == a].recompensa_ext.abs().mean() * 100) for a in (1, 2)}
VECES_RND = PESO_RND[2] / PESO_RND[1]

# Curva de aprendizaje: episodios de hora pico de la config. base, bloques de 10 episodios
CURVA = {}
for a, pref in [(1, "v2_base_"), (2, "v2a2_base_")]:
    x = e[e.corrida.str.startswith(pref) & (e.ventana == "pico")].copy()
    x["bloque"] = (x.episodio - 1) // 10
    CURVA[a] = x.groupby("bloque").espera_media.mean()
    CURVA[(a, "ini")] = x[x.episodio <= 10].espera_media.mean()
    CURVA[(a, "fin")] = x[x.episodio > 140].espera_media.mean()

ROB = {esc: {c: media("pico", esc, c, "espera_media") for c in ["fijo", "v2_base", "v2a2_base"]}
       for esc in (0.8, 1.0, 1.2)}
VALLE = {c: pct(media("valle", 1.0, c, "espera_media"), media("valle", 1.0, "fijo", "espera_media"))
         for c in ["actuado", "max_pressure", "v2_base", "v2a2_base"]}
NUBE = (r[(r.ventana == "pico") & (r.escala == 1.0)]
        .groupby(["alg", "politica"])[["peatones_espera_media", "espera_media"]].mean())
VIAJES_VALLE = media("valle", 1.0, "fijo", "viajes_completados")
VIAJES_PICO = media("pico", 1.0, "v2_base", "viajes_completados")

pv = max(p_pareado("pico", 1.0, c, m) for c in ("v2_base", "v2a2_base")
         for m in ("espera_media", "peatones_espera_media", "viajes_completados", "co2_total_kg"))
assert pv < 0.001, pv

# ================================================================== guion (texto hablado)
# (expositor, duración en s, texto). Va en las notas de las diapositivas (tecla N) y en el guion.
GUION = {
"cover": ("Tomy", 20,
 "Buenos días, profesor. Somos Tomás, Andrés, Mariana y Santiago, y este es C.I.T.A.D.E.L.: un sistema de "
 "control semafórico con aprendizaje por refuerzo multiagente. Cada uno de los cuatro semáforos de un sector "
 "de Bogotá es un agente que aprende, en el simulador SUMO, cuándo cambiar de fase, y lo comparamos con los "
 "controladores que se usan hoy."),
"agenda": ("Tomy", 15,
 "Primero les cuento el problema y cómo lo formulamos. Andrés explica el algoritmo y cómo lo entrenamos. "
 "Santi presenta los resultados, y Mariana cierra con el análisis, el dashboard y las conclusiones."),
"escenario": ("Tomy", 55,
 "El escenario es un sector real de Bogotá: tomamos la red vial de OpenStreetMap y la llevamos a SUMO, un "
 "simulador que mueve cada vehículo y cada peatón por separado. Hay cuatro intersecciones semaforizadas, C1 a "
 "C4, y en C1 y C4 hay pasos peatonales. Simulamos dos horas del día: la hora valle y la hora pico, en la que "
 "pasan unos dos mil vehículos por hora. Cada semáforo respeta 15 segundos de verde mínimo, 3 de amarillo y 60 "
 "de verde máximo. Comparamos contra tres controladores: el tiempo fijo que trae la red, el semáforo actuado "
 "de SUMO, que alarga el verde cuando detecta carros, y max-pressure, un método clásico que da verde a la "
 "dirección con más cola acumulada."),
"formulacion": ("Tomy", 45,
 "Formalmente es un juego estocástico parcialmente observable: cuatro agentes, y ninguno ve la red completa. "
 "Cada uno observa su cruce: la fase activa, si ya cumplió el verde mínimo, y la densidad y la cola de cada "
 "carril; en C1 y C4, además, cuántos peatones esperan y cuánto lleva el que más espera. Cada 5 segundos elige "
 "la siguiente fase verde. La recompensa premia que bajen la cola, la espera de los carros y la de los "
 "peatones, que pesa el doble, y castiga cada cambio de fase. Un episodio es una hora: 720 decisiones. Andrés "
 "les cuenta cómo aprenden."),
"arquitectura": ("Andrés", 45,
 "Usamos MAPPO, la versión multiagente de PPO, con entrenamiento centralizado y ejecución descentralizada. "
 "Cada cruce tiene su actor: una red que recibe la observación local y da una probabilidad para cada fase; son "
 "cuatro redes distintas porque cada cruce tiene distintos carriles y fases. El crítico ve las cuatro "
 "observaciones juntas y solo se usa para entrenar: le dice a cada actor qué tan buena fue su decisión. En "
 "operación, cada semáforo decide solo con lo que ve, sin comunicarse. La actualización es la de PPO, con "
 "recorte de 0,2, bono de entropía y ventaja GAE."),
"recompensa": ("Andrés", 35,
 "A la recompensa del tráfico le sumamos curiosidad, con Random Network Distillation. Hay una red aleatoria "
 "fija y otra que aprende a imitarla: en los estados que el agente ya conoce la imita bien; en los nuevos "
 "falla, y ese error se suma como premio por explorar. Su peso λ baja de 1 a casi 0 durante el entrenamiento: "
 "al inicio explora y al final solo optimiza el tráfico."),
"versiones": ("Andrés", 40,
 "Implementamos dos versiones del algoritmo. En la primera, el crítico tenía una sola salida y se actualizaba "
 "por turnos con la recompensa de cada agente; como las recompensas de C2 y C3 son mucho más grandes que las "
 "de C4, un solo valor no puede ajustarlas todas. Además, la normalización de RND dejaba la curiosidad en "
 f"cerca del {n(PESO_RND[1], 1)} % de la recompensa, o sea, sin efecto. En la versión 2 el crítico da un valor "
 f"por agente, con normalización de valores, y la curiosidad pesa unas {n(VECES_RND, 0)} veces más."),
"protocolo": ("Andrés", 35,
 "Con cada versión entrenamos seis configuraciones: la base, una sin curiosidad y cuatro que cambian la "
 "exploración o los pesos de la recompensa; cada una con tres semillas y 150 episodios. Son 36 corridas. "
 "Luego evaluamos cada política en valle y en pico, con 80, 100 y 120 % de la demanda y cinco semillas de "
 "SUMO, siempre pareadas: todos los métodos ven exactamente el mismo tráfico. En total, 1.170 episodios de "
 "evaluación. Santi les muestra los resultados."),
"curva": ("Santi", 30,
 "Esta es la curva de aprendizaje en hora pico de la configuración base, con las dos versiones. Al inicio la "
 f"política es casi aleatoria y los carros esperan unos {n(CURVA[(1, 'ini')], 0)} segundos, tres veces más que "
 "con el tiempo fijo. Hacia el episodio 30 el agente ya lo supera, y al final la espera queda en "
 f"{n(CURVA[(1, 'fin')], 1)} y {n(CURVA[(2, 'fin')], 1)} segundos. Las dos versiones aprenden al mismo ritmo."),
"resultados": ("Santi", 65,
 "Estos son los resultados en hora pico con la demanda nominal. Frente al tiempo fijo, la espera de los carros "
 f"baja {n(-pct(a1['espera_media'], fijo['espera_media']), 0)} % con la versión 1 y "
 f"{n(-pct(a2['espera_media'], fijo['espera_media']), 0)} % con la versión 2, y la de los peatones también baja, "
 f"de {n(fijo['peatones_espera_media'])} a unos {n(a1['peatones_espera_media'], 0)} segundos. Se completan "
 f"{n(pct(a1['viajes_completados'], fijo['viajes_completados']), 0)} % más viajes y el CO₂ baja "
 f"{n(-pct(a1['co2_total_kg'], fijo['co2_total_kg']), 0)} %. Todo con p menor que 0,001. Frente a los "
 "controladores clásicos hay que ser honestos: max-pressure deja a los carros esperando menos, "
 f"{n(mp['espera_media'])} segundos, y tanto max-pressure como el actuado tienen menor tiempo perdido por "
 f"viaje. Pero max-pressure no ve a los peatones: los hace esperar {n(mp['peatones_espera_media'], 0)} "
 "segundos. Eso se ve mejor en la siguiente gráfica."),
"compromiso": ("Santi", 45,
 "Aquí cada punto es un método: a la derecha más espera de peatones, arriba más espera de carros; lo ideal es "
 "abajo a la izquierda. Los puntos pequeños son las 36 políticas entrenadas. El agente es el único que baja las "
 "dos esperas a la vez frente al tiempo fijo: es un compromiso entre carros y peatones. A la derecha, la "
 "robustez: con 20 % menos o 20 % más tráfico, sin reentrenar, la espera sigue bajando más de 60 % en hora "
 "pico. En valle la mejora es menor, alrededor de 45 %, parecida al actuado. Mariana les cuenta qué aprendimos "
 "de las dos versiones."),
"comparacion": ("Mariana", 55,
 "Comparamos las 18 políticas de cada versión. En la espera de los carros no hay diferencia: "
 f"{n(ALG[('pico', 'espera_media')][0], 2)} contra {n(ALG[('pico', 'espera_media')][1], 2)} segundos en pico, "
 f"con p de {n(ALG[('pico', 'espera_media')][2], 2)}. Los peatones esperan alrededor de un segundo más con la "
 "versión 2, y eso sí es significativo. Lo que mejora es la estabilidad: la variación entre semillas baja a la "
 "tercera parte en hora pico y ningún episodio se tranca; con la versión 1, una política cayó en gridlock con "
 "20 % más demanda. Sobre la curiosidad: quitar RND da el mismo resultado en las dos versiones. Con la versión "
 "1 era esperable, porque casi no pesaba; con la 2 sí pesa y aun así no aporta."),
"dashboard": ("Mariana", 45,
 "Todo esto se puede explorar en el dashboard, conectado a los datos reales: las 36 corridas, las tres líneas "
 "base y los 1.170 episodios. Se elige la versión de la IA, la hora, el tráfico y contra qué comparar, y se "
 "pueden ver dos métodos lado a lado animados con la simulación de SUMO. Está pensado para alguien que no "
 "conoce el proyecto, y funciona en el celular: pueden escanear este código y abrirlo ahora mismo."),
"conclusiones": ("Mariana", 50,
 "Para cerrar. Uno: en hora pico el agente reduce la espera de los carros entre "
 f"{n(-pct(a2['espera_media'], fijo['espera_media']), 0)} y {n(-pct(a1['espera_media'], fijo['espera_media']), 0)} % "
 "frente al tiempo fijo y completa 12 % más viajes. Dos: es el único método que mejora a la vez a carros y "
 "peatones; max-pressure gana en carros pero castiga a los peatones. Tres: la mejora se mantiene con 20 % más o "
 "menos demanda sin reentrenar. Cuatro: la versión 2 es más estable, pero no mejora el desempeño, y la "
 "curiosidad no aporta en este problema. Como trabajo futuro: un peso inicial de curiosidad menor o más "
 "episodios, mejorar el tiempo perdido por viaje y validar con conteos reales."),
"cierre": ("Mariana", 10, "Muchas gracias. Quedamos atentos a sus preguntas."),
}
ORDEN = list(GUION)
TOTAL = len(ORDEN)
ACUM, t = {}, 0
for sid in ORDEN:
    t += GUION[sid][1]
    ACUM[sid] = t


def mmss(s):
    return f"{s // 60}:{s % 60:02d}"


def nota(sid):
    quien, dur, texto = GUION[sid]
    return f"[{quien} · {mmss(dur)} · acumulado {mmss(ACUM[sid])}]\n{texto}"


# ================================================================== temas de color
AMARILLA = dict(
    nombre="amarilla", fuente="'Plus Jakarta Sans', Arial, sans-serif",
    fuente_href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400..800&display=swap",
    fondo="#FDFDFB", tinta="#0C1829", texto="#334155", suave="#475569", kicker="#B45309",
    tarjeta="#F8FAFC", borde="#E2E8F0", acento1="#FFD100", acento2="#B45309",
    th="#0C1829", th_texto="#FDFDFB", fila="#FFFFFF", fila_alt="#F8FAFC", fila_hl="#FEF3C7", linea="#E2DFD6",
    oscura="#0C1829", oscura_borde="none", oscura_titulo="#FFD100", oscura_texto="#E2E8F0",
    cifra1="#0C1829", cifra2="#B45309",
    alg1="#1D4ED8", alg2="#7C3AED", ref="#B45309", base="#0C1829",
    rejilla="#E2E8F0", eje="#64748B", grafico="#FFFFFF", pie="#CBD5E1",
)
CITADEL = dict(
    nombre="citadel", fuente="'Inter', Arial, sans-serif",
    fuente_href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
    fondo="#141118", tinta="#FFFFFF", texto="#D9D5E2", suave="#A6A2B4", kicker="#E975A8",
    tarjeta="#1C1924", borde="rgba(255,255,255,0.07)", acento1="#E975A8", acento2="#9D50BB",
    th="#2E2140", th_texto="#FFFFFF", fila="#18151F", fila_alt="#1C1924", fila_hl="rgba(233,117,168,0.14)",
    linea="rgba(255,255,255,0.06)",
    oscura="linear-gradient(135deg,#2A1B38 0%,#1C1924 100%)", oscura_borde="1px solid rgba(233,117,168,0.35)",
    oscura_titulo="#E975A8", oscura_texto="#D9D5E2",
    cifra1="#FFFFFF", cifra2="#E975A8",
    alg1="#E975A8", alg2="#A97BF0", ref="#D9D5E2", base="#FFFFFF",
    rejilla="rgba(255,255,255,0.08)", eje="#6B6778", grafico="#1C1924", pie="#1C1924",
)
MONO = "'JetBrains Mono', 'Courier New', monospace"
DEGRADADO = "linear-gradient(90deg,#9D50BB,#E975A8)"
FONDO_PORTADA = "linear-gradient(135deg,#2A1B38 0%,#0F0C14 100%)"
TEXTO_DEGRADADO = f"background:{DEGRADADO};-webkit-background-clip:text;background-clip:text;color:transparent"


def construir(T):
    """HTML de la presentación completa con el tema T."""
    oscuro = T["nombre"] == "citadel"
    color_pie = T["suave"] if oscuro else T["texto"]

    # ---------------------------------------------------------- piezas
    def marco(num, kicker, titulo, sub=None):
        if oscuro:
            barra = f"""<div style="position:absolute;left:0;top:0;width:1920px;height:112px;background:#17141E"></div>
<div style="position:absolute;left:0;top:112px;width:1920px;height:4px;background:{DEGRADADO}"></div>
<img src="citadel_emblema.png" alt="" style="position:absolute;left:128px;top:24px;width:64px;height:64px">
<p style="position:absolute;left:210px;top:34px;font-size:30px;font-weight:800;letter-spacing:2px;{TEXTO_DEGRADADO}">C.I.T.A.D.E.L.</p>
<img src="uniandes.png" alt="Universidad de los Andes" style="position:absolute;right:128px;top:34px;width:200px;height:44px;object-fit:contain;filter:invert(1)">"""
        else:
            barra = f"""<div style="position:absolute;left:0;top:0;width:1920px;height:112px;background:{T['acento1']}"></div>
<div style="position:absolute;left:128px;top:24px;width:288px;height:64px;background:#FFFFFF;border-radius:10px"></div>
<img src="uniandes.png" alt="Universidad de los Andes" style="position:absolute;left:146px;top:31px;width:252px;height:50px;object-fit:contain">
<p style="position:absolute;right:128px;top:38px;width:900px;text-align:right;font-family:{MONO};font-size:24px;font-weight:700;color:{T['tinta']}">C.I.T.A.D.E.L. · RND-MAPPO</p>"""
        h = barra + f"""
<div style="position:absolute;left:0;top:1040px;width:1920px;height:40px;background:{T['pie']}"></div>
<p style="position:absolute;left:128px;top:1044px;width:1200px;font-size:24px;line-height:1.3;color:{color_pie}">IELE · Taller de Control · Uniandes</p>
<p style="position:absolute;right:128px;top:1044px;width:300px;text-align:right;font-size:24px;line-height:1.3;color:{color_pie}">{num} / {TOTAL}</p>
<div style="display:flex;flex-direction:column;gap:12px">
<div style="display:flex;align-items:center;gap:16px"><div style="width:40px;height:5px;background:{DEGRADADO if oscuro else T['kicker']}"></div><p style="font-size:24px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:{T['kicker']}">{kicker}</p></div>
<h2 style="font-size:56px;font-weight:800;line-height:1.15;letter-spacing:-0.5px;color:{T['tinta']}">{titulo}</h2>"""
        if sub:
            h += f'\n<p style="font-size:28px;line-height:1.4;color:{T["suave"]}">{sub}</p>'
        return h + "\n</div>"

    def tarjeta(titulo, texto, acento=1):
        borde = T["acento1"] if acento == 1 else T["acento2"]
        return f"""<div style="display:flex;flex-direction:column;gap:10px;background:{T['tarjeta']};border:1px solid {T['borde']};border-left:6px solid {borde};border-radius:12px;padding:24px 30px">
<h3 style="font-size:31px;font-weight:700;line-height:1.2;color:{T['tinta']}">{titulo}</h3>
<p style="font-size:25px;line-height:1.4;color:{T['texto']}">{texto}</p></div>"""

    def caja_oscura(titulo, mono, texto):
        return f"""<div style="display:flex;flex-direction:column;gap:12px;background:{T['oscura']};border:{T['oscura_borde']};border-radius:12px;padding:30px 38px">
<p style="font-size:40px;font-weight:800;line-height:1.1;color:{T['oscura_titulo']}">{titulo}</p>
<p style="font-family:{MONO};font-size:25px;line-height:1.4;color:{T['oscura_texto']}">{mono}</p>
<p style="font-size:26px;line-height:1.4;color:{T['oscura_texto']}">{texto}</p></div>"""

    def cifra(valor, texto, acento=1):
        color = T["cifra1"] if acento == 1 else T["cifra2"]
        linea = (DEGRADADO if acento == 2 else "rgba(255,255,255,0.35)") if oscuro else color
        return f"""<div style="display:flex;flex-direction:column;gap:4px;padding-top:14px;position:relative">
<div style="position:absolute;left:0;right:0;top:0;height:4px;background:{linea}"></div>
<p style="font-size:52px;font-weight:800;line-height:1.1;color:{color}">{valor}</p>
<p style="font-size:24px;line-height:1.3;color:{T['suave']}">{texto}</p></div>"""

    def tabla(cab, filas, anchos, resaltar=(), tam=26, der_desde=1, ancho="1664px"):
        def celda(txt, j, th=False):
            tag = "th" if th else "td"
            al = "text-align:right;" if j >= der_desde else ""
            col = f"color:{T['th_texto']};" if th else ""
            w = f"width:{anchos[j]};" if th else ""
            return f'<{tag} style="{w}{al}{col}">{txt}</{tag}>'
        h = f'<table style="font-size:{tam}px;color:{T["texto"]};width:{ancho}">\n<tr style="background:{T["th"]}">'
        h += "".join(celda(c, j, True) for j, c in enumerate(cab)) + "</tr>\n"
        for i, f in enumerate(filas):
            bg = T["fila_hl"] if i in resaltar else (T["fila_alt"] if i % 2 else T["fila"])
            h += f'<tr style="background:{bg}">' + "".join(celda(c, j) for j, c in enumerate(f)) + "</tr>\n"
        return h + "</table>"

    def seccion(sid, estilo, cuerpo, fondo=None):
        return (f'<section id="{sid}" style="background:{fondo or T["fondo"]};color:{T["tinta"]};font-family:{T["fuente"]};'
                f'{estilo}">\n{cuerpo}\n<aside class="notas">{nota(sid)}</aside>\n</section>\n')

    NORMAL = "padding:168px 128px 88px;display:flex;flex-direction:column;gap:36px"
    caja_grafico = f"background:{T['grafico']};border:1px solid {T['borde']};border-radius:12px"

    # ---------------------------------------------------------- gráficas
    def svg_curva():
        W, H, x0, x1, y0, y1, ymax = 1040, 500, 70, 1020, 46, 420, 25
        px = lambda b: x0 + (b + 0.5) / 15 * (x1 - x0)
        py = lambda v: y1 - v / ymax * (y1 - y0)
        s = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             'aria-label="Espera media en hora pico por bloque de 10 episodios: baja de unos 20 s a cerca de 2,5 s en las dos versiones">']
        for v in (0, 5, 10, 15, 20, 25):
            c = T["eje"] if v == 0 else T["rejilla"]
            s.append(f'<line x1="{x0}" y1="{py(v):.0f}" x2="{x1}" y2="{py(v):.0f}" stroke="{c}" stroke-width="2"/>')
            s.append(f'<text x="{x0 - 14}" y="{py(v) + 8:.0f}" font-size="24" text-anchor="end" fill="{T["suave"]}">{v}</text>')
        yf = py(fijo["espera_media"])
        s.append(f'<line x1="{x0}" y1="{yf:.0f}" x2="{x1}" y2="{yf:.0f}" stroke="{T["ref"]}" stroke-width="4" stroke-dasharray="14 10"/>')
        s.append(f'<text x="{x1}" y="{yf - 14:.0f}" font-size="24" font-weight="700" text-anchor="end" fill="{T["ref"]}">Tiempo fijo, pico nominal: {n(fijo["espera_media"])} s</text>')
        for a, col in [(1, T["alg1"]), (2, T["alg2"])]:
            pts = " ".join(f"{px(b):.0f},{py(v):.0f}" for b, v in CURVA[a].items())
            s.append(f'<polyline fill="none" stroke="{col}" stroke-width="5" stroke-linejoin="round" points="{pts}"/>')
            s += [f'<circle cx="{px(b):.0f}" cy="{py(v):.0f}" r="6" fill="{col}"/>' for b, v in CURVA[a].items()]
        for ep in (1, 50, 100, 150):
            x = x0 + (ep - 1) / 149 * (x1 - x0)
            s.append(f'<text x="{x:.0f}" y="{y1 + 34}" font-size="24" text-anchor="middle" fill="{T["suave"]}">{ep}</text>')
        s.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 64}" font-size="24" text-anchor="middle" fill="{T["suave"]}">Episodio de entrenamiento</text>')
        s.append(f'<text x="{x0 - 50}" y="20" font-size="24" fill="{T["suave"]}">Espera media de los carros (s)</text>')
        return "\n".join(s + ["</svg>"])

    def svg_compromiso():
        W, H, x0, x1, y0, y1, xmax, ymax = 900, 600, 70, 870, 46, 510, 40, 9
        px = lambda v: x0 + v / xmax * (x1 - x0)
        py = lambda v: y1 - v / ymax * (y1 - y0)
        s = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             'aria-label="Espera de carros frente a espera de peatones en hora pico: el agente queda abajo a la izquierda; max-pressure abajo a la derecha">']
        for v in (0, 3, 6, 9):
            c = T["eje"] if v == 0 else T["rejilla"]
            s.append(f'<line x1="{x0}" y1="{py(v):.0f}" x2="{x1}" y2="{py(v):.0f}" stroke="{c}" stroke-width="2"/>')
            s.append(f'<text x="{x0 - 14}" y="{py(v) + 8:.0f}" font-size="24" text-anchor="end" fill="{T["suave"]}">{v}</text>')
        for v in (0, 10, 20, 30, 40):
            s.append(f'<text x="{px(v):.0f}" y="{y1 + 34}" font-size="24" text-anchor="middle" fill="{T["suave"]}">{v}</text>')
        s.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 66}" font-size="24" text-anchor="middle" fill="{T["suave"]}">Espera media de los peatones (s)</text>')
        s.append(f'<text x="{x0 - 50}" y="20" font-size="24" fill="{T["suave"]}">Espera media de los carros (s)</text>')
        for (a, _), row in NUBE.iterrows():
            col = T["alg1"] if a == 1 else T["alg2"]
            s.append(f'<circle cx="{px(row.peatones_espera_media):.0f}" cy="{py(row.espera_media):.0f}" r="7" fill="{col}" fill-opacity="0.3"/>')
        etiquetas = {"fijo": ("Tiempo fijo", 16, 8, "start"), "actuado": ("Actuado", 16, 8, "start"),
                     "max_pressure": ("Max-pressure", 0, -22, "middle"),
                     "v2_base": ("IA alg. 1", -22, 44, "end"), "v2a2_base": ("IA alg. 2", -22, -40, "end")}
        for c, (txt, dx, dy, anc) in etiquetas.items():
            x, y = px(M[c]["peatones_espera_media"]), py(M[c]["espera_media"])
            col = T["alg1"] if c == "v2_base" else T["alg2"] if c == "v2a2_base" else T["base"]
            if c.startswith("v2"):
                s.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="13" fill="{col}" stroke="{T["grafico"]}" stroke-width="3"/>')
            else:
                s.append(f'<rect x="{x - 11:.0f}" y="{y - 11:.0f}" width="22" height="22" fill="{col}" transform="rotate(45 {x:.0f} {y:.0f})"/>')
            s.append(f'<text x="{x + dx:.0f}" y="{y + dy:.0f}" font-size="25" font-weight="700" text-anchor="{anc}" fill="{col}">{txt}</text>')
        return "\n".join(s + ["</svg>"])

    def leyenda(items):
        return '<div style="display:flex;gap:36px">' + "".join(
            f'<p style="font-size:24px;color:{T["texto"]}"><span style="color:{c}"><b>{m}</b></span> {t}</p>'
            for c, m, t in items) + "</div>"

    S = []

    # 1 · portada
    if oscuro:
        S.append(seccion("cover", "padding:250px 128px 100px;display:flex;flex-direction:column;gap:22px", f"""
<img src="uniandes.png" alt="Universidad de los Andes" style="position:absolute;left:128px;top:70px;width:270px;height:74px;object-fit:contain;filter:invert(1)">
<p style="position:absolute;right:128px;top:92px;font-family:{MONO};font-size:24px;font-weight:700;color:{T['suave']}">SUMO · sumo-rl · PyTorch</p>
<img src="citadel_logo.png" alt="Logo de C.I.T.A.D.E.L." style="position:absolute;right:110px;top:250px;width:680px;filter:drop-shadow(0 0 40px rgba(233,117,168,0.25))">
<div style="width:980px;display:flex;flex-direction:column;gap:22px">
<p style="font-family:{MONO};font-size:24px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{T['kicker']}">Ingeniería Eléctrica y Electrónica · Taller de Control</p>
<h1 style="font-size:64px;font-weight:800;line-height:1.1;letter-spacing:-1px;color:#FFFFFF">Control semafórico multiagente con <span style="{TEXTO_DEGRADADO}">RND-MAPPO</span></h1>
<p style="font-size:32px;line-height:1.35;color:{T['texto']}">Aprendizaje por refuerzo para cuatro intersecciones de Bogotá simuladas en SUMO, con peatones y hora pico</p>
<div style="height:4px;width:980px;background:{DEGRADADO};margin:14px 0;border-radius:2px"></div>
<p style="font-size:28px;line-height:1.4;color:{T['texto']}"><b style="color:#FFFFFF">Grupo:</b> Tomás Castro · Andrés Felipe Barbosa · Mariana Vanoy · Santiago Sánchez</p>
<p style="font-size:28px;line-height:1.4;color:{T['texto']}"><b style="color:#FFFFFF">Profesor:</b> Jorge Alfredo López Jiménez · noviembre de 2026</p>
</div>""", FONDO_PORTADA))
    else:
        S.append(seccion("cover", "padding:236px 128px 100px;display:flex;flex-direction:column;gap:22px", f"""
<div style="position:absolute;left:0;top:0;width:1920px;height:150px;background:{T['acento1']}"></div>
<div style="position:absolute;left:128px;top:36px;width:340px;height:78px;background:#FFFFFF;border-radius:10px"></div>
<img src="uniandes.png" alt="Universidad de los Andes" style="position:absolute;left:150px;top:44px;width:296px;height:62px;object-fit:contain">
<p style="position:absolute;right:128px;top:56px;width:900px;text-align:right;font-family:{MONO};font-size:24px;font-weight:700;color:{T['tinta']}">SUMO · sumo-rl · PyTorch</p>
<div style="position:absolute;left:0;top:1040px;width:1920px;height:40px;background:{T['pie']}"></div>
<div style="position:absolute;right:128px;top:250px;width:560px;height:560px;background:#000;border-radius:24px;overflow:hidden;display:flex;align-items:center;justify-content:center">
<img src="citadel_logo_oscuro.jpg" alt="Logo de C.I.T.A.D.E.L." style="width:520px"></div>
<div style="width:1040px;display:flex;flex-direction:column;gap:22px">
<p style="font-family:{MONO};font-size:24px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{T['kicker']}">Ingeniería Eléctrica y Electrónica · Taller de Control</p>
<h1 style="font-size:64px;font-weight:800;line-height:1.1;letter-spacing:-1px;color:{T['tinta']}">Control semafórico multiagente con RND-MAPPO</h1>
<p style="font-size:32px;line-height:1.35;color:{T['texto']}">Aprendizaje por refuerzo para cuatro intersecciones de Bogotá simuladas en SUMO, con peatones y hora pico</p>
<hr style="border-top:2px solid {T['borde']};width:1040px;margin:12px 0">
<p style="font-size:28px;line-height:1.4;color:{T['texto']}"><b>Grupo:</b> Tomás Castro · Andrés Felipe Barbosa · Mariana Vanoy · Santiago Sánchez</p>
<p style="font-size:28px;line-height:1.4;color:{T['texto']}"><b>Profesor:</b> Jorge Alfredo López Jiménez · noviembre de 2026</p>
</div>"""))

    # 2 · agenda
    def item_agenda(num, titulo, texto):
        return f"""<div style="display:flex;flex-direction:column;gap:12px;background:{T['tarjeta']};border:1px solid {T['borde']};border-radius:12px;padding:32px 40px">
<h3 style="font-size:36px;font-weight:700;color:{T['tinta']}"><span style="color:{T['kicker']}">{num}</span> {titulo}</h3>
<p style="font-size:28px;line-height:1.4;color:{T['texto']}">{texto}</p></div>"""
    S.append(seccion("agenda", NORMAL, marco(2, "Estructura", "Agenda de la presentación",
        "Formulación, método, resultados y análisis") + f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:32px">
{item_agenda("01", "El problema", "Escenario de cuatro intersecciones con peatones; qué observa, qué decide y qué gana cada agente.")}
{item_agenda("02", "El método", "RND-MAPPO, sus dos versiones y el protocolo de entrenamiento y evaluación.")}
{item_agenda("03", "Resultados", "Aprendizaje, comparación con tres controladores, carros frente a peatones y demanda variable.")}
{item_agenda("04", "Análisis y cierre", "Versión 1 frente a 2, papel de la curiosidad, dashboard y conclusiones.")}
</div>"""))

    # 3 · escenario
    filas = [["Simulador", "SUMO 1.27.1 · sumo-rl 1.4.5"],
             ["Peatones", "Pasos peatonales en C1 y C4"],
             ["Ventanas", "Valle (0–3600 s) y hora pico (4400–8000 s)"],
             ["Demanda", "Rutas aleatorias + flujos; ×0,7 a ×1,3 al entrenar"],
             ["Verde mínimo / máximo", "15 s / 60 s"],
             ["Amarillo · teletransporte", "3 s · desactivado"],
             ["Líneas base", "Tiempo fijo, actuado (SUMO), max-pressure"]]
    S.append(seccion("escenario", NORMAL, marco(3, "El problema", "Descripción del escenario",
        "Red vial de un sector de Bogotá importada de OpenStreetMap y simulada en SUMO") + f"""
<div style="display:flex;gap:48px;align-items:flex-start">
<div style="width:620px;display:flex;flex-direction:column;gap:30px">
{cifra("4 cruces", "intersecciones semaforizadas, C1 a C4; peatones en C1 y C4")}
{cifra("2 horas", "valle y hora pico, una hora simulada cada una (720 decisiones de 5 s)", 2)}
{cifra(f"≈ {n(round(VIAJES_PICO, -2), 0)} viajes/h", f"en hora pico (≈ {n(round(VIAJES_VALLE, -1), 0)} en valle)")}
</div>
<div style="flex:1">{tabla(["Parámetro", "Valor"], filas, ["40%", "60%"], der_desde=9, ancho="100%")}</div>
</div>"""))

    # 4 · formulación
    filas = [["Agente i", "Controlador de la intersección Ci, i = 1…4", "Un actor por intersección"],
             ["Observación o<sub>i</sub>", "Fase activa (one-hot), verde mínimo cumplido, densidad y cola por carril. En C1 y C4: peatones esperando y espera máxima", "Observación de sumo-rl + peatones (TraCI)"],
             ["Acción a<sub>i</sub>", "Siguiente fase verde (15 s mín., 3 s amarillo, 60 s máx.)", "Discreta; |A<sub>i</sub>| = n.º de fases"],
             ["Recompensa r<sub>i</sub>", "Reducción de cola, espera de carros y espera de peatones; penalización por cambio de fase", "Por intersección, cada Δt"],
             ["Estado global s", "Concatenación de las cuatro observaciones", "Entrada del crítico"],
             ["Horizonte", "720 pasos de 5 s · factor de descuento γ = 0,99", "1 episodio = 1 h simulada"]]
    S.append(seccion("formulacion", NORMAL, marco(4, "Formulación", "Formulación del problema de control",
        "Juego estocástico parcialmente observable con un agente por intersección") + "\n" +
        tabla(["Elemento", "Definición", "Implementación"], filas, ["19%", "52%", "29%"], der_desde=9, tam=25)))

    # 5 · arquitectura
    S.append(seccion("arquitectura", NORMAL, marco(5, "Arquitectura", "MAPPO: entrenamiento centralizado, ejecución descentralizada") + f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:32px">
{caja_oscura("Actor π<sub>i</sub>(a<sub>i</sub> | o<sub>i</sub>)", "o<sub>i</sub> → 128 → 64 → |A<sub>i</sub>| · ReLU · softmax", "Distribución sobre las fases. Cuatro actores independientes, porque |o<sub>i</sub>| y |A<sub>i</sub>| cambian entre intersecciones.")}
{caja_oscura("Crítico V(s)", "s → 256 → 128 → V · ReLU", "Valor del estado global (las cuatro observaciones). Solo se usa en el entrenamiento.")}
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:32px">
{tarjeta("Objetivo PPO", "Pérdida recortada con ε = 0,2 y bono de entropía c<sub>H</sub> = 0,01.")}
{tarjeta("Ventaja GAE", "γ = 0,99 y λ = 0,95, con ventajas normalizadas por lote.")}
{tarjeta("Ejecución", "Cada controlador decide solo con o<sub>i</sub>: no necesita comunicarse con los demás.")}
</div>"""))

    # 6 · recompensa y RND
    formula = lambda nom, resto: (f'<p style="font-family:{MONO};font-size:31px;line-height:1.5;color:{T["oscura_texto"]}">'
                                  f'<span style="color:{T["oscura_titulo"]}">{nom}</span> {resto}</p>')
    S.append(seccion("recompensa", NORMAL, marco(6, "Recompensa", "Recompensa extrínseca y exploración por RND",
        "Random Network Distillation como bono de curiosidad con peso decreciente") + f"""
<div style="background:{T['oscura']};border:{T['oscura_borde']};border-radius:12px;padding:34px 44px;display:flex;flex-direction:column;gap:14px">
{formula("R_ext", "= −(w1·ΔQ + w2·ΔW_veh + w3·ΔW_ped) − P·1[cambio de fase]")}
{formula("R_int", "= ‖ f<sub>θ</sub>(s′) − g(s′) ‖²")}
{formula("R_total", "= R_ext + λ<sub>e</sub>·R_int,  λ<sub>e</sub> = max(0,01 ; 1 − e/E)")}
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:32px">
{tarjeta("Término extrínseco", "ΔQ y ΔW: cambio de la cola y de la espera entre pasos. w1 = w2 = 1, <b>w3 = 2</b> (los peatones pesan el doble), P = 0,5.")}
{tarjeta("Término intrínseco", "g: red aleatoria fija; f<sub>θ</sub>: predictor entrenado para imitarla. El error es alto en estados poco visitados.")}
{tarjeta("Decaimiento de λ", "Lineal en el episodio e de E: exploración al inicio y optimización del tráfico al final.", 2)}
</div>"""))

    # 7 · dos versiones
    filas = [["Crítico", "Una salida V(s), actualizada por turnos con la recompensa de cada agente", "Una salida V<sub>i</sub>(s) por agente + normalización de valores (ValueNorm), un solo optimizador"],
             ["Normalización de RND", "Cada observación con su propia media", "Media y desviación acumuladas; R<sub>int</sub> normalizada por su desviación"],
             ["Peso real de la curiosidad", f"≈ {n(PESO_RND[1], 1)} % de |R<sub>ext</sub>| (sin efecto)", f"≈ {n(PESO_RND[2], 0)} % de |R<sub>ext</sub>| (≈ {n(VECES_RND, 0)} veces más)"]]
    S.append(seccion("versiones", NORMAL, marco(7, "Arquitectura", "Dos versiones del algoritmo",
        "Mismo actor y misma recompensa; cambian el crítico y la normalización de la curiosidad") + "\n" +
        tabla(["", "Algoritmo 1", "Algoritmo 2"], filas, ["22%", "37%", "41%"], resaltar=(2,), der_desde=9) + f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:32px">
{tarjeta("Por qué un valor por agente", "Las recompensas de C2 y C3 son mucho mayores que las de C4: un único V(s) no puede ajustar los cuatro retornos a la vez.")}
{tarjeta("Por qué renormalizar RND", "Con la normalización original la curiosidad no tenía efecto: no se podía saber si RND servía o no.", 2)}
</div>"""))

    # 8 · configuración experimental
    filas = [["Episodios por corrida", "150"], ["Tasa de aprendizaje actor / crítico", "3·10⁻⁴ / 1·10⁻³"],
             ["Recorte PPO ε / entropía c<sub>H</sub>", "0,2 / 0,01"], ["Épocas PPO por episodio", "4"],
             ["γ / λ<sub>GAE</sub>", "0,99 / 0,95"], ["Tasa de aprendizaje RND", "1·10⁻⁴"],
             ["Corridas", "2 versiones × 6 config. × 3 semillas = 36"]]
    S.append(seccion("protocolo", NORMAL, marco(8, "Entrenamiento", "Configuración experimental",
        "Hiperparámetros del entrenamiento y protocolo de evaluación") + f"""
<div style="display:flex;gap:44px;align-items:flex-start">
<div style="width:860px">{tabla(["Hiperparámetro", "Valor"], filas, ["55%", "45%"], ancho="100%", tam=25)}</div>
<div style="flex:1;display:flex;flex-direction:column;gap:22px">
{tarjeta("Seis configuraciones", "Base, sin curiosidad, más exploración (c<sub>H</sub> = 0,05), prioriza espera (w2 = 2), penaliza cambio (P = 2) y prioriza peatones (w3 = 5).")}
{tarjeta("Evaluación pareada", "Valle y pico × demanda 80, 100 y 120 % × 5 semillas de SUMO: todos los métodos ven el mismo tráfico.")}
{tarjeta("1.170 episodios", "36 políticas + 3 líneas base; prueba t pareada.", 2)}
</div></div>"""))

    # 9 · curvas de aprendizaje
    S.append(seccion("curva", NORMAL, marco(9, "Resultados", "Curvas de aprendizaje",
        "Episodios de hora pico de la configuración base · promedio de 3 semillas, en bloques de 10 episodios") + f"""
<div style="display:flex;gap:48px">
<div style="width:1100px;display:flex;flex-direction:column;gap:8px;{caja_grafico};padding:16px 24px">
{svg_curva()}
{leyenda([(T['alg1'], '━━', 'Algoritmo 1'), (T['alg2'], '━━', 'Algoritmo 2'), (T['ref'], '╍╍', 'Tiempo fijo (evaluación)')])}
</div>
<div style="flex:1;display:flex;flex-direction:column;gap:28px">
{cifra(f"{n(CURVA[(1, 'ini')], 1)} → {n(CURVA[(1, 'fin')], 1)} s", "algoritmo 1: primeros y últimos 10 episodios en pico")}
{cifra(f"{n(CURVA[(2, 'ini')], 1)} → {n(CURVA[(2, 'fin')], 1)} s", "algoritmo 2: primeros y últimos 10 episodios en pico", 2)}
{cifra("≈ 30", "episodio en que se supera al tiempo fijo")}
</div></div>"""))

    # 10 · resultados en hora pico
    def fila(nom, m, dec=1):
        return [nom] + [n(M[c][m], dec) for c, _ in METODOS]
    filas = [fila("Espera media de los carros (s)", "espera_media", 2),
             fila("Espera media de los peatones (s)", "peatones_espera_media"),
             fila("Tiempo perdido por viaje (s)", "perdida_tiempo_media", 0),
             fila("Espera por viaje, percentil 95 (s)", "espera_viaje_p95", 0),
             fila("CO₂ emitido (kg/h)", "co2_total_kg", 0),
             fila("Viajes completados", "viajes_completados", 0),
             fila("Viajes sin terminar", "viajes_sin_terminar", 0),
             fila("Cambios de fase", "cambios_fase_total", 0)]
    S.append(seccion("resultados", "padding:168px 128px 88px;display:flex;flex-direction:column;gap:30px",
        marco(10, "Resultados", "Desempeño en hora pico frente a las líneas base",
              "Demanda nominal, 5 semillas pareadas · IA = configuración base, promedio de sus 3 semillas de entrenamiento") + "\n" +
        tabla(["Métrica"] + [m for _, m in METODOS], filas, ["32%", "13%", "13%", "15%", "13.5%", "13.5%"],
              resaltar=(0, 1), tam=25) + f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:32px">
{cifra(f"{pc(pct(a1['espera_media'], fijo['espera_media']))} / {pc(pct(a2['espera_media'], fijo['espera_media']))}", "espera de los carros frente al tiempo fijo (alg. 1 / alg. 2), p &lt; 0,001")}
{cifra(f"{pc(pct(a1['peatones_espera_media'], fijo['peatones_espera_media']))} / {pc(pct(a2['peatones_espera_media'], fijo['peatones_espera_media']))}", "espera de los peatones; max-pressure la sube " + pc(pct(mp['peatones_espera_media'], fijo['peatones_espera_media'])), 2)}
{cifra(pc(pct(a1['viajes_completados'], fijo['viajes_completados'])), "viajes completados en la hora pico frente al tiempo fijo")}
</div>"""))

    # 11 · carros vs peatones y sensibilidad a la demanda
    filas = []
    for esc in (0.8, 1.0, 1.2):
        f0 = ROB[esc]["fijo"]
        filas.append([f"{int(esc * 100)} %", n(f0, 2)] +
                     [f"{n(ROB[esc][c], 2)} · {pc(pct(ROB[esc][c], f0))}" for c in ("v2_base", "v2a2_base")])
    rob = tabla(["Demanda", "Fijo", "IA alg. 1", "IA alg. 2"], filas, ["19%", "15%", "33%", "33%"],
                resaltar=(1,), tam=24, ancho="100%")
    S.append(seccion("compromiso", NORMAL, marco(11, "Resultados", "Carros frente a peatones, y sensibilidad a la demanda",
        "Hora pico · abajo a la izquierda es mejor · puntos pequeños: las 36 políticas entrenadas") + f"""
<div style="display:flex;gap:44px;align-items:flex-start">
<div style="{caja_grafico};padding:16px 20px">{svg_compromiso()}</div>
<div style="flex:1;min-width:0;display:flex;flex-direction:column;gap:22px">
<h3 style="font-size:31px;font-weight:700;color:{T['tinta']}">Espera de los carros (s) con otra demanda, sin reentrenar</h3>
{rob}
{tarjeta("En valle", f"La mejora es menor: {pc(VALLE['v2_base'])} y {pc(VALLE['v2a2_base'])}, a la par del actuado ({pc(VALLE['actuado'])}). La hora pico es donde el agente se diferencia.", 2)}
</div></div>"""))

    # 12 · algoritmo 1 frente a 2 y RND
    pe, pp = ALG[("pico", "espera_media")], ALG[("pico", "peatones_espera_media")]
    ve, vp = ALG[("valle", "espera_media")], ALG[("valle", "peatones_espera_media")]
    fp = lambda p: "&lt; 0,001" if p < 0.001 else n(p, 2)
    filas = [["Espera de los carros, pico (s)", n(pe[0], 2), n(pe[1], 2), fp(pe[2])],
             ["Espera de los carros, valle (s)", n(ve[0], 2), n(ve[1], 2), fp(ve[2])],
             ["Espera de los peatones, pico (s)", n(pp[0], 2), n(pp[1], 2), fp(pp[2])],
             ["Espera de los peatones, valle (s)", n(vp[0], 2), n(vp[1], 2), fp(vp[2])],
             ["Variación entre semillas, pico (s)", n(ALG[("pico", "sd")][0], 2), n(ALG[("pico", "sd")][1], 2), "—"],
             ["Episodios trancados (gridlock)", str(TRANCADOS[1]), str(TRANCADOS[2]), "—"]]
    S.append(seccion("comparacion", NORMAL, marco(12, "Análisis", "Algoritmo 1 frente a 2, y ¿sirve la curiosidad?",
        "18 políticas de cada versión, pareadas por configuración y semilla · promedio de las 3 demandas") + "\n" +
        tabla(["Métrica", "Algoritmo 1", "Algoritmo 2", "p"], filas, ["46%", "18%", "18%", "18%"], resaltar=(4, 5), tam=25) + f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:32px">
{tarjeta("RND no aporta", f"Sin curiosidad se obtiene lo mismo: espera en pico {n(RND[1]['sin_rnd'], 2)} vs. {n(RND[1]['base'], 2)} s (alg. 1) y {n(RND[2]['sin_rnd'], 2)} vs. {n(RND[2]['base'], 2)} s (alg. 2). Con la versión 2 la conclusión es sólida, porque ahí R<sub>int</sub> sí pesa.")}
{tarjeta("Lectura", "La versión 2 es más correcta y estable, pero con 150 episodios no mejora el desempeño. Causa probable: con λ = 1 al inicio explora demasiado.", 2)}
</div>"""))

    # 13 · dashboard
    S.append(seccion("dashboard", NORMAL, marco(13, "Herramienta", "Dashboard conectado a los resultados reales") + f"""
<div style="display:flex;gap:44px">
<img src="dashboard_final.png" alt="Captura del dashboard de C.I.T.A.D.E.L." style="width:1060px;border-radius:12px;border:1px solid {T['borde']}">
<div style="flex:1;min-width:0;display:flex;flex-direction:column;gap:22px">
{tarjeta("Qué se elige", "Versión de la IA (alg. 1 o 2), hora, tráfico y método de comparación, sobre las 36 corridas y los 1.170 episodios.")}
{tarjeta("Simulación", "Dos métodos lado a lado, animados con 10 min de hora pico grabados en SUMO.", 2)}
<div style="display:flex;gap:24px;align-items:center;background:{T['oscura']};border:{T['oscura_borde']};border-radius:12px;padding:20px 24px">
<img src="qr_dashboard.png" alt="Código QR del dashboard" style="width:240px;height:240px;border-radius:8px;flex:none">
<div style="display:flex;flex-direction:column;gap:10px">
<p style="font-size:34px;font-weight:800;line-height:1.1;color:{T['oscura_titulo']}">Ábrelo en tu celular</p>
<p style="font-size:23px;line-height:1.35;color:{T['oscura_texto']}">Escanea el código con la cámara</p>
<p style="font-family:{MONO};font-size:17px;line-height:1.35;color:{T['oscura_texto']};opacity:0.85;overflow-wrap:anywhere">{URL_DASHBOARD.replace("https://", "")}</p>
</div></div>
</div></div>"""))

    # 14 · conclusiones
    S.append(seccion("conclusiones", NORMAL, marco(14, "Cierre", "Conclusiones") + f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:28px">
{tarjeta("01 · Desempeño", f"En hora pico la espera de los carros baja entre {n(-pct(a2['espera_media'], fijo['espera_media']), 0)} y {n(-pct(a1['espera_media'], fijo['espera_media']), 0)} % frente al tiempo fijo y se completan {n(pct(a1['viajes_completados'], fijo['viajes_completados']), 0)} % más viajes (p &lt; 0,001).")}
{tarjeta("02 · Carros y peatones", f"Es el único método que mejora las dos esperas. Max-pressure deja a los carros en {n(mp['espera_media'])} s, pero a los peatones en {n(mp['peatones_espera_media'], 0)} s, contra {n(a1['peatones_espera_media'], 0)}–{n(a2['peatones_espera_media'], 0)} s con el agente.", 2)}
{tarjeta("03 · Robustez", "Con 20 % más o menos demanda, sin reentrenar, la espera en pico sigue bajando más de 60 %.", 2)}
{tarjeta("04 · Método", "El algoritmo 2 corrige el crítico y RND: es más estable y no se tranca, pero no mejora el desempeño. La curiosidad no aporta en este problema.")}
</div>
<div style="background:{T['oscura']};border:{T['oscura_borde']};border-radius:12px;padding:26px 40px">
<p style="font-size:27px;line-height:1.45;color:{T['oscura_texto']}"><b style="color:{T['oscura_titulo']}">Trabajo futuro:</b> λ inicial menor o más episodios; reducir el tiempo perdido por viaje (hoy mayor que con actuado y max-pressure); validar con conteos reales de tráfico.</p>
</div>"""))

    # 15 · cierre
    if oscuro:
        S.append(seccion("cierre", "padding:300px 128px 100px;display:flex;flex-direction:column;gap:24px", f"""
<img src="uniandes.png" alt="Universidad de los Andes" style="position:absolute;left:128px;top:72px;width:270px;height:74px;object-fit:contain;filter:invert(1)">
<img src="citadel_logo.png" alt="" style="position:absolute;right:110px;top:260px;width:640px;filter:drop-shadow(0 0 40px rgba(233,117,168,0.25))">
<p style="font-family:{MONO};font-size:24px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{T['kicker']}">Control semafórico multiagente con RND-MAPPO</p>
<h1 style="font-size:88px;font-weight:800;line-height:1.05;letter-spacing:-1px;color:#FFFFFF">Gracias.<br><span style="{TEXTO_DEGRADADO}">¿Preguntas?</span></h1>
<div style="height:16px"></div>
<p style="font-size:28px;line-height:1.4;color:{T['texto']}">Tomás Castro · Andrés Felipe Barbosa · Mariana Vanoy · Santiago Sánchez</p>
<p style="font-family:{MONO};font-size:24px;line-height:1.4;color:{T['suave']}">github.com/santisan1205/Taller-Control</p>""", FONDO_PORTADA))
    else:
        S.append(seccion("cierre", "padding:300px 128px 100px;display:flex;flex-direction:column;gap:24px", f"""
<img src="uniandes.png" alt="Universidad de los Andes" style="position:absolute;left:128px;top:72px;width:300px;height:82px;object-fit:contain">
<div style="position:absolute;left:0;top:1040px;width:1920px;height:40px;background:{T['tinta']}"></div>
<div style="position:absolute;right:128px;top:240px;width:520px;height:520px;background:#000;border-radius:24px;overflow:hidden;display:flex;align-items:center;justify-content:center">
<img src="citadel_logo_oscuro.jpg" alt="" style="width:490px"></div>
<p style="font-family:{MONO};font-size:24px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{T['tinta']}">Control semafórico multiagente con RND-MAPPO</p>
<h1 style="font-size:88px;font-weight:800;line-height:1.05;letter-spacing:-1px;color:{T['tinta']}">Gracias. ¿Preguntas?</h1>
<div style="height:16px"></div>
<p style="font-size:28px;line-height:1.4;color:{T['tinta']}">Tomás Castro · Andrés Felipe Barbosa · Mariana Vanoy · Santiago Sánchez</p>
<p style="font-family:{MONO};font-size:24px;line-height:1.4;color:{T['tinta']}">github.com/santisan1205/Taller-Control</p>""", T["acento1"]))

    assert len(S) == TOTAL

    css = f"""<style>
  * {{ box-sizing: border-box; margin: 0; }}
  html, body {{ height: 100%; background: #0E1118; overflow: hidden; }}
  #lienzo {{ position: absolute; left: 50%; top: 50%; width: 1920px; height: 1080px; transform-origin: center; }}
  section {{ position: absolute; inset: 0; width: 1920px; height: 1080px; overflow: hidden; }}
  @media screen {{ section:not(.activa) {{ display: none !important; }} }}
  table {{ border-collapse: collapse; width: 100%; overflow: hidden; }}
  th, td {{ padding: 0.36em 0.6em; text-align: left; border-bottom: 1px solid {T['linea']}; }}
  th {{ font-weight: 700; }}
  hr {{ border: none; }}
  img {{ display: block; }}
  svg text {{ font-family: {T['fuente']}; }}
  .notas {{ display: none; white-space: pre-line; }}
  body.ver-notas section.activa .notas {{ display: block; position: absolute; left: 0; right: 0; bottom: 0;
    background: rgba(14,17,24,0.94); color: #F5F3EE; font: 26px/1.45 Arial, sans-serif;
    padding: 32px 64px; max-height: 50%; overflow: auto; }}
  @media print {{
    @page {{ size: 1920px 1080px; margin: 0; }}
    html, body {{ overflow: visible; background: none; }}
    #lienzo {{ position: static; transform: none !important; width: auto; height: auto; }}
    section {{ position: relative; page-break-after: always; break-after: page; }}
    .notas {{ display: none !important; }}
  }}
</style>"""
    return (f'<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>C.I.T.A.D.E.L. · Presentación final</title>\n'
            f'<link rel="stylesheet" href="{T["fuente_href"]}">\n'
            f'<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap">\n'
            f'{css}\n</head>\n<body>\n'
            f'<!-- Generado por presentacion/generar_final.py: no editar a mano. Teclas: flechas, N = notas, F = pantalla completa -->\n'
            f'<div id="lienzo">\n' + "".join(S) + f'</div>\n{JS}\n</body>\n</html>\n')


JS = """<script>
  const slides = [...document.querySelectorAll('section')];
  let actual = 0;
  function mostrar(i) {
    actual = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((s, j) => s.classList.toggle('activa', j === actual));
    history.replaceState(null, '', '#' + (actual + 1));
  }
  function ajustar() {
    const e = Math.min(innerWidth / 1920, innerHeight / 1080);
    document.getElementById('lienzo').style.transform = `translate(-50%, -50%) scale(${e})`;
  }
  addEventListener('resize', ajustar);
  addEventListener('keydown', ev => {
    if (['ArrowRight', 'PageDown', ' ', 'Enter'].includes(ev.key)) mostrar(actual + 1);
    else if (['ArrowLeft', 'PageUp', 'Backspace'].includes(ev.key)) mostrar(actual - 1);
    else if (ev.key === 'Home') mostrar(0);
    else if (ev.key === 'End') mostrar(slides.length - 1);
    else if (ev.key.toLowerCase() === 'n') document.body.classList.toggle('ver-notas');
    else if (ev.key.toLowerCase() === 'f') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  });
  addEventListener('click', ev => mostrar(actual + (ev.clientX > innerWidth / 3 ? 1 : -1)));
  ajustar();
  mostrar((parseInt(location.hash.slice(1)) || 1) - 1);
</script>"""

for tema, archivo in [(AMARILLA, "presentacion_final_amarilla.html"), (CITADEL, "presentacion_final_citadel.html")]:
    (SALIDA / archivo).write_text(construir(tema), encoding="utf-8")

# ================================================================== guion
TITULOS = {"cover": "Portada", "agenda": "Agenda de la presentación", "escenario": "Descripción del escenario",
           "formulacion": "Formulación del problema de control",
           "arquitectura": "MAPPO: entrenamiento centralizado, ejecución descentralizada",
           "recompensa": "Recompensa extrínseca y exploración por RND", "versiones": "Dos versiones del algoritmo",
           "protocolo": "Configuración experimental", "curva": "Curvas de aprendizaje",
           "resultados": "Desempeño en hora pico frente a las líneas base",
           "compromiso": "Carros frente a peatones, y sensibilidad a la demanda",
           "comparacion": "Algoritmo 1 frente a 2, y ¿sirve la curiosidad?",
           "dashboard": "Dashboard", "conclusiones": "Conclusiones", "cierre": "Gracias. ¿Preguntas?"}
NOMBRES = {"Tomy": "Tomy · Tomás Castro", "Andrés": "Andrés · Andrés Felipe Barbosa",
           "Santi": "Santi · Santiago Sánchez", "Mariana": "Mariana · Mariana Vanoy"}
TEMAS = {"Tomy": "Portada, agenda, escenario y formulación",
         "Andrés": "Arquitectura, recompensa, las dos versiones y configuración experimental",
         "Santi": "Curvas de aprendizaje, resultados en hora pico, carros vs. peatones y demanda",
         "Mariana": "Versión 1 vs. 2 y curiosidad, dashboard, conclusiones y preguntas"}
bloques = {}
for i, sid in enumerate(ORDEN, 1):
    bloques.setdefault(GUION[sid][0], []).append((i, sid))
filas_exp = "".join(
    f"<tr><td>{NOMBRES[q]}</td><td>{l[0][0]}–{l[-1][0]}</td><td>{TEMAS[q]}</td><td>{mmss(sum(GUION[s][1] for _, s in l))}</td></tr>"
    for q, l in bloques.items())

pe, pp = ALG[("pico", "espera_media")], ALG[("pico", "peatones_espera_media")]
CIFRAS = [
    ("Escenario", "4 cruces · peatones en C1 y C4 · valle y pico · ≈ 2.000 viajes/h en pico"),
    ("Restricciones", "verde mín. 15 s · amarillo 3 s · verde máx. 60 s"),
    ("Recompensa", "w1 = w2 = 1 · w3 = 2 (peatones) · P = 0,5"),
    ("Redes", "actor o → 128 → 64 → |A| · crítico s → 256 → 128 → V"),
    ("PPO", "ε = 0,2 · c_H = 0,01 · γ = 0,99 · λ_GAE = 0,95 · 4 épocas"),
    ("Experimentos", "36 corridas (2 alg. × 6 config. × 3 semillas) · 150 episodios"),
    ("Evaluación", "1.170 episodios · 5 semillas pareadas · 3 demandas · 3 líneas base"),
    ("Aprendizaje (pico)", f"{n(CURVA[(1, 'ini')], 1)} → {n(CURVA[(1, 'fin')], 1)} s (alg. 1) · {n(CURVA[(2, 'ini')], 1)} → {n(CURVA[(2, 'fin')], 1)} s (alg. 2)"),
    ("Espera carros, pico", f"fijo {n(fijo['espera_media'], 2)} → {n(a1['espera_media'], 2)} / {n(a2['espera_media'], 2)} s ({pc(pct(a1['espera_media'], fijo['espera_media']))} / {pc(pct(a2['espera_media'], fijo['espera_media']))})"),
    ("Espera peatones, pico", f"fijo {n(fijo['peatones_espera_media'])} · agente {n(a1['peatones_espera_media'])}–{n(a2['peatones_espera_media'])} · max-pressure {n(mp['peatones_espera_media'])} s"),
    ("Max-pressure", f"carros {n(mp['espera_media'])} s (mejor), peatones {n(mp['peatones_espera_media'], 0)} s (peor)"),
    ("Viajes y CO₂", f"viajes {pc(pct(a1['viajes_completados'], fijo['viajes_completados']))} · CO₂ {pc(pct(a1['co2_total_kg'], fijo['co2_total_kg']))}"),
    ("Robustez", "demanda ±20 % sin reentrenar: espera en pico −61 a −73 %"),
    ("Alg. 1 vs 2", f"carros {n(pe[0], 2)} vs {n(pe[1], 2)} s (p = {n(pe[2], 2)}) · peatones {n(pp[0], 2)} vs {n(pp[1], 2)} s (p &lt; 0,001)"),
    ("Curiosidad", f"pesa ≈ {n(PESO_RND[1], 1)} % (alg. 1) y ≈ {n(PESO_RND[2], 0)} % (alg. 2) · sin RND = base"),
]
cifras_html = "".join(f"<p><span>{a}:</span> {b}</p>" for a, b in CIFRAS)

tarjetas = ""
for q, l in bloques.items():
    tarjetas += f'<h2 class="exp">{NOMBRES[q]}</h2>\n'
    for i, sid in l:
        _, dur, texto = GUION[sid]
        tarjetas += (f'<div class="card"><div class="cab"><h3>{i} · {TITULOS[sid]}</h3>'
                     f'<span>{mmss(dur)} · acumulado {mmss(ACUM[sid])}</span></div><p>{texto}</p></div>\n')

PREGUNTAS = [
    ("¿Por qué MAPPO y no un solo agente para toda la red?", "Un agente central tendría un espacio de acciones combinatorio (el producto de las fases de los cuatro cruces) y necesitaría comunicación en operación. Con entrenamiento centralizado y ejecución descentralizada cada semáforo decide con sus propios sensores."),
    ("Si max-pressure deja menos espera de carros, ¿para qué el agente?", f"Porque max-pressure no ve a los peatones: los hace esperar {n(mp['peatones_espera_media'], 0)} s, casi el triple que el tiempo fijo. El agente baja las dos esperas a la vez. Si solo importaran los carros, max-pressure sería mejor, y lo decimos."),
    ("¿Por qué el tiempo perdido por viaje es mayor que con actuado o max-pressure?", "La recompensa optimiza cola y espera en cada cruce, no el tiempo de viaje completo, y además cuida a los peatones, lo que le cuesta verde a los carros. Es la primera mejora que proponemos."),
    ("¿Por qué hicieron una segunda versión si no mejora?", "Porque la primera tenía dos fallas de método: un crítico que no podía ajustar los cuatro retornos y una curiosidad que no tenía efecto. La segunda es correcta y más estable (ningún gridlock); que no mejore el desempeño también es un resultado."),
    ("Entonces, ¿RND sirve?", "En este problema no: sin RND se obtiene lo mismo en las dos versiones. Con la versión 2 lo podemos afirmar porque ahí la curiosidad sí pesa. Lo siguiente sería un λ inicial menor o más episodios."),
    ("¿Qué pasó con el gridlock?", "Una política de la versión 1 (prioriza peatones, semilla 1) se trancó en hora pico con 20 % más demanda: 94 s de espera. Es la única alerta de toda la evaluación; con la versión 2 no ocurre."),
    ("¿Qué tan confiables son las diferencias?", "Comparación pareada: todos los métodos ven exactamente el mismo tráfico en cada semilla. Frente al tiempo fijo todos los p son menores que 0,001. Entre versiones, la espera de los carros no es significativa (p ≈ 0,84) y la de los peatones sí."),
    ("¿Por qué los números son distintos a los de octubre?", "En octubre evaluamos solo la hora valle y sin peatones, y sumo-rl no aplicaba el verde máximo. Ahora el escenario es más completo (peatones, hora pico, demanda variable, verde máximo de 60 s) y por eso reportamos estos resultados."),
    ("¿Se podría implementar en campo?", "Cada controlador solo necesita sensores de su cruce (colas, densidad y peatones esperando). Faltaría validar con conteos reales y pruebas de seguridad."),
    ("¿Cuánto tarda entrenar?", "Del orden de medio minuto de cómputo por hora simulada: una corrida de 150 episodios toma un par de horas en CPU. Las 36 corridas se repartieron en cuatro computadores."),
]
preg_html = "".join(f"<p><b>{a}</b> {b}</p>" for a, b in PREGUNTAS)
total = ACUM[ORDEN[-1]]

GUION_CSS = """<style>
  @page { size: letter; margin: 14mm 16mm; }
  * { box-sizing: border-box; margin: 0; }
  body { font-family: 'Inter', Arial, sans-serif; color: #1C1924; font-size: 11pt; line-height: 1.5; background: #FFFFFF; max-width: 820px; margin: 0 auto; padding: 0 16px; }
  .top { background: linear-gradient(135deg, #2A1B38 0%, #141118 100%); border-radius: 10px; padding: 14px 22px; display: flex; justify-content: space-between; align-items: center; margin: 18px 0 20px; }
  .top .marca { display: flex; align-items: center; gap: 12px; }
  .top .marca img { height: 40px; }
  .top .marca span { font-weight: 800; letter-spacing: 2px; background: linear-gradient(90deg, #B57BE0, #E975A8); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .top .uni { height: 26px; filter: invert(1); }
  .kick { font-size: 9pt; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; color: #9D50BB; border-left: 26px solid #E975A8; padding-left: 10px; line-height: 4px; margin: 6px 0 14px; }
  h1 { font-size: 23pt; line-height: 1.15; letter-spacing: -0.5px; }
  .sub { color: #5E5A6B; font-size: 12pt; margin: 6px 0 16px; }
  table { border-collapse: collapse; width: 100%; font-size: 10pt; margin-bottom: 18px; }
  th { background: #2E2140; color: #fff; text-align: left; padding: 7px 10px; }
  td { padding: 7px 10px; border-bottom: 1px solid #ECE8F2; vertical-align: top; }
  .cifras { background: #1C1924; color: #E6E2EE; border-radius: 10px; padding: 16px 22px; display: grid; grid-template-columns: 1fr 1fr; gap: 6px 22px; font-size: 9.4pt; }
  .cifras h3 { grid-column: 1 / -1; color: #E975A8; font-size: 12pt; margin-bottom: 4px; }
  .cifras span { color: #E975A8; font-weight: 700; }
  .aviso { background: #F7EEF8; border-left: 5px solid #9D50BB; border-radius: 8px; padding: 12px 16px; margin: 16px 0; font-size: 10pt; }
  .aviso p + p { margin-top: 5px; }
  h2.exp { font-size: 9.5pt; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; color: #9D50BB; border-left: 26px solid #E975A8; padding-left: 10px; line-height: 4px; margin: 26px 0 14px; break-after: avoid; }
  h2.exp:not(:first-of-type) { break-before: page; }
  .card { background: #FAF8FC; border: 1px solid #ECE8F2; border-left: 5px solid #E975A8; border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; break-inside: avoid; }
  .cab { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 6px; }
  .cab h3 { font-size: 12pt; }
  .cab span { font-family: 'JetBrains Mono', monospace; font-size: 8.5pt; color: #9D50BB; white-space: nowrap; }
  .preg { break-before: page; }
  .preg p { margin-bottom: 9px; font-size: 10.2pt; }
</style>"""

guion = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Guion · C.I.T.A.D.E.L. · presentación final</title>
<link rel="stylesheet" href="{CITADEL['fuente_href']}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap">
{GUION_CSS}
</head><body>
<!-- Generado por presentacion/generar_final.py: no editar a mano -->
<div class="top"><div class="marca"><img src="citadel_emblema.png" alt=""><span>C.I.T.A.D.E.L.</span></div><img class="uni" src="uniandes.png" alt="Uniandes"></div>
<p class="kick">Guion de la presentación final</p>
<h1>Control semafórico multiagente con RND-MAPPO</h1>
<p class="sub">Profesor: Jorge Alfredo López Jiménez · {TOTAL} diapositivas · unos {total // 60} min {total % 60} s · 4 expositores</p>
<table><tr><th>Expositor</th><th>Diapositivas</th><th>Tema</th><th>Tiempo</th></tr>{filas_exp}</table>
<div class="cifras"><h3>Cifras para memorizar (hora pico, demanda nominal salvo que se diga otra cosa)</h3>{cifras_html}</div>
<div class="aviso">
<p><b>Lo que todos deben saber explicar:</b></p>
<p>1. El agente baja las dos esperas a la vez; max-pressure gana solo en carros y castiga a los peatones. En tiempo perdido por viaje el agente queda detrás del actuado y de max-pressure: decirlo si preguntan.</p>
<p>2. Las dos versiones del algoritmo: la 2 tiene un crítico con un valor por agente y una curiosidad que sí pesa. Mismo desempeño en carros, peatones ~1 s peor, más estable y sin gridlock.</p>
<p>3. La curiosidad (RND) no aporta en este problema: sin RND da lo mismo en las dos versiones.</p>
</div>
{tarjetas}
<div class="preg">
<p class="kick" style="margin-top:20px">Preguntas probables y respuesta corta</p>
{preg_html}
</div>
</body></html>
"""
(SALIDA / "guion_final.html").write_text(guion, encoding="utf-8")

for q, l in bloques.items():
    print(f"{q:8s} {mmss(sum(GUION[s][1] for _, s in l))}  palabras: {sum(len(GUION[s][2].split()) for _, s in l)}")
print("total", mmss(total), "· palabras", sum(len(v[2].split()) for v in GUION.values()))
