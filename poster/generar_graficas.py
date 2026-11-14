"""Gráficas del póster (Semana de la Innovación) a partir de datos_dashboard/v2.

Uso (desde la carpeta del proyecto):
    python poster/generar_graficas.py
Genera SVG en poster/graficas/ y imprime los KPI que van en el póster.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos_dashboard" / "v2"
SALIDA = Path(__file__).resolve().parent / "graficas"
SALIDA.mkdir(exist_ok=True)

# Paleta (validada con el validador de dataviz; el gris es la línea base, con etiqueta directa)
VERDE = "#1E9E63"      # RND-MAPPO (nuestro)
MORADO = "#6C5BD6"     # max-pressure
AMBAR = "#C9821F"      # actuado
GRIS = "#6B6B6B"       # tiempo fijo
TINTA = "#2F3A34"
TINTA2 = "#5E6B63"
REJILLA = "#DCE6DF"

ESTRATEGIAS = [
    ("fijo", "Tiempo fijo (hoy)", GRIS),
    ("actuado", "Actuado (SUMO)", AMBAR),
    ("max_pressure", "Max-pressure", MORADO),
    ("v2_base", "RND-MAPPO (nuestro)", VERDE),
]

plt.rcParams.update({
    "font.family": "Verdana",
    "font.size": 19,
    "axes.edgecolor": "#9AA8A0",
    "axes.labelcolor": TINTA2,
    "xtick.color": TINTA2,
    "ytick.color": TINTA2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": REJILLA,
    "grid.linewidth": 1,
    "svg.fonttype": "none",
})

d = pd.read_csv(DATOS / "evaluacion_detalle.csv")
d["config"] = d.politica.str.replace(r"_seed\d", "", regex=True)


def media(ventana, escala, config, metrica):
    s = d[(d.ventana == ventana) & (d.escala == escala) & (d.config == config)]
    return s[metrica].mean()


# ---------------------------------------------------------------- KPI
print("KPI hora pico, demanda nominal (RND-MAPPO base vs tiempo fijo):")
kpi = {}
for m, sentido in [("espera_media", -1), ("peatones_espera_media", -1),
                   ("perdida_tiempo_media", -1), ("co2_total_kg", -1),
                   ("viajes_completados", 1)]:
    f, r = media("pico", 1.0, "fijo", m), media("pico", 1.0, "v2_base", m)
    kpi[m] = (r - f) / f * 100
    print(f"  {m:24s} fijo {f:8.2f}  rl {r:8.2f}  cambio {kpi[m]:+.1f} %")
for v in ["valle", "pico"]:
    print(v, {c: round(media(v, 1.0, c, "peatones_espera_media"), 1) for c, *_ in ESTRATEGIAS})


# ------------------------------------------- 1. compromiso vehículos/peatones
# Un solo panel (hora pico, demanda nominal): es donde están las cifras del póster.
fig, ax = plt.subplots(figsize=(13.2, 3.95))
ventana = "pico"
# 18 políticas entrenadas (6 configuraciones x 3 semillas): nube verde clara
pol = (d[(d.ventana == ventana) & (d.escala == 1.0) & (d.modo == "rnd_mappo")]
       .groupby("politica")[["peatones_espera_media", "espera_media"]].mean())
ax.scatter(pol.peatones_espera_media, pol.espera_media, s=90, color=VERDE,
           alpha=0.28, edgecolor="none", zorder=2)
for config, nombre, color in ESTRATEGIAS:
    x = media(ventana, 1.0, config, "peatones_espera_media")
    y = media(ventana, 1.0, config, "espera_media")
    ax.scatter(x, y, s=380, color=color, edgecolor="white", linewidth=2.5, zorder=4)
    dx, dy, ha = 1.2, 0.0, "left"
    if config == "max_pressure":  # encima del punto para no chocar con la etiqueta del agente
        dx, dy, ha = 0.0, 1.0, "center"
    ax.annotate(nombre, (x, y), xytext=(x + dx, y + dy), ha=ha, va="center",
                fontsize=19, color=TINTA, fontweight="bold" if config == "v2_base" else "normal",
                zorder=5)
ax.set_xlim(0, 42)
ax.set_ylim(0, 9)
ax.set_xlabel("Espera de peatones (s)")
ax.set_ylabel("Espera de vehículos (s)")
ax.annotate("", xy=(0.8, 0.25), xytext=(4.6, 1.1),
            arrowprops=dict(arrowstyle="-|>", color=TINTA2, lw=1.8))
ax.text(5.0, 1.15, "mejor", color=TINTA2, fontsize=17, va="center")
ax.text(19, 7.0, "Puntos claros: las 18 políticas entrenadas\n(6 variantes × 3 semillas)",
        fontsize=16, color=TINTA2, va="top")
fig.tight_layout()
fig.savefig(SALIDA / "compromiso.svg", transparent=True)
plt.close(fig)


# ------------------------------------------- 2. robustez ante la demanda (pico)
fig, axes = plt.subplots(1, 2, figsize=(13.2, 3.3))
escalas = [0.8, 1.0, 1.2]
ancho = 0.2
for ax, metrica, titulo in [(axes[0], "espera_media", "Espera de vehículos (s)"),
                            (axes[1], "peatones_espera_media", "Espera de peatones (s)")]:
    for i, (config, nombre, color) in enumerate(ESTRATEGIAS):
        vals = [media("pico", e, config, metrica) for e in escalas]
        xs = np.arange(len(escalas)) + (i - 1.5) * (ancho + 0.012)
        ax.bar(xs, vals, width=ancho, color=color, label=nombre, zorder=3,
               edgecolor="white", linewidth=1.5)
        if config == "v2_base":
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.012 * ax.get_ylim()[1], f"{v:.1f}".replace(".", ","),
                        ha="center", va="bottom", fontsize=17, color=TINTA, fontweight="bold")
    ax.set_xticks(range(len(escalas)), ["80 %", "100 %", "120 %"])
    ax.set_xlabel("Demanda vehicular (hora pico)")
    ax.set_title(titulo, loc="left", fontsize=21, color=TINTA, fontweight="bold", pad=10)
    ax.grid(axis="x", visible=False)
# las etiquetas se recalculan con el ylim final
for ax in axes:
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=18,
           bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=1.6)
fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=3)
fig.savefig(SALIDA / "robustez.svg", transparent=True)
plt.close(fig)


# ------------------------------------------- 3. aprendizaje: con y sin RND
t = pd.read_csv(DATOS / "entrenamiento_episodios.csv")
fig, ax = plt.subplots(figsize=(7.4, 3.2))
for config, nombre, color, estilo in [("v2_base", "Con RND", VERDE, "-"),
                                      ("v2_sin_rnd", "Sin RND", GRIS, "--")]:
    s = (t[t.config == config].pivot_table(index="episodio", columns="semilla_entrenamiento",
                                           values="espera_media"))
    s = s.rolling(15, min_periods=5, center=True).median()
    m = s.mean(axis=1)
    ax.fill_between(m.index, s.min(axis=1), s.max(axis=1), color=color, alpha=0.12, lw=0)
    ax.plot(m.index, m, color=color, lw=2.6, ls=estilo, label=nombre)
ax.set_xlim(0, 152)
ax.legend(frameon=False, fontsize=19, loc="upper right", handlelength=2.2)
ax.set_ylim(0, None)
ax.set_xlabel("Episodio de entrenamiento")
ax.set_title("Espera de vehículos al entrenar (s)", loc="left", fontsize=21, color=TINTA,
             fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(SALIDA / "aprendizaje.svg", transparent=True)
plt.close(fig)

print("Listo:", sorted(p.name for p in SALIDA.glob("*.svg")))
