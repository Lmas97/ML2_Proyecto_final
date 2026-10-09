"""Estilo común y gráficos reutilizados por los notebooks."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_COLORS = {"Reglas (Milojević)": "#8a8984", "Decision Tree": PALETTE[1],
                "Random Forest": PALETTE[0], "Hist-GBDT": PALETTE[2]}
ART_COLOR, SIM_COLOR = "#b5b3ab", PALETTE[0]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e4dd"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
        "font.size": 10, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save(fig, name: str):
    fig.savefig(C.FIGURES / name)


def paired_bars(ax, labels, art, sim, ylabel="", fmt="{:.2f}", ylim=None):
    """Barras pareadas artículo vs simulación con etiquetas de valor."""
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, art, w, color=ART_COLOR, label="Artículo")
    b2 = ax.bar(x + w / 2 + 0.01, sim, w, color=SIM_COLOR, label="Simulación")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            if np.isfinite(v):
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                            fontsize=8, color=MUTED, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2)
    return ax


def ratio_effect(ax, table: pd.DataFrame, subset: str, metric: str, scoring: str = "accuracy"):
    """Efecto de la proporción de clases en una métrica BCubed para los 3 algoritmos."""
    order = ["{0:1,1:2}", "{0:1,1:1}", "original", "{0:2,1:1}", "{0:5,1:1}"]
    sub = table[(table.subset == subset) & (table.scoring == scoring)]
    for algo in ["Decision Tree", "Random Forest", "Hist-GBDT"]:
        s = sub[sub.algorithm == algo].set_index("ratio").reindex(order)[metric]
        ax.plot(order, s.values, marker="o", lw=2, ms=6, color=MODEL_COLORS[algo], label=algo)
    ax.set_xlabel("proporción {clase 0 : clase 1} en entrenamiento")
    ax.grid(axis="x", visible=False)
    return ax
