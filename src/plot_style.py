"""
Estilo gráfico padronizado para figuras de artigo (IEEE, duas colunas).

- Largura de uma coluna IEEE: ~3.5 in; largura total: ~7.16 in.
- Fontes de 8-9 pt (legíveis depois de inseridas no PDF).
- Salva em PDF (vetorial, para o LaTeX) e PNG 300 dpi (para o notebook/README).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as cfg

COL_W = 3.5     # largura de uma coluna (polegadas)
FULL_W = 7.16   # largura de página inteira (polegadas)


def set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "axes.edgecolor": cfg.INK_2,
        "axes.labelcolor": cfg.INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": cfg.GRID,
        "grid.linewidth": 0.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.color": cfg.INK_2,
        "ytick.color": cfg.INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# No notebook, defina plot_style.SHOW = True para exibir as figuras inline.
SHOW = False


def savefig(fig, name: str) -> None:
    """Salva a figura em PDF e PNG dentro de outputs/figures."""
    fig.savefig(cfg.FIG_DIR / f"{name}.pdf")
    fig.savefig(cfg.FIG_DIR / f"{name}.png")
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
