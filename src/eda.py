"""
Análise Exploratória de Dados (EDA).

Cada função faz uma etapa e devolve tabelas (DataFrames) para inspeção;
as figuras são salvas em outputs/figures. O notebook chama as funções
uma a uma, e o main.py chama `run_eda` que executa tudo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config as cfg
from .plot_style import COL_W, FULL_W, savefig
from .utils import save_table

NUMERIC = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
           "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
           "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
           "Temp9am", "Temp3pm"]
CATEG = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday",
         "RainTomorrow"]

BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#bcd4f2", "#6fa3e6", "#2a78d6", "#1a4f93", "#0d2c55"])


# --------------------------------------------------------------------------
# 1. Visão geral
# --------------------------------------------------------------------------
def overview(df: pd.DataFrame) -> dict:
    info = {
        "n_linhas": len(df),
        "n_colunas": df.shape[1],
        "n_estacoes": df["Location"].nunique(),
        "data_inicio": df["Date"].min(),
        "data_fim": df["Date"].max(),
        "n_numericas": len(NUMERIC),
        "n_categoricas": len(CATEG),
        "pct_celulas_faltantes": 100 * df.isna().mean().mean(),
        "pct_linhas_completas": 100 * df.dropna().shape[0] / len(df),
    }
    return info


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas descritivas das variáveis numéricas + % de faltantes."""
    d = df[NUMERIC].describe().T
    d["Faltantes (%)"] = 100 * df[NUMERIC].isna().mean()
    d["Assimetria"] = df[NUMERIC].skew()
    tab = d[["mean", "std", "min", "50%", "max", "Assimetria", "Faltantes (%)"]]
    tab.columns = ["Média", "DP", "Mín", "Mediana", "Máx", "Assimetria", "Faltantes (%)"]
    save_table(tab, "tab_descritiva", float_fmt="{:.1f}")
    return tab


# --------------------------------------------------------------------------
# 2. Dados ausentes
# --------------------------------------------------------------------------
def missing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Percentual de faltantes por variável e por estação.

    A matriz estação x variável mostra que boa parte da ausência é
    ESTRUTURAL: várias estações simplesmente não medem Sunshine,
    Evaporation ou Cloud (100% faltante), o que não é ausência aleatória
    (MCAR). Isso justifica criar indicadores de ausência como features.
    """
    miss = 100 * df.isna().mean().sort_values(ascending=True)
    miss = miss[miss > 0]
    by_loc = df.groupby("Location")[NUMERIC].apply(lambda g: g.isna().mean() * 100)
    cols = miss.index[::-1][: 10]
    cols = [c for c in cols if c in NUMERIC]
    by_loc = by_loc[cols].sort_values(cols[0])

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.9),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    ax.barh(miss.index, miss.values, color=cfg.C["blue"], height=0.7)
    for y, v in enumerate(miss.values):
        ax.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=6.5, color=cfg.INK_2)
    ax.set_xlabel("Valores ausentes (%)")
    ax.set_title("(a) Ausência por variável")
    ax.set_xlim(0, miss.max() * 1.18)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    im = ax.imshow(by_loc.T.values, aspect="auto", cmap=BLUES, vmin=0, vmax=100,
                   interpolation="nearest")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_xticks(range(len(by_loc)))
    ax.set_xticklabels(by_loc.index, rotation=90, fontsize=4.8)
    ax.set_title("(b) Ausência por estação (%)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    savefig(fig, "eda_missing")

    full_missing = (by_loc == 100).sum()
    out = pd.DataFrame({"Faltantes (%)": 100 * df[NUMERIC].isna().mean(),
                        "Estações sem medição": (df.groupby("Location")[NUMERIC]
                                                 .apply(lambda g: g.isna().all()).sum())})
    return out.sort_values("Faltantes (%)", ascending=False)


# --------------------------------------------------------------------------
# 3. Inconsistências
# --------------------------------------------------------------------------
def consistency_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica regras físicas e lógicas que os dados deveriam obedecer."""
    nxt = df.groupby("Location")["RainToday"].shift(-1)
    nxt_date = df.groupby("Location")["Date"].shift(-1)
    consecutive = (nxt_date - df["Date"]).dt.days.eq(1)
    both = consecutive & nxt.notna() & df["RainTomorrow"].notna()
    x = df.dropna(subset=["Rainfall", "RainToday"])
    checks = {
        "Linhas duplicadas (Date, Location)": df.duplicated(["Date", "Location"]).sum(),
        "MinTemp > MaxTemp": (df["MinTemp"] > df["MaxTemp"]).sum(),
        "Umidade fora de [0, 100]": ((df[["Humidity9am", "Humidity3pm"]] < 0) |
                                     (df[["Humidity9am", "Humidity3pm"]] > 100)).any(axis=1).sum(),
        "Precipitação negativa": (df["Rainfall"] < 0).sum(),
        "Nebulosidade > 8 oitavos": ((df["Cloud9am"] > 8) | (df["Cloud3pm"] > 8)).sum(),
        "RainToday incoerente com Rainfall > 1 mm": ((x["Rainfall"] > 1) != (x["RainToday"] == "Yes")).sum(),
        "RainTomorrow(t) diferente de RainToday(t+1)": (df.loc[both, "RainTomorrow"] != nxt[both]).sum(),
        "Datas faltantes no calendário (estação-dia)": int(
            df.groupby("Location")["Date"].agg(lambda s: (s.max() - s.min()).days + 1 - s.size).sum()),
    }
    tab = pd.DataFrame({"Ocorrências": checks})
    tab.to_csv(cfg.TAB_DIR / "tab_inconsistencias.csv")
    return tab


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as correções decorrentes das checagens.

    - Nebulosidade = 9 (céu obstruído) não é uma medida em oitavos: vira NaN.
    """
    df = df.copy()
    for c in ["Cloud9am", "Cloud3pm"]:
        df.loc[df[c] > 8, c] = np.nan
    return df


# --------------------------------------------------------------------------
# 4. Distribuições e outliers
# --------------------------------------------------------------------------
def distributions(df: pd.DataFrame) -> None:
    cols = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
            "WindGustSpeed", "Humidity3pm", "Pressure3pm", "Cloud3pm"]
    fig, axes = plt.subplots(3, 3, figsize=(FULL_W, 4.6))
    for ax, c in zip(axes.ravel(), cols):
        v = df[c].dropna()
        if c in ("Rainfall", "Evaporation"):
            ax.hist(np.log1p(v), bins=50, color=cfg.C["blue"], edgecolor="white", linewidth=0.3)
            ax.set_title(f"log(1 + {c})")
        else:
            ax.hist(v, bins=40, color=cfg.C["blue"], edgecolor="white", linewidth=0.3)
            ax.set_title(c)
        ax.text(0.98, 0.92, f"assim. = {v.skew():.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, color=cfg.INK_2)
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    savefig(fig, "eda_distribuicoes")


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    """Contagem de outliers pela regra de Tukey (1,5 x IQR)."""
    rows = {}
    for c in NUMERIC:
        v = df[c].dropna()
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = ((v < lo) | (v > hi)).sum()
        rows[c] = {"Limite inf.": lo, "Limite sup.": hi, "Outliers": n,
                   "Outliers (%)": 100 * n / len(v)}
    tab = pd.DataFrame(rows).T
    tab["Outliers"] = tab["Outliers"].astype(int)
    tab = tab.sort_values("Outliers (%)", ascending=False)
    save_table(tab, "tab_outliers", float_fmt="{:.1f}")

    # boxplot das variáveis padronizadas (z-score), para comparar escalas
    z = (df[NUMERIC] - df[NUMERIC].mean()) / df[NUMERIC].std()
    order = tab.index.tolist()
    fig, ax = plt.subplots(figsize=(FULL_W, 2.6))
    data = [z[c].dropna().values for c in order]
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.55,
                    flierprops=dict(marker=".", markersize=1.5, alpha=0.25,
                                    markerfacecolor=cfg.C["orange"], markeredgecolor="none"),
                    medianprops=dict(color=cfg.INK, linewidth=1),
                    boxprops=dict(facecolor="#d6e4f7", edgecolor=cfg.C["blue"], linewidth=0.8),
                    whiskerprops=dict(color=cfg.C["blue"], linewidth=0.8),
                    capprops=dict(color=cfg.C["blue"], linewidth=0.8))
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Valor padronizado (z)")
    ax.axhline(0, color=cfg.INK_2, linewidth=0.5)
    fig.tight_layout()
    savefig(fig, "eda_outliers")
    return tab


def correlation(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[NUMERIC].corr()
    fig, ax = plt.subplots(figsize=(COL_W, 3.3))
    div = LinearSegmentedColormap.from_list(
        "div", ["#c2410c", "#eb6834", "#f6c3a9", "#f2f2f0", "#a9c9f0", "#2a78d6", "#1a4f93"])
    im = ax.imshow(corr.values, cmap=div, vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUMERIC)))
    ax.set_yticks(range(len(NUMERIC)))
    ax.set_xticklabels(NUMERIC, rotation=90, fontsize=5.5)
    ax.set_yticklabels(NUMERIC, fontsize=5.5)
    for i in range(len(NUMERIC)):
        for j in range(len(NUMERIC)):
            v = corr.values[i, j]
            if abs(v) >= 0.6 and i != j:
                ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center",
                        fontsize=3.8, color="white" if abs(v) > 0.75 else cfg.INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    savefig(fig, "eda_correlacao")
    return corr


# --------------------------------------------------------------------------
# 5. Alvo da classificação
# --------------------------------------------------------------------------
def target_analysis(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["RainTomorrow"])
    y = (d["RainTomorrow"] == "Yes").astype(int)
    share = y.mean()
    by_month = y.groupby(d["Date"].dt.month).mean() * 100
    by_loc = y.groupby(d["Location"]).mean().sort_values() * 100

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.3),
                             gridspec_kw={"width_ratios": [0.55, 1, 1.6]})
    ax = axes[0]
    counts = y.value_counts().sort_index()
    ax.bar(["Não", "Sim"], counts.values, color=[cfg.C["blue"], cfg.C["orange"]], width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{100 * v / counts.sum():.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_title("(a) Classes")
    ax.set_ylabel("Dias")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ax.bar(by_month.index, by_month.values, color=cfg.C["blue"], width=0.7)
    ax.axhline(100 * share, color=cfg.INK_2, linestyle="--", linewidth=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_title("(b) % de Sim por mês")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    ax.bar(range(len(by_loc)), by_loc.values, color=cfg.C["blue"], width=0.75)
    ax.axhline(100 * share, color=cfg.INK_2, linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(by_loc)))
    ax.set_xticklabels(by_loc.index, rotation=90, fontsize=4.6)
    ax.set_title("(c) % de Sim por estação")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, "eda_alvo_classificacao")
    return pd.DataFrame({"Proporção de Sim (%)": [100 * share],
                         "Razão Não:Sim": [(1 - share) / share]})


# --------------------------------------------------------------------------
# 6. Relações das variáveis com o alvo
# --------------------------------------------------------------------------
def features_vs_target(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["RainTomorrow"])
    cols = ["Humidity3pm", "Sunshine", "Cloud3pm", "Pressure3pm", "WindGustSpeed", "Rainfall"]
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 3.4))
    for ax, c in zip(axes.ravel(), cols):
        for lab, col, name in (("No", cfg.C["blue"], "Não"), ("Yes", cfg.C["orange"], "Sim")):
            v = d.loc[d["RainTomorrow"] == lab, c].dropna()
            if c == "Rainfall":
                v = np.log1p(v)
            ax.hist(v, bins=40, density=True, alpha=0.55, color=col, label=name,
                    edgecolor="none")
        ax.set_title("log(1 + Rainfall)" if c == "Rainfall" else c)
        ax.set_yticks([])
        ax.grid(False)
    axes[0, 0].legend(title="Chove amanhã?", title_fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "eda_features_alvo")


def run_eda(df: pd.DataFrame) -> dict:
    info = overview(df)
    descriptive_table(df)
    miss = missing_analysis(df)
    checks = consistency_checks(df)
    dfc = clean(df)
    distributions(dfc)
    outlier_table(dfc)
    correlation(dfc)
    tgt = target_analysis(dfc)
    features_vs_target(dfc)
    info["checagens"] = checks["Ocorrências"].to_dict()
    info["estacoes_sem_sunshine"] = int(miss.loc["Sunshine", "Estações sem medição"])
    info["estacoes_sem_evaporation"] = int(miss.loc["Evaporation", "Estações sem medição"])
    info["pct_rain_tomorrow"] = float(tgt.iloc[0, 0])
    return info
