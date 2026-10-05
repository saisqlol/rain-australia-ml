"""
Classificação: "vai chover amanhã?" (RainTomorrow), binária e desbalanceada
(~22% de Sim).

Modelos escolhidos:
  1. Regressão Logística  - GLM binomial com ligação logit (baseline interpretável)
  2. KNN                  - classificador não paramétrico baseado em distância
  3. Random Forest        - bagging de árvores de decisão (Breiman, 2001)
  4. LightGBM             - gradient boosting de árvores (Ke et al., 2017)

Protocolo:
  - Holdout TEMPORAL: treino = 2007-11 a 2015-12, teste = 2016-01 a 2017-06.
    (Evita usar o futuro para prever o passado.)
  - Todo o pré-processamento (imputação, codificação, padronização) fica
    DENTRO de um Pipeline do scikit-learn, logo é reajustado em cada dobra
    da validação cruzada -> sem vazamento de informação.
  - Ajuste de hiperparâmetros: busca em grade/aleatória, 3 dobras
    estratificadas, métrica ROC-AUC (amostra estratificada de 40 mil
    linhas do treino para os modelos mais caros).
  - Avaliação de robustez: validação cruzada estratificada de 5 dobras no
    treino completo (média ± desvio-padrão das métricas).
  - Limiar de decisão: escolhido nas previsões fora-da-dobra (OOF) do
    treino para maximizar o F1 da classe positiva e só então aplicado ao teste.
"""
from __future__ import annotations

import re
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from lightgbm import LGBMClassifier

from . import config as cfg
from .eda import clean, BLUES
from .plot_style import COL_W, FULL_W, savefig
from .utils import save_table

warnings.filterwarnings("ignore")

MODEL_NAMES = ["Regressão Logística", "KNN", "Random Forest", "LightGBM"]
TEST_SHORT = {"Limiar": "Lim.", "Acurácia": "Acur.", "Precisão": "Prec.", "Revocação": "Rev.",
              "ROC-AUC": "AUC", "PR-AUC": "AP", "Regressão Logística": "Logística"}

DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
DIR_ANGLE = {d: i * 22.5 for i, d in enumerate(DIRS)}

BASE_NUM = ["MinTemp", "MaxTemp", "Sunshine", "WindGustSpeed", "WindSpeed9am",
            "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
            "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"]


# ==========================================================================
# 1. Engenharia de atributos (linha a linha; não usa o alvo)
# ==========================================================================
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as variáveis explicativas. Só usa informação disponível até o
    dia t (inclusive), portanto pode ser aplicada antes da divisão
    treino/teste sem vazamento."""
    d = df.sort_values(["Location", "Date"]).copy()
    X = pd.DataFrame(index=d.index)
    for c in BASE_NUM:
        X[c] = d[c]

    # Transformação log para variáveis muito assimétricas (outliers reais)
    X["log_Rainfall"] = np.log1p(d["Rainfall"])
    X["log_Evaporation"] = np.log1p(d["Evaporation"])
    X["RainToday"] = d["RainToday"].map({"Yes": 1.0, "No": 0.0})

    # Amplitudes / variações intradiárias (dinâmica do dia)
    X["TempRange"] = d["MaxTemp"] - d["MinTemp"]
    X["dTemp_9a15"] = d["Temp3pm"] - d["Temp9am"]
    X["dPressure_9a15"] = d["Pressure3pm"] - d["Pressure9am"]
    X["dHumidity_9a15"] = d["Humidity3pm"] - d["Humidity9am"]
    X["dCloud_9a15"] = d["Cloud3pm"] - d["Cloud9am"]

    # Variações em relação ao dia anterior na mesma estação (tendência barométrica)
    g = d.groupby("Location")
    consecutive = (d["Date"] - g["Date"].shift(1)).dt.days.eq(1)
    for c, new in [("Pressure3pm", "dPressure_24h"), ("Humidity3pm", "dHumidity_24h"),
                   ("Temp3pm", "dTemp_24h")]:
        X[new] = (d[c] - g[c].shift(1)).where(consecutive)
    X["RainYesterday"] = g["RainToday"].shift(1).map({"Yes": 1.0, "No": 0.0}).where(consecutive)

    # Direções do vento: variável circular -> (seno, cosseno); ausente -> (0, 0)
    for c in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        ang = np.deg2rad(d[c].map(DIR_ANGLE))
        X[f"{c}_sin"] = np.sin(ang).fillna(0.0)
        X[f"{c}_cos"] = np.cos(ang).fillna(0.0)

    # Sazonalidade (codificação cíclica do dia do ano)
    doy = d["Date"].dt.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Indicadores de ausência (ausência estrutural, informativa)
    for c in ["Sunshine", "Evaporation", "Cloud9am", "Cloud3pm", "Pressure3pm", "WindGustSpeed"]:
        X[f"miss_{c}"] = d[c].isna().astype(float)

    X["Location"] = d["Location"]
    return X.loc[df.index]


def prepare(df: pd.DataFrame):
    """Limpeza, remoção de alvo ausente, engenharia de atributos e divisão temporal."""
    d = clean(df)
    X_all = engineer(d)
    mask = d["RainTomorrow"].notna()
    X_all, d = X_all[mask], d[mask]
    y_all = (d["RainTomorrow"] == "Yes").astype(int)
    is_test = d["Date"] >= pd.Timestamp(cfg.CLF_TEST_START)
    Xtr, Xte = X_all[~is_test], X_all[is_test]
    ytr, yte = y_all[~is_test], y_all[is_test]
    info = {"n_total": int(mask.sum()), "n_removidas_alvo": int((~mask).sum()),
            "n_treino": len(Xtr), "n_teste": len(Xte), "n_features": X_all.shape[1],
            "pos_treino": float(ytr.mean()), "pos_teste": float(yte.mean())}
    return Xtr, Xte, ytr, yte, info


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    flags = [c for c in X.columns if c.startswith("miss_")]
    num = [c for c in X.columns if c not in flags + ["Location"]]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("loc", Pipeline([("te", TargetEncoder(target_type="binary", cv=5,
                                               random_state=cfg.SEED)),
                          ("sc", StandardScaler())]), ["Location"]),
        ("flag", "passthrough", flags),
    ], verbose_feature_names_out=False)


def feature_names(pre: ColumnTransformer) -> list[str]:
    names = list(pre.get_feature_names_out())
    return ["Location (target enc.)" if n == "Location" else n for n in names]


# ==========================================================================
# 2. Modelos e espaços de busca
# ==========================================================================
def model_specs():
    """(estimador, espaço de busca, tipo de busca, usa subamostra?)"""
    return {
        "Regressão Logística": (
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]}, "grid", False),
        "KNN": (
            KNeighborsClassifier(algorithm="brute", n_jobs=cfg.N_JOBS),
            {"clf__n_neighbors": [15, 31, 61, 121, 201],
             "clf__weights": ["uniform", "distance"]}, "grid", True),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   n_jobs=cfg.N_JOBS, random_state=cfg.SEED),
            {"clf__max_depth": [12, 20, None], "clf__min_samples_leaf": [1, 5, 15],
             "clf__max_features": ["sqrt", 0.4]}, "random", True),
        "LightGBM": (
            LGBMClassifier(n_estimators=600, class_weight="balanced", subsample=0.8,
                           subsample_freq=1, random_state=cfg.SEED, n_jobs=cfg.N_JOBS,
                           verbose=-1),
            {"clf__num_leaves": [15, 31, 63, 127], "clf__learning_rate": [0.02, 0.05, 0.1],
             "clf__min_child_samples": [20, 50, 100], "clf__colsample_bytree": [0.6, 0.8, 1.0],
             "clf__reg_lambda": [0.0, 1.0, 5.0], "clf__n_estimators": [300, 600, 1000]},
            "random", True),
    }


def tune(Xtr, ytr, verbose=True, n_sub=40000):
    """Busca de hiperparâmetros (3 dobras estratificadas, ROC-AUC)."""
    Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=n_sub, stratify=ytr,
                                    random_state=cfg.SEED)
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.SEED)
    best, rows = {}, []
    for name, (est, space, kind, sub) in model_specs().items():
        t0 = time.time()
        pipe = Pipeline([("pre", make_preprocessor(Xtr)), ("clf", est)])
        if kind == "grid":
            search = GridSearchCV(pipe, space, scoring="roc_auc", cv=cv3, n_jobs=1)
        else:
            search = RandomizedSearchCV(pipe, space, n_iter=12 if name == "LightGBM" else 8,
                                        scoring="roc_auc", cv=cv3, n_jobs=1,
                                        random_state=cfg.SEED)
        Xf, yf = (Xs, ys) if sub else (Xtr, ytr)
        search.fit(Xf, yf)
        best[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        rows.append({"Modelo": name, "Melhores hiperparâmetros": _fmt_params(best[name]),
                     "AUC (busca)": search.best_score_,
                     "Amostra": f"{len(Xf) // 1000} mil"})
        if verbose:
            print(f"[{name}] {best[name]}  AUC={search.best_score_:.4f}  ({time.time() - t0:.0f}s)")
    tab = pd.DataFrame(rows).set_index("Modelo")
    save_table(tab, "tab_clf_hiperparametros", float_fmt="{:.4f}")
    return best, tab


def _fmt_params(p: dict) -> str:
    short = {"n_neighbors": "k", "weights": "pesos", "max_depth": "prof.",
             "min_samples_leaf": "min_folha", "max_features": "max_feat",
             "num_leaves": "folhas", "learning_rate": "eta", "min_child_samples": "min_filhos",
             "colsample_bytree": "colsample", "reg_lambda": "lambda", "n_estimators": "árvores"}
    return ", ".join(f"{short.get(k, k)}={v}" for k, v in p.items())


def build_pipeline(name: str, params: dict, Xtr) -> Pipeline:
    est = clone(model_specs()[name][0]).set_params(**params)
    return Pipeline([("pre", make_preprocessor(Xtr)), ("clf", est)])


# ==========================================================================
# 3. Validação cruzada (5 dobras) + previsões fora-da-dobra
# ==========================================================================
def _scores(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    return {"Acurácia": accuracy_score(y, yhat),
            "Acurácia bal.": balanced_accuracy_score(y, yhat),
            "Precisão": precision_score(y, yhat, zero_division=0),
            "Revocação": recall_score(y, yhat),
            "F1": f1_score(y, yhat),
            "ROC-AUC": roc_auc_score(y, p),
            "PR-AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p)}


def cross_validate_models(best: dict, Xtr, ytr, verbose=True):
    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.SEED)
    oof = {n: np.zeros(len(ytr)) for n in MODEL_NAMES}
    recs = []
    for name in MODEL_NAMES:
        t0 = time.time()
        for k, (i_tr, i_va) in enumerate(cv.split(Xtr, ytr)):
            pipe = build_pipeline(name, best[name], Xtr)
            pipe.fit(Xtr.iloc[i_tr], ytr.iloc[i_tr])
            p = pipe.predict_proba(Xtr.iloc[i_va])[:, 1]
            oof[name][i_va] = p
            recs.append({"Modelo": name, "Dobra": k + 1, **_scores(ytr.iloc[i_va].values, p)})
        if verbose:
            print(f"[CV] {name}: {time.time() - t0:.0f}s")
    long = pd.DataFrame(recs)
    agg = long.groupby("Modelo").agg(["mean", "std"]).drop(columns="Dobra")
    tab = pd.DataFrame(index=MODEL_NAMES)
    for m in ["ROC-AUC", "PR-AUC", "F1", "Revocação", "Precisão", "Acurácia bal."]:
        tab[m] = [f"{agg.loc[n, (m, 'mean')]:.3f} ± {agg.loc[n, (m, 'std')]:.3f}"
                  for n in MODEL_NAMES]
    tab.to_csv(cfg.TAB_DIR / "tab_clf_cv.csv")
    (cfg.TAB_DIR / "tab_clf_cv.tex").write_text(
        re.sub(r"(\d)\.(\d)", r"\1,\2", tab.to_latex(column_format="l" + "c" * tab.shape[1])
        .replace("±", r"$\pm$")), encoding="utf-8")
    long.to_csv(cfg.TAB_DIR / "clf_cv_dobras.csv", index=False)
    return tab, long, oof


def best_thresholds(ytr, oof: dict) -> dict:
    """Limiar que maximiza o F1 nas previsões fora-da-dobra do treino."""
    thr = {}
    for name, p in oof.items():
        prec, rec, t = precision_recall_curve(ytr, p)
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        thr[name] = float(t[np.nanargmax(f1[:-1])])
    return thr


# ==========================================================================
# 4. Avaliação no teste (holdout temporal 2016-2017)
# ==========================================================================
def bootstrap_auc(y, p, n=500, seed=cfg.SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        vals.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


def evaluate_test(best, thr, Xtr, ytr, Xte, yte, verbose=True):
    fitted, probs, rows, rows05 = {}, {}, {}, {}
    for name in MODEL_NAMES:
        t0 = time.time()
        pipe = build_pipeline(name, best[name], Xtr).fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        fitted[name], probs[name] = pipe, p
        sc = _scores(yte.values, p, thr[name])
        lo, hi = bootstrap_auc(yte.values, p)
        rows[name] = {"Limiar": thr[name], **sc, "AUC IC95 inf": lo, "AUC IC95 sup": hi}
        rows05[name] = _scores(yte.values, p, 0.5)
        if verbose:
            print(f"[Teste] {name}: AUC={sc['ROC-AUC']:.4f} F1={sc['F1']:.4f} ({time.time() - t0:.0f}s)")
    tab = pd.DataFrame(rows).T
    tab.to_csv(cfg.TAB_DIR / "tab_clf_teste_completa.csv")
    pd.DataFrame(rows05).T.to_csv(cfg.TAB_DIR / "tab_clf_teste_limiar05.csv")
    show = tab[["Limiar", "Acurácia", "Precisão", "Revocação", "F1", "ROC-AUC", "PR-AUC", "Brier"]]
    save_table(show, "tab_clf_teste", float_fmt="{:.3f}",
               bold_best={"Acurácia": "max", "F1": "max", "ROC-AUC": "max", "PR-AUC": "max",
                          "Brier": "min", "Precisão": "max", "Revocação": "max"},
               latex_rename=TEST_SHORT)
    return tab, fitted, probs


# ==========================================================================
# 5. Figuras
# ==========================================================================
def plot_cv(long: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9))
    for ax, m in zip(axes, ["ROC-AUC", "F1"]):
        for i, name in enumerate(MODEL_NAMES):
            v = long.loc[long["Modelo"] == name, m].values
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=10,
                       color=cfg.CLF_COLORS[name], zorder=3)
            ax.hlines(v.mean(), i - 0.25, i + 0.25, color=cfg.INK, linewidth=1)
        ax.set_xticks(range(len(MODEL_NAMES)))
        ax.set_xticklabels(["Log.", "KNN", "RF", "LGBM"])
        ax.set_title(f"{m} por dobra" + (" (limiar 0,5)" if m == "F1" else ""))
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, "clf_cv")


def plot_roc_pr(yte, probs: dict, thr: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.8))
    ax = axes[0]
    for name in MODEL_NAMES:
        fpr, tpr, t = roc_curve(yte, probs[name])
        auc = roc_auc_score(yte, probs[name])
        ax.plot(fpr, tpr, color=cfg.CLF_COLORS[name], linewidth=1.3,
                label=f"{name} (AUC = {auc:.3f})")
        j = np.argmin(np.abs(t - thr[name]))
        ax.scatter(fpr[j], tpr[j], s=18, color=cfg.CLF_COLORS[name], edgecolor="white",
                   linewidth=0.8, zorder=4)
    ax.plot([0, 1], [0, 1], color=cfg.INK_2, linestyle=":", linewidth=0.8, label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos (1 - especificidade)")
    ax.set_ylabel("Taxa de verdadeiros positivos (revocação)")
    ax.set_title("(a) Curvas ROC no teste", loc="left")
    ax.legend(loc="lower right", fontsize=6, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9, borderaxespad=0.2)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    ax = axes[1]
    for name in MODEL_NAMES:
        prec, rec, t = precision_recall_curve(yte, probs[name])
        ap = average_precision_score(yte, probs[name])
        ax.plot(rec, prec, color=cfg.CLF_COLORS[name], linewidth=1.3,
                label=f"{name} (AP = {ap:.3f})")
        j = np.argmin(np.abs(t - thr[name]))
        ax.scatter(rec[j], prec[j], s=18, color=cfg.CLF_COLORS[name], edgecolor="white",
                   linewidth=0.8, zorder=4)
    ax.axhline(np.mean(yte), color=cfg.INK_2, linestyle=":", linewidth=0.8,
               label=f"Prevalência ({np.mean(yte):.2f})")
    ax.set_xlabel("Revocação")
    ax.set_ylabel("Precisão")
    ax.set_title("(b) Curvas precisão-revocação no teste", loc="left")
    ax.legend(loc="lower left", fontsize=6, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9, borderaxespad=0.2)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    savefig(fig, "clf_roc_pr")


def plot_confusion(yte, p, thr: float, name: str) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.95))
    out = {}
    for ax, (t, lab) in zip(axes, [(0.5, "limiar = 0,50"),
                                    (thr, "limiar ótimo = " + f"{thr:.2f}".replace(".", ","))]):
        cm = confusion_matrix(yte, (p >= t).astype(int))
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap=BLUES, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}".replace(",", ".") + f"\n({100 * pct[i, j]:.1f}%)",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if pct[i, j] > 0.55 else cfg.INK)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Não", "Sim"])
        ax.set_yticklabels(["Não", "Sim"])
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        ax.set_title(lab, fontsize=7.5)
        ax.grid(False)
        out[lab] = cm.ravel()
    fig.tight_layout()
    savefig(fig, "clf_confusao")
    return pd.DataFrame(out, index=["VN", "FP", "FN", "VP"])


def plot_importance(fitted: dict, Xte, yte, best_name: str) -> pd.DataFrame:
    """(a) Importância por permutação (queda de ROC-AUC no teste) do melhor modelo;
    (b) coeficientes da regressão logística (variáveis padronizadas)."""
    pipe = fitted[best_name]
    Xs, _, ys, _ = train_test_split(Xte, yte, train_size=8000, stratify=yte,
                                    random_state=cfg.SEED)
    pi = permutation_importance(pipe, Xs, ys, scoring="roc_auc", n_repeats=5,
                                random_state=cfg.SEED, n_jobs=1)
    imp = pd.DataFrame({"media": pi.importances_mean, "dp": pi.importances_std},
                       index=Xte.columns).sort_values("media", ascending=False)
    imp.to_csv(cfg.TAB_DIR / "clf_importancia_permutacao.csv")

    lr = fitted["Regressão Logística"]
    coef = pd.Series(lr.named_steps["clf"].coef_[0], index=feature_names(lr.named_steps["pre"]))
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index)
    odds = pd.DataFrame({"coef": coef, "odds_ratio": np.exp(coef)})
    odds.to_csv(cfg.TAB_DIR / "clf_logistica_coeficientes.csv")

    top = imp.head(15)[::-1]
    topc = coef.head(15)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.0))
    ax = axes[0]
    ax.barh(top.index, top["media"], xerr=top["dp"], color=cfg.CLF_COLORS[best_name],
            height=0.7, error_kw=dict(lw=0.6, capsize=1.5, ecolor=cfg.INK_2))
    ax.set_xlabel("Queda média de ROC-AUC ao permutar")
    ax.set_title(f"(a) Importância por permutação - {best_name}", loc="left")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    cols = [cfg.C["blue"] if v > 0 else cfg.C["orange"] for v in topc.values]
    ax.barh(topc.index, topc.values, color=cols, height=0.7)
    ax.axvline(0, color=cfg.INK_2, linewidth=0.6)
    ax.set_xlabel("Coeficiente (log-odds por 1 DP)")
    ax.set_title("(b) Reg. logística: maiores coeficientes", loc="left")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cfg.C["blue"], label="Aumenta P(chuva)"),
                       Patch(color=cfg.C["orange"], label="Reduz P(chuva)")],
              loc="lower right")
    fig.tight_layout()
    savefig(fig, "clf_importancia")
    return imp


def plot_calibration(yte, probs: dict) -> None:
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.plot([0, 1], [0, 1], color=cfg.INK_2, linestyle=":", linewidth=0.8,
            label="Calibração perfeita")
    for name in MODEL_NAMES:
        fr, mp = calibration_curve(yte, probs[name], n_bins=10, strategy="quantile")
        ax.plot(mp, fr, marker="o", markersize=3, color=cfg.CLF_COLORS[name],
                linewidth=1.1, label=name)
    ax.set_xlabel("Probabilidade prevista (média no bin)")
    ax.set_ylabel("Frequência observada de chuva")
    ax.legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    savefig(fig, "clf_calibracao")


# ==========================================================================
# 6. Pipeline completo
# ==========================================================================
def run_classification(df: pd.DataFrame, verbose: bool = True) -> dict:
    Xtr, Xte, ytr, yte, info = prepare(df)
    if verbose:
        print(info)
    best, tab_hp = tune(Xtr, ytr, verbose)
    tab_cv, cv_long, oof = cross_validate_models(best, Xtr, ytr, verbose)
    plot_cv(cv_long)
    thr = best_thresholds(ytr, oof)
    tab_test, fitted, probs = evaluate_test(best, thr, Xtr, ytr, Xte, yte, verbose)
    best_name = tab_test["ROC-AUC"].astype(float).idxmax()
    plot_roc_pr(yte, probs, thr)
    cm = plot_confusion(yte, probs[best_name], thr[best_name], best_name)
    imp = plot_importance(fitted, Xte, yte, best_name)
    plot_calibration(yte, probs)
    if verbose:
        print(tab_cv)
        print(tab_test.round(4))
        print(cm)
        print(imp.head(10))
    lr = fitted["Regressão Logística"]
    coef = pd.Series(lr.named_steps["clf"].coef_[0], index=feature_names(lr.named_steps["pre"]))
    return {"info": info, "hiperparametros": best, "limiares": thr,
            "cv": cv_long.groupby("Modelo").mean(numeric_only=True).drop(columns="Dobra")
                         .to_dict(orient="index"),
            "cv_std": cv_long.groupby("Modelo").std(numeric_only=True).drop(columns="Dobra")
                             .to_dict(orient="index"),
            "teste": tab_test.astype(float).to_dict(orient="index"),
            "melhor_modelo": best_name, "confusao": cm.to_dict(),
            "importancia_top10": imp.head(10)["media"].to_dict(),
            "logistica_odds_top": np.exp(coef.reindex(coef.abs().sort_values(ascending=False)
                                                      .index).head(10)).to_dict(),
            "_objects": {"Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte, "fitted": fitted,
                         "probs": probs, "oof": oof}}
