"""
Regressão em séries temporais: previsão da temperatura máxima semanal
média em Sydney.

Modelos (todos supervisionados: aprendem f(passado) -> futuro):
  1. SARIMAX  - regressão harmônica dinâmica: termos de Fourier como
                variáveis exógenas (X) + erros ARMA(p, q).
  2. Prophet  - modelo aditivo bayesiano (tendência por partes +
                sazonalidade de Fourier), Taylor & Letham (2018).
  3. STL      - decomposição STL (Cleveland et al., 1990) + ARIMA na
                série dessazonalizada; a sazonalidade é projetada de
                forma ingênua (repete o último ciclo).
  4. Combinado - média simples das previsões dos três modelos acima.
Baseline de referência (não conta como modelo): sazonal ingênuo, y(t) = y(t-52).

Protocolo:
  treino (2008-2014) -> ajuste de hiperparâmetros pelo RMSE na validação (2015)
  treino+validação (2008-2015) -> previsão de 78 semanas no teste (2016-01 a 2017-06)
  backtest com origem móvel (4 dobras, h = 52) para avaliar a robustez.
"""
from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from . import config as cfg
from .plot_style import COL_W, FULL_W, savefig
from .utils import save_table

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

ORIGIN = pd.Timestamp("2008-01-06")  # origem fixa do tempo para os termos de Fourier


# ==========================================================================
# 1. Construção da série
# ==========================================================================
def build_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
    """Série diária e semanal de MaxTemp para a estação escolhida.

    - Reindexa para o calendário diário completo (expõe os dias faltantes).
    - Média semanal (semana terminando no domingo) exigindo >= 4 dias
      válidos; semanas sem dados suficientes são interpoladas linearmente
      no tempo (são poucas e isoladas).
    - Remove a primeira e a última semana se forem incompletas.
    """
    d = df[df["Location"] == cfg.REG_LOCATION].set_index("Date")[cfg.REG_TARGET]
    d = d.asfreq("D")
    n_missing_days = int(d.isna().sum())
    grp = d.resample(cfg.REG_FREQ)
    w = grp.mean()
    cnt = grp.count()
    w[cnt < cfg.REG_MIN_DAYS] = np.nan
    # primeira e última semana incompletas
    if cnt.iloc[0] < 7:
        w, cnt = w.iloc[1:], cnt.iloc[1:]
    if cnt.iloc[-1] < 7:
        w, cnt = w.iloc[:-1], cnt.iloc[:-1]
    n_missing_weeks = int(w.isna().sum())
    w = w.interpolate(method="time")
    w.name = "MaxTemp_semanal"
    info = {"dias_faltantes": n_missing_days, "semanas_interpoladas": n_missing_weeks,
            "n_semanas": len(w), "inicio": w.index.min(), "fim": w.index.max()}
    return d, w, info


def split_series(y: pd.Series):
    train = y[: cfg.TRAIN_END]
    val = y[(y.index > pd.Timestamp(cfg.TRAIN_END)) & (y.index <= pd.Timestamp(cfg.VAL_END))]
    test = y[y.index > pd.Timestamp(cfg.VAL_END)]
    return train, val, test


# ==========================================================================
# 2. Exploração da série
# ==========================================================================
def plot_series(daily: pd.Series, y: pd.Series) -> None:
    train, val, test = split_series(y)
    fig, ax = plt.subplots(figsize=(FULL_W, 2.2))
    ax.plot(daily.index, daily.values, color="#c9c8c2", linewidth=0.4, label="Diária")
    ax.plot(y.index, y.values, color=cfg.OBS, linewidth=1.0, label="Média semanal")
    ymin, ymax = daily.min() - 1, daily.max() + 1
    for (a, b, lab) in [(val.index[0], val.index[-1], "Validação"),
                        (test.index[0], test.index[-1], "Teste")]:
        ax.axvspan(a, b, color=cfg.C["blue"] if lab == "Teste" else cfg.C["aqua"],
                   alpha=0.08, linewidth=0)
        ax.text(a + (b - a) / 2, ymax - 0.5, lab, ha="center", va="top", fontsize=7,
                color=cfg.INK_2)
    ax.text(train.index[len(train) // 2], ymax - 0.5, "Treino", ha="center", va="top",
            fontsize=7, color=cfg.INK_2)
    ax.set_ylim(ymin, ymax + 1.5)
    ax.set_ylabel("MaxTemp (°C)")
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    savefig(fig, "reg_serie")


def stationarity_tests(y: pd.Series) -> pd.DataFrame:
    """ADF (H0: raiz unitária) e KPSS (H0: estacionária) na série e na
    série dessazonalizada por STL."""
    stl = STL(y, period=cfg.SEASON_PERIOD, robust=True).fit()
    rows = {}
    for name, s in [("Série original", y), ("Dessazonalizada (STL)", y - stl.seasonal)]:
        adf = adfuller(s, autolag="AIC")
        kp = kpss(s, regression="c", nlags="auto")
        rows[name] = {"ADF estat.": adf[0], "ADF p-valor": adf[1],
                      "KPSS estat.": kp[0], "KPSS p-valor": kp[1]}
    tab = pd.DataFrame(rows).T
    save_table(tab, "tab_estacionariedade", float_fmt="{:.3f}")
    return tab


def plot_decomposition(y: pd.Series) -> dict:
    """Decomposição STL e funções de autocorrelação (ACF/PACF)."""
    stl = STL(y, period=cfg.SEASON_PERIOD, robust=True).fit()
    var = np.var(y)
    # força da sazonalidade e da tendência (Wang, Smith & Hyndman, 2006)
    f_season = max(0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))
    f_trend = max(0, 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid))

    fig, axes = plt.subplots(4, 1, figsize=(COL_W, 4.0), sharex=True)
    comps = [("Observada", y), ("Tendência", stl.trend), ("Sazonal", stl.seasonal),
             ("Resíduo", stl.resid)]
    for ax, (lab, s) in zip(axes, comps):
        if lab == "Resíduo":
            ax.scatter(s.index, s.values, s=1.5, color=cfg.OBS)
            ax.axhline(0, color=cfg.INK_2, linewidth=0.5)
        else:
            ax.plot(s.index, s.values, color=cfg.OBS if lab == "Observada" else cfg.C["blue"],
                    linewidth=0.8)
        ax.set_ylabel(lab, fontsize=7)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.3)
    savefig(fig, "reg_stl")

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.7), sharey=True)
    nl = 110
    for ax, (lab, fun) in zip(axes, [("ACF", acf), ("PACF", pacf)]):
        vals = fun(y, nlags=nl) if lab == "ACF" else fun(y, nlags=nl, method="ywm")
        ax.vlines(range(len(vals)), 0, vals, color=cfg.C["blue"], linewidth=0.7)
        ci = 1.96 / np.sqrt(len(y))
        ax.axhspan(-ci, ci, color=cfg.C["blue"], alpha=0.12, linewidth=0)
        ax.axhline(0, color=cfg.INK_2, linewidth=0.5)
        ax.set_title(lab)
        ax.set_xlabel("Defasagem (semanas)")
        ax.set_xticks([0, 26, 52, 78, 104])
    fig.tight_layout()
    savefig(fig, "reg_acf")
    return {"forca_sazonal": f_season, "forca_tendencia": f_trend}


# ==========================================================================
# 3. Modelos
# ==========================================================================
def fourier_terms(index: pd.DatetimeIndex, K: int) -> pd.DataFrame:
    """Termos de Fourier sin(2πkt/P), cos(2πkt/P), k = 1..K, com P = 52,18 semanas."""
    t = ((index - ORIGIN).days / 7.0).values
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / cfg.SEASON_PERIOD_EXACT)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / cfg.SEASON_PERIOD_EXACT)
    return pd.DataFrame(cols, index=index)


def future_index(y: pd.Series, h: int) -> pd.DatetimeIndex:
    return pd.date_range(y.index[-1] + pd.Timedelta(weeks=1), periods=h, freq=cfg.REG_FREQ)


def _frame(idx, mean, lower, upper) -> pd.DataFrame:
    return pd.DataFrame({"yhat": np.asarray(mean), "lower": np.asarray(lower),
                         "upper": np.asarray(upper)}, index=idx)


@dataclass
class SarimaxModel:
    """Regressão harmônica dinâmica (SARIMAX com Fourier exógeno):

        y_t = β0 [+ β1 t] + Σ_k (a_k sin(2πkt/P) + b_k cos(2πkt/P)) + η_t
        φ(B) η_t = θ(B) ε_t ,   ε_t ~ N(0, σ²)
    """
    order: tuple = (1, 0, 0)
    K: int = 2
    trend: str = "c"
    name: str = "SARIMAX"
    res_: object = field(default=None, repr=False)
    y_: pd.Series = field(default=None, repr=False)

    def fit(self, y: pd.Series):
        self.y_ = y
        self.res_ = SARIMAX(y, exog=fourier_terms(y.index, self.K), order=self.order,
                            trend=self.trend, enforce_stationarity=True,
                            enforce_invertibility=True).fit(disp=False)
        return self

    def forecast(self, h: int, alpha: float = 0.05) -> pd.DataFrame:
        idx = future_index(self.y_, h)
        fc = self.res_.get_forecast(h, exog=fourier_terms(idx, self.K))
        ci = fc.conf_int(alpha=alpha)
        return _frame(idx, fc.predicted_mean, ci.iloc[:, 0], ci.iloc[:, 1])

    def one_step(self, y_new: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
        """Previsões 1 passo à frente em y_new: parâmetros fixos, estado do
        filtro de Kalman atualizado a cada nova observação."""
        ext = self.res_.append(y_new, exog=fourier_terms(y_new.index, self.K), refit=False)
        p = ext.get_prediction(start=len(self.y_), end=len(self.y_) + len(y_new) - 1)
        ci = p.conf_int(alpha=alpha)
        return _frame(y_new.index, p.predicted_mean, ci.iloc[:, 0], ci.iloc[:, 1])

    def fitted(self) -> pd.Series:
        return self.res_.fittedvalues

    @property
    def aic(self):
        return self.res_.aic

    def describe(self):
        return f"ARMA({self.order[0]},{self.order[2]}) + Fourier K={self.K}, tendência={self.trend}"


@dataclass
class ProphetModel:
    """y(t) = g(t) + s(t) + ε_t ;  g: tendência linear por partes; s: Fourier anual."""
    fourier_order: int = 5
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    name: str = "Prophet"
    m_: object = field(default=None, repr=False)
    y_: pd.Series = field(default=None, repr=False)

    def _new(self):
        from prophet import Prophet
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                    daily_seasonality=False, interval_width=0.95,
                    changepoint_prior_scale=self.changepoint_prior_scale,
                    seasonality_prior_scale=self.seasonality_prior_scale,
                    uncertainty_samples=500)
        m.add_seasonality("anual", period=365.25, fourier_order=self.fourier_order)
        return m

    def fit(self, y: pd.Series):
        self.y_ = y
        self.m_ = self._new()
        np.random.seed(cfg.SEED)
        self.m_.fit(pd.DataFrame({"ds": y.index, "y": y.values}))
        return self

    def forecast(self, h: int, alpha: float = 0.05) -> pd.DataFrame:
        idx = future_index(self.y_, h)
        np.random.seed(cfg.SEED)
        p = self.m_.predict(pd.DataFrame({"ds": idx}))
        return _frame(idx, p["yhat"], p["yhat_lower"], p["yhat_upper"])

    def one_step(self, y_new: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
        """O Prophet não tem estado dinâmico: para usar a informação nova é
        preciso reajustar. Reajustamos a cada semana (janela expansível)."""
        base_y, base_m = self.y_, self.m_
        rows = []
        hist = base_y.copy()
        for t, val in y_new.items():
            mdl = ProphetModel(self.fourier_order, self.changepoint_prior_scale,
                               self.seasonality_prior_scale).fit(hist)
            rows.append(mdl.forecast(1, alpha))
            hist = pd.concat([hist, pd.Series([val], index=[t])])
        self.y_, self.m_ = base_y, base_m
        out = pd.concat(rows)
        out.index = y_new.index
        return out

    def fitted(self) -> pd.Series:
        p = self.m_.predict(pd.DataFrame({"ds": self.y_.index}))
        return pd.Series(p["yhat"].values, index=self.y_.index)

    def describe(self):
        return (f"Fourier anual N={self.fourier_order}, "
                f"tau={self.changepoint_prior_scale}, sigma_s={self.seasonality_prior_scale}")


@dataclass
class STLModel:
    """Decomposição + ARIMA:

        y_t = T_t + S_t + R_t                (STL robusta, período 52)
        Ŝ(c) = suavização circular da média de S_t por posição c = t mod 52
        a_t = y_t - Ŝ(t mod 52)              (série dessazonalizada)
        a_t ~ ARIMA(p, d, q)                 (tendência + componente irregular)
        ŷ_{t+h} = â_{t+h} + Ŝ((t+h) mod 52)
    """
    seasonal: int = 13
    smooth: int = 1
    order: tuple = (1, 0, 0)
    trend: str = "c"
    name: str = "STL"
    res_: object = field(default=None, repr=False)
    y_: pd.Series = field(default=None, repr=False)
    profile_: np.ndarray = field(default=None, repr=False)
    stl_: object = field(default=None, repr=False)

    def _pos(self, index):
        return (((index - ORIGIN).days // 7).values % cfg.SEASON_PERIOD).astype(int)

    def _season(self, index):
        return self.profile_[self._pos(index)]

    def fit(self, y: pd.Series):
        self.y_ = y
        self.stl_ = STL(y, period=cfg.SEASON_PERIOD, seasonal=self.seasonal, robust=True).fit()
        pos = self._pos(y.index)
        prof = pd.Series(self.stl_.seasonal.values).groupby(pos).mean()
        prof = prof.reindex(range(cfg.SEASON_PERIOD)).interpolate().values
        if self.smooth > 1:  # média móvel circular entre semanas vizinhas
            k = self.smooth // 2
            ext = np.concatenate([prof[-k:], prof, prof[:k]])
            prof = np.convolve(ext, np.ones(self.smooth) / self.smooth, mode="valid")
        self.profile_ = prof - prof.mean()
        sa = y - self._season(y.index)
        self.res_ = ARIMA(sa, order=self.order, trend=self.trend).fit()
        return self

    def forecast(self, h: int, alpha: float = 0.05) -> pd.DataFrame:
        idx = future_index(self.y_, h)
        fc = self.res_.get_forecast(h)
        ci = fc.conf_int(alpha=alpha)
        s = self._season(idx)
        return _frame(idx, fc.predicted_mean.values + s, ci.iloc[:, 0].values + s,
                      ci.iloc[:, 1].values + s)

    def one_step(self, y_new: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
        s = self._season(y_new.index)
        ext = self.res_.append(y_new - s, refit=False)
        p = ext.get_prediction(start=len(self.y_), end=len(self.y_) + len(y_new) - 1)
        ci = p.conf_int(alpha=alpha)
        return _frame(y_new.index, p.predicted_mean.values + s, ci.iloc[:, 0].values + s,
                      ci.iloc[:, 1].values + s)

    def fitted(self) -> pd.Series:
        return self.res_.fittedvalues + self._season(self.y_.index)

    def describe(self):
        return (f"STL(janela={self.seasonal}), suav.={self.smooth} + "
                f"ARIMA({self.order[0]},{self.order[1]},{self.order[2]}), tendência={self.trend}")


@dataclass
class CombinedModel:
    """Média simples:  ŷ_t = (ŷ_SARIMAX + ŷ_Prophet + ŷ_STL) / 3.

    Intervalo: média dos limites dos três modelos (aproximação que supõe
    erros fortemente correlacionados entre os modelos - razoável, pois
    todos erram juntos diante de anomalias climáticas)."""
    members: list
    name: str = "Combinado"

    def fit(self, y: pd.Series):
        for m in self.members:
            m.fit(y)
        return self

    def forecast(self, h: int, alpha: float = 0.05) -> pd.DataFrame:
        return sum(m.forecast(h, alpha) for m in self.members) / len(self.members)

    def one_step(self, y_new: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
        return sum(m.one_step(y_new, alpha) for m in self.members) / len(self.members)

    def fitted(self) -> pd.Series:
        return sum(m.fitted() for m in self.members) / len(self.members)

    def describe(self):
        return "Média simples de SARIMAX, Prophet e STL"


@dataclass
class SeasonalNaive:
    """Baseline: longo prazo y(t+h) = y(t+h-52); um passo: persistência y(t+1) = y(t)."""
    name: str = "Ingênuo"
    y_: pd.Series = field(default=None, repr=False)

    def fit(self, y):
        self.y_ = y
        return self

    def forecast(self, h, alpha=0.05):
        idx = future_index(self.y_, h)
        last = self.y_.iloc[-cfg.SEASON_PERIOD:].values
        vals = np.array([last[i % cfg.SEASON_PERIOD] for i in range(h)])
        resid = (self.y_ - self.y_.shift(cfg.SEASON_PERIOD)).dropna()
        z = stats.norm.ppf(1 - alpha / 2)
        se = resid.std() * np.sqrt(np.floor(np.arange(h) / cfg.SEASON_PERIOD) + 1)
        return _frame(idx, vals, vals - z * se, vals + z * se)

    def one_step(self, y_new, alpha=0.05):
        full = pd.concat([self.y_, y_new])
        pred = full.shift(1).loc[y_new.index]
        sd = self.y_.diff().std()
        z = stats.norm.ppf(1 - alpha / 2)
        return _frame(y_new.index, pred, pred - z * sd, pred + z * sd)

    def fitted(self):
        return self.y_.shift(cfg.SEASON_PERIOD)

    def describe(self):
        return "y(t+h)=y(t+h-52); 1 passo: y(t+1)=y(t)"

# ==========================================================================
# 4. Métricas
# ==========================================================================
def metrics(y_true: pd.Series, fc: pd.DataFrame) -> dict:
    y, p = y_true.values, fc["yhat"].values
    cover = np.mean((y >= fc["lower"].values) & (y <= fc["upper"].values)) * 100
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)),
            "MAPE (%)": float(np.mean(np.abs((y - p) / y)) * 100),
            "R2": float(r2_score(y, p)),
            "Cobertura IC95 (%)": float(cover)}


def _rmse(y, fc):
    return float(np.sqrt(mean_squared_error(y.values, fc["yhat"].values)))


# ==========================================================================
# 5. Ajuste de hiperparâmetros
# ==========================================================================
def tune_sarimax(train, val, verbose=True):
    """Grade em (K, p, q, tendência). Critério: AIC no treino.

    O AIC = -2 log L + 2k é calculado a partir da verossimilhança dos erros
    de previsão 1 passo à frente (inovações do filtro de Kalman); penaliza
    a complexidade e favorece modelos cujos resíduos são ruído branco,
    que é a premissa do modelo. O RMSE na validação é registrado para
    comparação."""
    rows = []
    for K, p, q, tr in itertools.product([1, 2, 3, 4, 6], [0, 1, 2], [0, 1, 2], ["c", "ct"]):
        try:
            m = SarimaxModel(order=(p, 0, q), K=K, trend=tr).fit(train)
            rows.append({"K": K, "p": p, "q": q, "tendência": tr, "AIC": m.aic,
                         "RMSE val": _rmse(val, m.forecast(len(val)))})
        except Exception:
            continue
    tab = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    b = tab.iloc[0]
    best = dict(order=(int(b.p), 0, int(b.q)), K=int(b.K), trend=b["tendência"])
    if verbose:
        print(f"[SARIMAX] melhor (AIC): {best}  AIC = {b['AIC']:.1f}  RMSE val = {b['RMSE val']:.3f}")
    return best, tab


def tune_prophet(train, val, verbose=True):
    """Grade em (N de Fourier, τ, σ_s). Critério: RMSE na validação (2015)."""
    rows = []
    for N, cps, sps in itertools.product([3, 5, 10], [0.001, 0.01, 0.1, 0.5], [1.0, 10.0]):
        m = ProphetModel(fourier_order=N, changepoint_prior_scale=cps,
                         seasonality_prior_scale=sps).fit(train)
        rows.append({"N Fourier": N, "tau": cps, "sigma_s": sps,
                     "RMSE val": _rmse(val, m.forecast(len(val)))})
    tab = pd.DataFrame(rows).sort_values("RMSE val").reset_index(drop=True)
    b = tab.iloc[0]
    best = dict(fourier_order=int(b["N Fourier"]), changepoint_prior_scale=float(b["tau"]),
                seasonality_prior_scale=float(b["sigma_s"]))
    if verbose:
        print(f"[Prophet] melhor: {best}  RMSE val = {b['RMSE val']:.3f}")
    return best, tab


def tune_stl(train, val, verbose=True):
    """Grade em (janela sazonal, suavização do perfil, ARIMA). Critério: RMSE na validação."""
    rows = []
    specs = [((1, 0, 0), "c"), ((2, 0, 0), "c"), ((1, 0, 1), "c"), ((1, 0, 0), "ct"),
             ((0, 1, 1), "n"), ((1, 1, 1), "n")]
    for s, w, (order, tr) in itertools.product([7, 13, 25, 53], [1, 3, 5, 9], specs):
        try:
            m = STLModel(seasonal=s, smooth=w, order=order, trend=tr).fit(train)
            rows.append({"janela sazonal": s, "suavização": w, "ARIMA": str(order),
                         "tendência": tr, "RMSE val": _rmse(val, m.forecast(len(val)))})
        except Exception:
            continue
    tab = pd.DataFrame(rows).sort_values("RMSE val").reset_index(drop=True)
    b = tab.iloc[0]
    best = dict(seasonal=int(b["janela sazonal"]), smooth=int(b["suavização"]),
                order=tuple(int(v) for v in b["ARIMA"].strip("()").split(",")),
                trend=b["tendência"])
    if verbose:
        print(f"[STL] melhor: {best}  RMSE val = {b['RMSE val']:.3f}")
    return best, tab


MODEL_NAMES = ["SARIMAX", "Prophet", "STL", "Combinado"]


def make_models(best: dict) -> dict:
    """Instancia os 4 modelos (+ baseline ingênuo) com os hiperparâmetros escolhidos."""
    def fresh():
        return [SarimaxModel(**best["SARIMAX"]), ProphetModel(**best["Prophet"]),
                STLModel(**best["STL"])]
    s, p, t = fresh()
    return {"SARIMAX": s, "Prophet": p, "STL": t,
            "Combinado": CombinedModel(members=fresh()),
            "Ingênuo": SeasonalNaive()}


# ==========================================================================
# 6. Avaliação no teste e backtest
# ==========================================================================
def evaluate_test(models: dict, train_full: pd.Series, test: pd.Series,
                  one_step: bool = True):
    """Ajusta em treino+validação e avalia no teste de duas formas:

    (A) Longo prazo: previsão única de h = 1, ..., 78 semanas a partir do fim
        de 2015 (o modelo não vê nenhuma observação de teste).
    (B) Um passo à frente: a cada semana do teste, prevê a semana seguinte
        usando todas as observações disponíveis até ali.
    """
    fc_long, fc_one, rows_l, rows_o = {}, {}, {}, {}
    for name, m in models.items():
        m.fit(train_full)
        fc = m.forecast(len(test))
        fc.index = test.index
        fc_long[name] = fc
        rows_l[name] = metrics(test, fc)
        if one_step:
            fo = m.one_step(test)
            fc_one[name] = fo
            rows_o[name] = metrics(test, fo)
    bold = {"RMSE": "min", "MAE": "min", "MAPE (%)": "min", "R2": "max"}
    tab_l = pd.DataFrame(rows_l).T.rename(index={"Ingênuo": "Ingênuo sazonal"})
    save_table(tab_l, "tab_reg_teste", float_fmt="{:.2f}", bold_best=bold)
    tab_o = None
    if one_step:
        tab_o = pd.DataFrame(rows_o).T.rename(index={"Ingênuo": "Persistência"})
        save_table(tab_o, "tab_reg_um_passo", float_fmt="{:.2f}", bold_best=bold)
    return tab_l, tab_o, fc_long, fc_one


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Teste de Diebold-Mariano (perda quadrática) com correção de Harvey,
    Leybourne & Newbold (1997). H0: mesma acurácia. DM < 0 -> modelo 1 melhor."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = dbar / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)


def dm_table(test: pd.Series, fc_one: dict, ref: str = "SARIMAX") -> pd.DataFrame:
    """DM do modelo de referência contra cada um dos outros (erros 1 passo à frente)."""
    e_ref = (test - fc_one[ref]["yhat"]).values
    rows = {}
    for name in MODEL_NAMES + ["Ingênuo"]:
        if name == ref:
            continue
        e = (test - fc_one[name]["yhat"]).values
        stat, p = diebold_mariano(e_ref, e)
        rows["Persistência" if name == "Ingênuo" else name] = {"DM": stat, "p-valor": p}
    tab = pd.DataFrame(rows).T
    save_table(tab, "tab_reg_dm", float_fmt="{:.3f}")
    return tab


def backtest(best: dict, y: pd.Series):
    """Origem móvel: para cada origem, treina com dados até ela e prevê 52 semanas.
    Usa apenas o período treino+validação (o teste nunca é tocado)."""
    recs = []
    for origin in cfg.BACKTEST_ORIGINS:
        tr = y[: origin]
        te = y[y.index > pd.Timestamp(origin)].iloc[: cfg.BACKTEST_H]
        for name, m in make_models(best).items():
            fc = m.fit(tr).forecast(len(te))
            fc.index = te.index
            recs.append({"origem": origin[:4], "modelo": name, "RMSE": _rmse(te, fc),
                         "MAE": float(mean_absolute_error(te, fc["yhat"]))})
    long = pd.DataFrame(recs)
    long["origem"] = (long["origem"].astype(int) + 1).astype(str)  # ano previsto
    wide = long.pivot(index="modelo", columns="origem", values="RMSE")
    wide.columns = [f"RMSE {c}" for c in wide.columns]
    wide["Média"] = wide.mean(axis=1)
    wide["DP"] = long.groupby("modelo")["RMSE"].std()
    wide = wide.loc[MODEL_NAMES + ["Ingênuo"]].rename(index={"Ingênuo": "Ingênuo sazonal"})
    save_table(wide, "tab_reg_backtest", float_fmt="{:.2f}", bold_best={"Média": "min"},
               latex_rename=BT_SHORT)
    return wide, long


# ==========================================================================
# 7. Figuras de previsão
# ==========================================================================
def plot_forecasts(train_full, test, forecasts: dict) -> None:
    ctx = train_full.iloc[-104:]
    names = ["SARIMAX", "Prophet", "STL", "Combinado"]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 3.6), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        fc = forecasts[name]
        col = cfg.REG_COLORS[name]
        ax.plot(ctx.index, ctx.values, color="#a3a29c", linewidth=0.8, label="Treino")
        ax.plot(test.index, test.values, color=cfg.OBS, linewidth=1.0, label="Observado (teste)")
        ax.fill_between(fc.index, fc["lower"], fc["upper"], color=col, alpha=0.15,
                        linewidth=0, label="IC 95%")
        ax.plot(fc.index, fc["yhat"], color=col, linewidth=1.6, label="Previsão")
        ax.axvline(test.index[0], color=cfg.INK_2, linewidth=0.6, linestyle=":")
        rmse = _rmse(test, fc)
        ax.set_title(f"{name}  (RMSE = {rmse:.2f} °C)", loc="left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes[:, 0]:
        ax.set_ylabel("MaxTemp (°C)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color="#a3a29c", lw=0.8, label="Treino (2014-2015)"),
               Line2D([], [], color=cfg.OBS, lw=1.0, label="Observado (teste)"),
               Line2D([], [], color=cfg.INK_2, lw=1.6, label="Previsão (cor do modelo)"),
               Patch(color=cfg.INK_2, alpha=0.18, label="IC 95%")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    savefig(fig, "reg_previsoes")


def plot_one_step(test, fc_one: dict) -> None:
    """Previsões um passo à frente no teste (painel superior) e erro absoluto
    acumulado (painel inferior), que mostra onde cada modelo ganha/perde."""
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.3), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(test.index, test.values, color=cfg.OBS, linewidth=1.4, label="Observado")
    for name in MODEL_NAMES:
        ax.plot(fc_one[name].index, fc_one[name]["yhat"], color=cfg.REG_COLORS[name],
                linewidth=1.0, label=name, linestyle="--" if name == "Combinado" else "-")
    ax.set_ylabel("MaxTemp (°C)")
    ax.legend(ncol=3, loc="upper center", fontsize=6, bbox_to_anchor=(0.5, 1.33))
    ax = axes[1]
    for name in MODEL_NAMES + ["Ingênuo"]:
        err = (test - fc_one[name]["yhat"]).abs().cumsum()
        lab = "Persistência" if name == "Ingênuo" else name
        col = cfg.REG_COLORS.get(name, "#8f8e88")
        ax.plot(err.index, err.values, color=col, linewidth=1.1, label=lab,
                linestyle="--" if name == "Combinado" else (":" if name == "Ingênuo" else "-"))
    ax.set_ylabel("Erro abs. acum. (°C)")
    ax.legend(fontsize=5.5, loc="upper left", ncol=2)
    ax.set_xlim(test.index[0], test.index[-1])
    fig.autofmt_xdate()
    fig.tight_layout(h_pad=0.4)
    savefig(fig, "reg_um_passo")


# ==========================================================================
# 8. Análise de resíduos
# ==========================================================================
RES_SHORT = {"p t (média=0)": "$p_t$", "p Ljung-Box(10)": "LB(10)", "p Ljung-Box(52)": "LB(52)",
             "p Jarque-Bera": "JB", "p Shapiro-Wilk": "SW", "p ARCH-LM(10)": "ARCH(10)"}
BT_SHORT = {f"RMSE {a}": str(a) for a in range(2000, 2030)}


def residual_tests(resid: pd.Series) -> dict:
    r = resid.dropna()
    lb10 = acorr_ljungbox(r, lags=[10], return_df=True)
    lb52 = acorr_ljungbox(r, lags=[52], return_df=True)
    jb = stats.jarque_bera(r)
    sw = stats.shapiro(r)
    arch = het_arch(r, nlags=10)
    tt = stats.ttest_1samp(r, 0)
    return {"Média": float(r.mean()), "p t (média=0)": float(tt.pvalue),
            "p Ljung-Box(10)": float(lb10["lb_pvalue"].iloc[0]),
            "p Ljung-Box(52)": float(lb52["lb_pvalue"].iloc[0]),
            "p Jarque-Bera": float(jb.pvalue), "p Shapiro-Wilk": float(sw.pvalue),
            "p ARCH-LM(10)": float(arch[1])}


def residual_analysis(models: dict, train_full: pd.Series, test: pd.Series,
                      forecasts: dict, best_name: str) -> pd.DataFrame:
    """Testes nos resíduos dentro da amostra (treino+validação) dos 4 modelos
    e figura de diagnóstico do melhor modelo."""
    rows = {}
    resids = {}
    for name in MODEL_NAMES:
        fit = models[name].fitted()
        r = (train_full - fit).iloc[cfg.SEASON_PERIOD:]  # descarta o 1º ano (burn-in)
        resids[name] = r
        rows[name] = residual_tests(r)
    tab = pd.DataFrame(rows).T
    save_table(tab, "tab_reg_residuos", float_fmt="{:.3f}", latex_rename=RES_SHORT)

    r = resids[best_name]
    err_test = test - forecasts[best_name]["yhat"]
    col = cfg.REG_COLORS[best_name]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 3.6))
    ax = axes[0, 0]
    ax.plot(r.index, r.values, color=col, linewidth=0.6, label="Dentro da amostra")
    ax.plot(err_test.index, err_test.values, color=cfg.C["orange"], linewidth=0.8,
            label="Erro no teste")
    ax.axhline(0, color=cfg.INK_2, linewidth=0.5)
    ax.set_title("(a) Resíduos ao longo do tempo", loc="left")
    ax.set_ylabel("°C")
    ax.legend(loc="lower left", ncol=2)

    ax = axes[0, 1]
    ax.hist(r, bins=35, density=True, color=col, alpha=0.6, edgecolor="white", linewidth=0.3)
    xs = np.linspace(r.min(), r.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), color=cfg.INK, linewidth=1,
            label="Normal ajustada")
    ax.set_title("(b) Histograma", loc="left")
    ax.legend(loc="upper left")
    ax.set_yticks([])

    ax = axes[1, 0]
    (osm, osr), (slope, inter, _) = stats.probplot(r, dist="norm")
    ax.scatter(osm, osr, s=3, color=col)
    ax.plot(osm, slope * np.array(osm) + inter, color=cfg.INK, linewidth=0.8)
    ax.set_title("(c) Gráfico Q-Q normal", loc="left")
    ax.set_xlabel("Quantis teóricos")
    ax.set_ylabel("Quantis amostrais")

    ax = axes[1, 1]
    vals = acf(r, nlags=60)
    ax.vlines(range(len(vals)), 0, vals, color=col, linewidth=0.8)
    ci = 1.96 / np.sqrt(len(r))
    ax.axhspan(-ci, ci, color=col, alpha=0.12, linewidth=0)
    ax.axhline(0, color=cfg.INK_2, linewidth=0.5)
    ax.set_title("(d) ACF dos resíduos", loc="left")
    ax.set_xlabel("Defasagem (semanas)")
    fig.tight_layout()
    savefig(fig, "reg_residuos")

    return tab


# ==========================================================================
# 9. Pipeline completo
# ==========================================================================
def run_regression(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Executa todas as etapas da regressão e devolve um dicionário de resultados."""
    daily, y, info = build_series(df)
    train, val, test = split_series(y)
    train_full = pd.concat([train, val])
    plot_series(daily, y)
    stat = stationarity_tests(train_full)
    strength = plot_decomposition(train_full)

    best_sx, tab_sx = tune_sarimax(train, val, verbose)
    best_pr, tab_pr = tune_prophet(train, val, verbose)
    best_st, tab_st = tune_stl(train, val, verbose)
    best = {"SARIMAX": best_sx, "Prophet": best_pr, "STL": best_st}
    tab_sx.to_csv(cfg.TAB_DIR / "tuning_sarimax.csv", index=False)
    tab_pr.to_csv(cfg.TAB_DIR / "tuning_prophet.csv", index=False)
    tab_st.to_csv(cfg.TAB_DIR / "tuning_stl.csv", index=False)

    # hiperparâmetros escolhidos + RMSE na validação (modelos treinados até 2014)
    val_rows = {}
    for name, m in make_models(best).items():
        fc = m.fit(train).forecast(len(val))
        fc.index = val.index
        val_rows[name] = {"Configuração": m.describe(), "RMSE val": _rmse(val, fc)}
    tab_hp = pd.DataFrame(val_rows).T.rename(index={"Ingênuo": "Ingênuo sazonal"})
    tab_hp["RMSE val"] = tab_hp["RMSE val"].astype(float)
    save_table(tab_hp, "tab_reg_hiperparametros", float_fmt="{:.2f}")

    models = make_models(best)
    tab_long, tab_one, fc_long, fc_one = evaluate_test(models, train_full, test)
    plot_forecasts(train_full, test, fc_long)
    plot_one_step(test, fc_one)
    bt_wide, bt_long = backtest(best, train_full)

    best_name = tab_long.loc[MODEL_NAMES, "RMSE"].idxmin()
    tab_dm = dm_table(test, fc_one, ref=tab_one.loc[MODEL_NAMES, "RMSE"].idxmin())
    tab_res = residual_analysis(models, train_full, test, fc_long, best_name)
    if verbose:
        print("\n== Teste: longo prazo (h = 1..78) ==\n", tab_long.round(3))
        print("\n== Teste: um passo à frente ==\n", tab_one.round(3))
        print("\n== Backtest (h = 52) ==\n", bt_wide.round(3))
        print("\n== Diebold-Mariano (1 passo) ==\n", tab_dm.round(3))
        print("\n== Resíduos ==\n", tab_res.round(3).to_string())
    sx = models["SARIMAX"].res_
    return {"info_serie": info, "estacionariedade": stat.to_dict(), "forca": strength,
            "melhores_hiperparametros": {k: {kk: str(vv) for kk, vv in v.items()}
                                         for k, v in best.items()},
            "validacao": tab_hp["RMSE val"].to_dict(),
            "teste_longo": tab_long.to_dict(orient="index"),
            "teste_um_passo": tab_one.to_dict(orient="index"),
            "backtest": bt_wide.to_dict(orient="index"),
            "residuos": tab_res.to_dict(orient="index"), "melhor_modelo": best_name,
            "diebold_mariano": tab_dm.to_dict(orient="index"),
            "n_treino": len(train), "n_val": len(val), "n_teste": len(test),
            "sarimax_params": sx.params.to_dict(), "sarimax_pvalues": sx.pvalues.to_dict(),
            "_objects": {"y": y, "daily": daily, "train": train, "val": val, "test": test,
                         "models": models, "fc_long": fc_long, "fc_one": fc_one,
                         "tuning": {"SARIMAX": tab_sx, "Prophet": tab_pr, "STL": tab_st}}}
