"""
Configurações globais do projeto.

Centraliza caminhos, sementes aleatórias, datas de corte e paleta de cores,
para que o main.py e o notebook usem exatamente os mesmos parâmetros.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
REPORT_DIR = ROOT / "report"

for _d in (DATA_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dados (Kaggle)
# --------------------------------------------------------------------------
KAGGLE_DATASET = "jsphyg/weather-dataset-rattle-package"
KAGGLE_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
KAGGLE_PAGE = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
CSV_NAME = "weatherAUS.csv"
CSV_PATH = DATA_DIR / CSV_NAME

# --------------------------------------------------------------------------
# Reprodutibilidade
# --------------------------------------------------------------------------
SEED = 42
N_JOBS = -1

# --------------------------------------------------------------------------
# Regressão (séries temporais)
# --------------------------------------------------------------------------
REG_LOCATION = "Sydney"          # estação escolhida
REG_TARGET = "MaxTemp"           # temperatura máxima diária (°C)
REG_FREQ = "W-SUN"               # agregação semanal (semanas terminando no domingo)
REG_MIN_DAYS = 4                 # mínimo de dias válidos para a média semanal
SEASON_PERIOD = 52               # período sazonal inteiro (STL)
SEASON_PERIOD_EXACT = 365.25 / 7  # período sazonal exato em semanas (Fourier/Prophet)
TRAIN_END = "2014-12-31"         # fim do treino (ajuste de hiperparâmetros)
VAL_END = "2015-12-31"           # fim da validação; teste = 2016-01 em diante
BACKTEST_ORIGINS = ["2011-12-31", "2012-12-31", "2013-12-31", "2014-12-31"]
BACKTEST_H = 52                  # horizonte (semanas) de cada dobra do backtest

# --------------------------------------------------------------------------
# Classificação
# --------------------------------------------------------------------------
CLF_TARGET = "RainTomorrow"
CLF_TEST_START = "2016-01-01"    # holdout temporal: 2016-01-01 em diante
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Paleta (ordem categórica fixa; validada para daltonismo nos pares adjacentes)
# --------------------------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
OBS = "#3b3b38"                   # série observada
C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
REG_COLORS = {
    "SARIMAX": C["blue"],
    "Prophet": C["orange"],
    "STL": C["aqua"],
    "Combinado": C["violet"],
    "Ingênuo": "#8f8e88",
}
CLF_COLORS = {
    "Regressão Logística": C["blue"],
    "KNN": C["orange"],
    "Random Forest": C["aqua"],
    "LightGBM": C["violet"],
}
