"""
Aquisição e carga dos dados.

O dataset "Rain in Australia" (Kaggle: jsphyg/weather-dataset-rattle-package)
contém observações meteorológicas diárias de 49 estações do Bureau of
Meteorology (BoM) da Austrália entre 2007 e 2017.

O download usa o endpoint público da API do Kaggle. Se ele falhar (sem
internet, por exemplo), basta baixar o arquivo manualmente em
https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
e salvar `weatherAUS.csv` na pasta `data/`.
"""
from __future__ import annotations

import io
import zipfile
import urllib.request

import pandas as pd

from . import config as cfg


def download_dataset(force: bool = False) -> None:
    """Baixa o CSV do Kaggle para data/weatherAUS.csv (se ainda não existir)."""
    if cfg.CSV_PATH.exists() and not force:
        print(f"[data] Arquivo já existe: {cfg.CSV_PATH.name}")
        return
    print(f"[data] Baixando {cfg.KAGGLE_DATASET} do Kaggle ...")
    req = urllib.request.Request(cfg.KAGGLE_URL, headers={"User-Agent": "python"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        with zf.open(cfg.CSV_NAME) as fsrc, open(cfg.CSV_PATH, "wb") as fdst:
            fdst.write(fsrc.read())
    print(f"[data] Salvo em {cfg.CSV_PATH}")


def load_raw() -> pd.DataFrame:
    """Lê o CSV bruto, convertendo a coluna Date para datetime."""
    if not cfg.CSV_PATH.exists():
        download_dataset()
    df = pd.read_csv(cfg.CSV_PATH, parse_dates=["Date"])
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)
    return df


# Dicionário de variáveis (usado na EDA e no relatório)
VARIABLES = {
    "Date": "Data da observação",
    "Location": "Estação meteorológica (49 locais)",
    "MinTemp": "Temperatura mínima (°C)",
    "MaxTemp": "Temperatura máxima (°C)",
    "Rainfall": "Precipitação nas 24h até 9h (mm)",
    "Evaporation": "Evaporação 'Class A pan' nas 24h até 9h (mm)",
    "Sunshine": "Horas de sol no dia",
    "WindGustDir": "Direção da rajada mais forte (16 pontos)",
    "WindGustSpeed": "Velocidade da rajada mais forte (km/h)",
    "WindDir9am": "Direção do vento às 9h",
    "WindDir3pm": "Direção do vento às 15h",
    "WindSpeed9am": "Velocidade do vento às 9h (km/h)",
    "WindSpeed3pm": "Velocidade do vento às 15h (km/h)",
    "Humidity9am": "Umidade relativa às 9h (%)",
    "Humidity3pm": "Umidade relativa às 15h (%)",
    "Pressure9am": "Pressão ao nível do mar às 9h (hPa)",
    "Pressure3pm": "Pressão ao nível do mar às 15h (hPa)",
    "Cloud9am": "Nebulosidade às 9h (oitavos de céu)",
    "Cloud3pm": "Nebulosidade às 15h (oitavos de céu)",
    "Temp9am": "Temperatura às 9h (°C)",
    "Temp3pm": "Temperatura às 15h (°C)",
    "RainToday": "Choveu hoje? (precipitação > 1 mm)",
    "RainTomorrow": "Choverá amanhã? (alvo da classificação)",
}
