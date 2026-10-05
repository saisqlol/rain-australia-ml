"""
Trabalho 1 - Aprendizado de Máquina em Sistemas Dinâmicos
Aplicações de Regressão e Classificação no dataset "Rain in Australia" (Kaggle)

Executa o projeto de ponta a ponta:
  1. Download dos dados (API pública do Kaggle)
  2. Análise exploratória e pré-processamento
  3. Regressão  : previsão da temperatura máxima semanal em Sydney
                  (SARIMAX, Prophet, decomposição STL e combinação por média simples)
  4. Classificação: "vai chover amanhã?" (Regressão Logística, KNN,
                  Random Forest e LightGBM)
  5. Exporta figuras/tabelas para report/ e (opcionalmente) compila o PDF

Uso:
    python main.py                 # tudo
    python main.py --skip-clf      # só EDA + regressão
    python main.py --skip-reg      # só EDA + classificação
    python main.py --compile       # também compila report/main.tex (requer pdflatex)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time

from src import config as cfg
from src import data, eda, regression, classification
from src.plot_style import set_style
from src.utils import save_json


def banner(txt: str) -> None:
    print("\n" + "=" * 78 + f"\n{txt}\n" + "=" * 78)


def export_to_report() -> None:
    """Copia figuras (PDF) e tabelas (.tex) para a pasta do relatório LaTeX."""
    fig_dst = cfg.REPORT_DIR / "figures"
    tab_dst = cfg.REPORT_DIR / "tables"
    fig_dst.mkdir(parents=True, exist_ok=True)
    tab_dst.mkdir(parents=True, exist_ok=True)
    for f in cfg.FIG_DIR.glob("*.pdf"):
        shutil.copy(f, fig_dst / f.name)
    for f in cfg.TAB_DIR.glob("*.tex"):
        shutil.copy(f, tab_dst / f.name)
    print(f"[report] figuras e tabelas copiadas para {cfg.REPORT_DIR}")


def compile_latex() -> None:
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", "main.tex"]
    try:
        subprocess.run(cmd, cwd=cfg.REPORT_DIR, check=True)
        print(f"[report] PDF gerado em {cfg.REPORT_DIR / 'main.pdf'}")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[report] Não foi possível compilar o LaTeX: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-reg", action="store_true", help="pula a regressão")
    ap.add_argument("--skip-clf", action="store_true", help="pula a classificação")
    ap.add_argument("--compile", action="store_true", help="compila o relatório LaTeX")
    args = ap.parse_args()

    t0 = time.time()
    set_style()
    results = {}

    banner("1. DADOS")
    data.download_dataset()
    df = data.load_raw()
    print(f"{df.shape[0]:,} linhas x {df.shape[1]} colunas")

    banner("2. ANÁLISE EXPLORATÓRIA")
    results["eda"] = eda.run_eda(df)
    for k, v in results["eda"].items():
        print(f"  {k}: {v}")

    if not args.skip_reg:
        banner("3. REGRESSÃO - temperatura máxima semanal (Sydney)")
        out = regression.run_regression(df)
        out.pop("_objects")
        results["regressao"] = out

    if not args.skip_clf:
        banner("4. CLASSIFICAÇÃO - RainTomorrow")
        out = classification.run_classification(df)
        out.pop("_objects")
        results["classificacao"] = out

    save_json(results, "resultados")
    banner("5. RELATÓRIO")
    export_to_report()
    if args.compile:
        compile_latex()
    print(f"\nConcluído em {(time.time() - t0) / 60:.1f} min. "
          f"Resultados em {cfg.OUT_DIR / 'resultados.json'}")


if __name__ == "__main__":
    main()
