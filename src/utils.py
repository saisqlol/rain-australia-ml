"""Funções auxiliares: exportação de tabelas (CSV + LaTeX) e de resultados."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as cfg


LATEX_NAMES = {
    "R2": r"$R^2$",
    "MAPE (%)": r"MAPE (\%)",
    "Cobertura IC95 (%)": r"Cob. IC95 (\%)",
    "PR-AUC": r"PR-AUC",
}


def _tex_escape(s: str) -> str:
    if s in LATEX_NAMES:
        return LATEX_NAMES[s]
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def save_table(df: pd.DataFrame, name: str, float_fmt: str = "{:.3f}",
               index: bool = True, bold_best: dict | None = None,
               latex_rename: dict | None = None) -> None:
    """Salva `df` em outputs/tables/<name>.csv e <name>.tex (booktabs).

    O .tex contém só o ambiente tabular, para ser incluído no relatório
    com \\input{tables/<name>.tex}. Assim os números do PDF sempre batem
    com os gerados pelo código.

    bold_best: {coluna: "min" | "max"} -> destaca em negrito o melhor valor.
    latex_rename: rótulos curtos (colunas/linhas) usados só no .tex, para a
                  tabela caber em uma coluna do artigo; o CSV mantém os nomes completos.
    """
    df.to_csv(cfg.TAB_DIR / f"{name}.csv", index=index)

    def num(v):  # vírgula decimal (padrão pt-BR) e sinal de menos tipográfico
        txt = float_fmt.format(v).replace(".", ",")
        return "$-$" + txt[1:] if txt.startswith("-") else txt

    fmt = df.copy().astype(object)
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            best = None
            if bold_best and col in bold_best:
                best = df[col].min() if bold_best[col] == "min" else df[col].max()
            fmt[col] = [
                "--" if pd.isna(v) else
                (r"\textbf{" + num(v) + "}" if best is not None and
                 np.isclose(v, best) else num(v))
                for v in df[col]
            ]
        else:
            fmt[col] = [_tex_escape(str(v)) for v in df[col]]
    ren = latex_rename or {}
    fmt.columns = [ren.get(c, _tex_escape(str(c))) for c in fmt.columns]
    fmt.index = [ren.get(i, _tex_escape(str(i))) for i in fmt.index]
    fmt.index.name = None
    colfmt = ("l" if index else "") + "".join(
        "r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns)
    latex = fmt.to_latex(index=index, escape=False, column_format=colfmt)
    (cfg.TAB_DIR / f"{name}.tex").write_text(latex, encoding="utf-8")


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o.date())
        return super().default(o)


def save_json(obj: dict, name: str) -> None:
    path = cfg.OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, cls=NpEncoder),
                    encoding="utf-8")
