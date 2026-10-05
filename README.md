# Trabalho 1 — Regressão e Classificação em Sistemas Dinâmicos

**Disciplina:** Aprendizado de Máquina em Sistemas Dinâmicos — Prof. Americo Cunha
**Aluno:** João Pedro Dias de Carvalho (CompMat/UERJ)

## Dados

*Rain in Australia*, do Kaggle: <https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package>.
São observações diárias de 49 estações do Bureau of Meteorology (Austrália), de 2007 a 2017, com 145.460 linhas e 23 colunas.

O arquivo `data/weatherAUS.csv` já vem no pacote. Se ele for apagado, o `main.py` baixa de novo pela API pública do Kaggle.
Se o download falhar, baixe o arquivo pela página acima e salve em `data/`.

## Problemas e modelos

| Tarefa | Alvo | Modelos |
|---|---|---|
| Regressão (série temporal) | Temperatura máxima média **semanal** em Sydney (°C) | SARIMAX (Fourier + ARMA), Prophet, decomposição STL + ARIMA e combinação por média simples |
| Classificação (binária) | `RainTomorrow` (chove > 1 mm amanhã?) | Regressão Logística (GLM), KNN, Random Forest e LightGBM |

## Como executar

```bash
pip install -r requirements.txt
python main.py              # pipeline completo (~25 min em 2 núcleos)
python main.py --skip-clf   # só EDA + regressão (~3 min)
python main.py --compile    # também compila o relatório (requer LaTeX)
jupyter notebook notebook_trabalho1.ipynb   # execução etapa por etapa, com explicações
```

## Estrutura

```
main.py                    # orquestra todas as etapas
notebook_trabalho1.ipynb   # mesmo fluxo, célula a célula, com a teoria e a interpretação (já executado)
src/
  config.py                # caminhos, datas de corte, semente, paleta
  data.py                  # download do Kaggle e dicionário de variáveis
  eda.py                   # análise exploratória (ausentes, inconsistências, outliers, correlação, alvo)
  regression.py            # série semanal, SARIMAX, Prophet, STL, combinação, backtest, Diebold-Mariano, resíduos
  classification.py        # engenharia de atributos, pipelines, busca de hiperparâmetros, CV, teste, figuras
  plot_style.py            # padrão das figuras (IEEE, PDF vetorial + PNG 300 dpi)
  utils.py                 # exportação de tabelas (CSV + LaTeX) e JSON
outputs/
  figures/                 # todas as figuras (.pdf e .png)
  tables/                  # todas as tabelas (.csv e .tex)
  resultados.json          # números-chave de todo o experimento
report/
  main.tex                 # relatório (IEEEtran, duas colunas, pt-BR)
  references.bib
  figures/, tables/        # cópias usadas pelo LaTeX (geradas pelo main.py)
  main.pdf                 # relatório compilado
```

## Reprodutibilidade

* Semente fixa (`SEED = 42`) em todas as etapas aleatórias.
* As tabelas do relatório são geradas pelo código (`\input{tables/...}`), então os números do PDF batem com os da execução.
* A divisão treino/teste é temporal nas duas tarefas: treino até 2015 e teste de 2016-01 a 2017-06.

## Principais resultados

* **Regressão:** o SARIMAX teve RMSE de 1,65 °C no longo prazo (78 semanas) e 1,59 °C um passo à frente, cerca de 22% abaixo do modelo ingênuo.
  Ele empata estatisticamente com o Prophet e com a combinação (teste de Diebold–Mariano).
* **Classificação:** o LightGBM teve ROC-AUC de 0,889, PR-AUC de 0,752 e F1 de 0,674 no teste temporal, seguido pela Random Forest (AUC de 0,880).
