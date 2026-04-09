# APS Failure - Guia de uso do notebook

Este projeto compara modelos supervisionados para diagnóstico de falhas no sistema APS de caminhões, com foco em custo de erro:

- Falso positivo (FP): custo `10`
- Falso negativo (FN): custo `500`

Por isso, a decisão final prioriza **custo total** e **recall**.

## Estrutura principal

- Notebook principal: `notebooks/aps_benchmark.ipynb`
- Configuração de hiperparâmetros: `src/config.py`
- Utilitários de experimento/log: `src/experiments.py`
- Script equivalente ao notebook: `src/train_eval.py`
- Dados de entrada:
  - `aps_failure_training_set.csv`
  - `aps_failure_test_set.csv`
- Saídas: pasta `outputs/`

## 1) Pré-requisitos

Recomendado:

- Python 3.11+ (o projeto está rodando no seu ambiente com Python 3.14)
- `pip` atualizado

Instale dependências na raiz do projeto:

```bash
pip install -r requirements.txt
```

Valide rapidamente:

```bash
pip show pandas scikit-learn xgboost matplotlib
```

## 2) Como abrir e executar o notebook

1. Abra `notebooks/aps_benchmark.ipynb`.
2. Selecione o kernel Python onde você instalou o `requirements.txt`.
3. Execute em ordem (Run All).

O notebook detecta automaticamente a raiz do projeto e cria `outputs/` se necessário.

## 3) Como alterar hiperparâmetros

Os hiperparâmetros estão centralizados em `src/config.py` na classe `APSConfig`.

### Principais grupos

- Random Forest: `rf_n_estimators`, `rf_max_depth`, `rf_min_samples_leaf`
- Regressão Logística: `lr_solver`, `lr_max_iter`, `lr_C`
- XGBoost: `xgb_n_estimators_max`, `xgb_early_stopping_rounds`, `xgb_max_depth`, `xgb_learning_rate`, `xgb_subsample`, `xgb_colsample_bytree`, `xgb_min_child_weight`
- MLP: `mlp_hidden_layers`, `mlp_alpha`, `mlp_batch_size`, `mlp_learning_rate_init`, `mlp_max_iter`, `mlp_validation_fraction`, `mlp_n_iter_no_change`
- Controle global: `random_state`, `fp_cost`, `fn_cost`, `validation_size`

### Modo rápido para debug

No notebook, use:

```python
FAST_DEBUG = True
cfg = with_fast_debug() if FAST_DEBUG else APSConfig()
```

Esse modo reduz custo computacional para testes curtos (menos árvores/iterações).

### Exemplo: alterar configuração-base

Edite em `src/config.py`:

```python
@dataclass(frozen=True)
class APSConfig:
    xgb_learning_rate: float = 0.03
    rf_n_estimators: int = 500
    mlp_hidden_layers: tuple[int, ...] = (256, 128, 64)
```

Depois, rode o notebook novamente para gerar uma nova execução.

## 4) Como rodar vários testes com parâmetros diferentes

## Fluxo recomendado (sem vazar o teste)

1. Ajuste hiperparâmetros em `src/config.py` (ou via `FAST_DEBUG` no notebook para debug).
2. Rode o notebook.
3. Verifique métricas e custo.
4. Repita com nova configuração.

Importante:

- O ajuste de limiar já é refeito na validação para cada run.
- Evite escolher parâmetros olhando o teste a cada tentativa; use validação para seleção e teste para comparação final.

### Log automático de execuções

Cada execução registra uma linha em:

- `outputs/experiments_log.csv`

Com campos principais:

- `timestamp`
- `run_id`
- `config_json`
- `best_model_test`
- `best_cost_test`
- `best_recall_test`

Isso facilita comparar múltiplas execuções sem perder histórico.

### Grid de hiperparâmetros (XGBoost) no notebook

O notebook inclui célula de grid usando `grid_search_xgboost_validation(...)` de `src/experiments.py`.

Essa célula:

- Varia combinações definidas em `param_grid_xgb`
- Avalia **somente em treino/validação**
- Salva resultado em:
  - `outputs/grid_xgb_validation.csv`

## 5) Como comparar os resultados

Arquivos principais em `outputs/`:

- `metrics_test.csv`: métricas no teste (ordenáveis por custo/recall)
- `metrics_train.csv`: métricas no treino
- `overfitting_train_vs_test.csv`: comparação treino vs teste
- `summary.json`: melhor modelo/custo/recall da execução
- `experiments_log.csv`: histórico de execuções
- `grid_xgb_validation.csv`: resultados do grid (quando a célula de grid for executada)
- `mlp_loss_history.csv`: pontos de loss por iteração do MLP
- `cost_vs_threshold_<model>.csv`: pontos da curva custo x limiar na validação
- `roc_test_<model>.csv`, `pr_test_<model>.csv`, `confusion_test_<model>.csv`: dados para gráficos no notebook

Colunas de confusão em `metrics_test.csv` / `metrics_train.csv` (rótulo 0 = negativo, 1 = positivo):

- `true_negatives`: TN — real 0, predito 0
- `false_positives`: FP — real 0, predito 1
- `false_negatives`: FN — real 1, predito 0
- `true_positives`: TP — real 1, predito 1

Critério sugerido:

1. Menor `cost`
2. Maior `recall` (desempate)

## 6) Boas práticas

- Mantenha `random_state` fixo para comparar runs de forma justa.
- Reotimize limiar em cada nova configuração (já feito no pipeline).
- Comece com `FAST_DEBUG=True` para validar fluxo.
- Rode `FAST_DEBUG=False` para resultados finais.

## 7) Problemas comuns

### Kernel não encontra pacotes

- O kernel selecionado não é o mesmo Python do `pip install -r requirements.txt`.
- Selecione o interpretador correto no notebook.

### Treinamento muito demorado

- Ative `FAST_DEBUG=True`.
- Reduza combinações do grid (`param_grid_xgb`).
- Diminua parâmetros como `xgb_n_estimators_max` e `mlp_max_iter`.

### Onde vejo o melhor resultado?

- `outputs/summary.json` (run atual)
- `outputs/experiments_log.csv` (histórico)

## 8) Execução via script (opcional)

Além do notebook:

```bash
python -m src.train_eval
```

Modo rápido:

```bash
python -m src.train_eval --fast
```

Saídas e logs são gravados na mesma pasta `outputs/`.
