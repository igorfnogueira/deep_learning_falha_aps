"""Registo de execuções e grelha de hiperparâmetros só em validação (sem teste)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier

from src.metrics import classification_metrics, find_best_threshold


def append_experiment_row(
    output_dir: Path,
    run_id: str,
    config_dict: dict[str, Any],
    metrics_summary: dict[str, Any],
) -> Path:
    """
    Acrescenta uma linha a outputs/experiments_log.csv com timestamp e JSON da config.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiments_log.csv"
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "config_json": json.dumps(config_dict, ensure_ascii=False, sort_keys=True),
        **metrics_summary,
    }
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")
    return path


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def grid_search_xgboost_validation(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
    param_grid: dict[str, list[Any]],
    fixed_params: dict[str, Any],
    fp_cost: float,
    fn_cost: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Varre combinações no **treino parcial** com early stopping na validação.
    Limiar ótimo por custo só na validação. **Não usa o conjunto de teste.**

    fixed_params: chaves comuns (ex. max_depth, subsample) e obrigatoriamente
    n_estimators_max (teto de árvores na fase com early stopping) e early_stopping_rounds.
    param_grid: apenas dimensões que quer variar (ex. learning_rate, max_depth).
    """
    rows: list[dict[str, Any]] = []
    n_est_max = int(fixed_params["n_estimators_max"])
    es_rounds = int(fixed_params.get("early_stopping_rounds", 60))
    base_fixed = {k: v for k, v in fixed_params.items() if k not in ("n_estimators_max", "early_stopping_rounds")}

    for params in ParameterGrid(param_grid):
        merged = {**base_fixed, **params}
        clf = XGBClassifier(
            objective="binary:logistic",
            n_estimators=n_est_max,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            early_stopping_rounds=es_rounds,
            eval_metric="aucpr",
            **merged,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        p_val = clf.predict_proba(X_val)[:, 1]
        thr, val_cost = find_best_threshold(y_val, p_val, fp_cost, fn_cost)
        y_pred = (p_val >= thr).astype(np.int32)
        m = classification_metrics(y_val, y_pred, p_val)
        bi = getattr(clf, "best_iteration", None)
        row: dict[str, Any] = {
            **params,
            "val_threshold": thr,
            "val_cost": val_cost,
            "best_iteration": int(bi) if bi is not None else None,
        }
        row.update({f"val_{k}": v for k, v in m.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def new_run_id() -> str:
    return str(uuid.uuid4())[:8]
