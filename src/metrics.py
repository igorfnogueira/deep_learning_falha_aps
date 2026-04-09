"""Métricas, custo e escolha de limiar para o problema APS."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def total_cost(y_true: np.ndarray, y_pred: np.ndarray, fp_cost: float = 10.0, fn_cost: float = 500.0) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp_cost * fp + fn_cost * fn


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fp_cost: float = 10.0,
    fn_cost: float = 500.0,
    n_thresholds: int = 501,
) -> tuple[float, float]:
    """
    Encontra o limiar que minimiza fp_cost * FP + fn_cost * FN.
    Inclui limites 0 e 1 e uma malha uniforme; também insere valores únicos dos escores como candidatos.
    """
    y_proba = np.asarray(y_proba, dtype=np.float64).ravel()
    y_true = np.asarray(y_true).ravel()
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, n_thresholds),
                np.quantile(y_proba, np.linspace(0, 1, min(101, len(y_proba)))),
            ]
        )
    )
    best_t = 0.5
    best_cost = float("inf")
    for t in candidates:
        y_pred = (y_proba >= t).astype(np.int8)
        c = total_cost(y_true, y_pred, fp_cost, fn_cost)
        if c < best_cost:
            best_cost = c
            best_t = float(t)
    return best_t, best_cost
