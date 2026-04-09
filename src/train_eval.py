"""
Pipeline: APS failure — comparação de modelos com custo FP=10, FN=500.
Execute a partir da raiz do projeto: python -m src.train_eval
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from src.config import APSConfig, to_log_dict, with_fast_debug
from src.data import (
    drop_degenerate_columns,
    encode_target,
    fit_imputer,
    load_raw,
    transform_imputer,
)
from src.experiments import append_experiment_row, new_run_id
from src.metrics import (
    classification_metrics,
    confusion_counts,
    find_best_threshold,
    total_cost,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "aps_failure_training_set.csv"
TEST_CSV = ROOT / "aps_failure_test_set.csv"
OUTPUT_DIR = ROOT / "outputs"


def set_seeds(random_state: int) -> None:
    np.random.seed(random_state)
    os.environ.setdefault("PYTHONHASHSEED", str(random_state))


def save_mlp_loss_curve_data(mlp: MLPClassifier, path: Path) -> None:
    pd.DataFrame(
        {
            "iteration": np.arange(1, len(mlp.loss_curve_) + 1, dtype=np.int32),
            "loss": mlp.loss_curve_,
        }
    ).to_csv(path, index=False)


def save_cost_vs_threshold_data(
    y_val: np.ndarray,
    proba_val: np.ndarray,
    path: Path,
    fp_cost: float,
    fn_cost: float,
) -> None:
    ts = np.linspace(0, 1, 401)
    costs = []
    for t in ts:
        pred = (proba_val >= t).astype(np.int32)
        costs.append(total_cost(y_val, pred, fp_cost, fn_cost))
    pd.DataFrame({"threshold": ts, "cost": costs}).to_csv(path, index=False)


def evaluate_split(
    name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    fp_cost: float,
    fn_cost: float,
) -> dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(np.int32)
    m = classification_metrics(y_true, y_pred, y_proba)
    m["cost"] = total_cost(y_true, y_pred, fp_cost, fn_cost)
    m["threshold"] = threshold
    m["model"] = name
    m.update(confusion_counts(y_true, y_pred))
    return m


def main(config: APSConfig | None = None) -> None:
    cfg = config or APSConfig()
    set_seeds(cfg.random_state)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id()

    raw_tr, raw_te = load_raw(TRAIN_CSV, TEST_CSV)
    X_tr_df, X_te_df, y_train, y_test = encode_target(raw_tr, raw_te)
    X_tr_df, X_te_df = drop_degenerate_columns(X_tr_df, X_te_df)
    feature_cols = list(X_tr_df.columns)

    imputer = fit_imputer(X_tr_df)
    X_imp_train = transform_imputer(imputer, X_tr_df, feature_cols)
    X_imp_test = transform_imputer(imputer, X_te_df, feature_cols)

    idx = np.arange(len(y_train))
    tr_idx, val_idx = train_test_split(
        idx, test_size=cfg.validation_size, stratify=y_train, random_state=cfg.random_state
    )
    X_tr = X_imp_train[tr_idx]
    X_val = X_imp_train[val_idx]
    y_tr = y_train[tr_idx]
    y_val = y_train[val_idx]

    scale_pos_weight = float((y_train == 0).sum() / max(1, int((y_train == 1).sum())))

    xgb_stage = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        n_estimators=cfg.xgb_n_estimators_max,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        min_child_weight=cfg.xgb_min_child_weight,
        random_state=cfg.random_state,
        n_jobs=-1,
        early_stopping_rounds=cfg.xgb_early_stopping_rounds,
        eval_metric="aucpr",
    )
    xgb_stage.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_trees = int(getattr(xgb_stage, "best_iteration", 0) or 0) + 1
    xgb_trees = max(xgb_trees, 50)

    scaler_mlp = StandardScaler()
    X_train_scaled = scaler_mlp.fit_transform(X_imp_train)
    X_test_scaled = scaler_mlp.transform(X_imp_test)
    X_val_scaled = scaler_mlp.transform(X_val)

    rf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    rf.fit(X_imp_train, y_train)

    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver=cfg.lr_solver,
                    max_iter=cfg.lr_max_iter,
                    C=cfg.lr_C,
                    class_weight="balanced",
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )
    lr_pipe.fit(X_imp_train, y_train)

    xgb_final = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        n_estimators=xgb_trees,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        min_child_weight=cfg.xgb_min_child_weight,
        random_state=cfg.random_state,
        n_jobs=-1,
        eval_metric="aucpr",
    )
    xgb_final.fit(X_imp_train, y_train, verbose=False)

    classes = np.unique(y_train)
    cw_arr = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = {int(c): float(w) for c, w in zip(classes, cw_arr)}
    sample_weight = np.array([weight_map[int(yi)] for yi in y_train])

    mlp = MLPClassifier(
        hidden_layer_sizes=cfg.mlp_hidden_layers,
        activation="relu",
        solver="adam",
        alpha=cfg.mlp_alpha,
        batch_size=cfg.mlp_batch_size,
        learning_rate_init=cfg.mlp_learning_rate_init,
        max_iter=cfg.mlp_max_iter,
        early_stopping=True,
        validation_fraction=cfg.mlp_validation_fraction,
        n_iter_no_change=cfg.mlp_n_iter_no_change,
        random_state=cfg.random_state,
        verbose=False,
    )
    mlp.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    model_names = ["random_forest", "logistic_regression", "xgboost", "mlp_sklearn"]

    thresholds: dict[str, float] = {}
    for name in model_names:
        if name == "mlp_sklearn":
            p_val = mlp.predict_proba(X_val_scaled)[:, 1]
        elif name == "random_forest":
            p_val = rf.predict_proba(X_val)[:, 1]
        elif name == "logistic_regression":
            p_val = lr_pipe.predict_proba(X_val)[:, 1]
        else:
            p_val = xgb_final.predict_proba(X_val)[:, 1]
        t, _ = find_best_threshold(y_val, p_val, cfg.fp_cost, cfg.fn_cost)
        thresholds[name] = t
        save_cost_vs_threshold_data(
            y_val,
            p_val,
            OUTPUT_DIR / f"cost_vs_threshold_{name}.csv",
            cfg.fp_cost,
            cfg.fn_cost,
        )

    save_mlp_loss_curve_data(mlp, OUTPUT_DIR / "mlp_loss_history.csv")

    def proba_test(name: str) -> np.ndarray:
        if name == "mlp_sklearn":
            return mlp.predict_proba(X_test_scaled)[:, 1]
        if name == "random_forest":
            return rf.predict_proba(X_imp_test)[:, 1]
        if name == "logistic_regression":
            return lr_pipe.predict_proba(X_imp_test)[:, 1]
        return xgb_final.predict_proba(X_imp_test)[:, 1]

    def proba_train(name: str) -> np.ndarray:
        if name == "mlp_sklearn":
            return mlp.predict_proba(X_train_scaled)[:, 1]
        if name == "random_forest":
            return rf.predict_proba(X_imp_train)[:, 1]
        if name == "logistic_regression":
            return lr_pipe.predict_proba(X_imp_train)[:, 1]
        return xgb_final.predict_proba(X_imp_train)[:, 1]

    results_test: list[dict[str, Any]] = []
    results_train: list[dict[str, Any]] = []
    for name in model_names:
        t = thresholds[name]
        pt = proba_test(name)
        ptr = proba_train(name)
        results_test.append(
            evaluate_split(name, y_test, pt, t, cfg.fp_cost, cfg.fn_cost)
        )
        m_tr = evaluate_split(name, y_train, ptr, t, cfg.fp_cost, cfg.fn_cost)
        m_tr["split"] = "train"
        results_train.append(m_tr)

    table = pd.DataFrame(results_test)
    cols = [
        "model",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "cost",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]
    table = table[[c for c in cols if c in table.columns]]
    table_sorted = table.sort_values(["cost", "recall"], ascending=[True, False])
    table_sorted.to_csv(OUTPUT_DIR / "metrics_test.csv", index=False)

    pd.DataFrame(results_train).to_csv(OUTPUT_DIR / "metrics_train.csv", index=False)

    overfit = []
    for name in model_names:
        te = next(x for x in results_test if x["model"] == name)
        tr = next(x for x in results_train if x["model"] == name)
        overfit.append(
            {
                "model": name,
                "acc_train": tr["accuracy"],
                "acc_test": te["accuracy"],
                "recall_train": tr["recall"],
                "recall_test": te["recall"],
                "f1_train": tr["f1"],
                "f1_test": te["f1"],
                "cost_train": tr["cost"],
                "cost_test": te["cost"],
            }
        )
    pd.DataFrame(overfit).to_csv(OUTPUT_DIR / "overfitting_train_vs_test.csv", index=False)

    for name in model_names:
        pt = proba_test(name)
        fpr, tpr, roc_thresholds = roc_curve(y_test, pt)
        pd.DataFrame(
            {"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}
        ).to_csv(OUTPUT_DIR / f"roc_test_{name}.csv", index=False)

        precision, recall, pr_thresholds = precision_recall_curve(y_test, pt)
        pr_thresholds_full = np.append(pr_thresholds, np.nan)
        pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "threshold": pr_thresholds_full,
            }
        ).to_csv(OUTPUT_DIR / f"pr_test_{name}.csv", index=False)

        t = thresholds[name]
        y_pred_te = (pt >= t).astype(np.int32)
        cm = confusion_matrix(y_test, y_pred_te, labels=[0, 1])
        pd.DataFrame(
            [
                {"true_label": 0, "pred_label": 0, "count": int(cm[0, 0])},
                {"true_label": 0, "pred_label": 1, "count": int(cm[0, 1])},
                {"true_label": 1, "pred_label": 0, "count": int(cm[1, 0])},
                {"true_label": 1, "pred_label": 1, "count": int(cm[1, 1])},
            ]
        ).to_csv(OUTPUT_DIR / f"confusion_test_{name}.csv", index=False)

    best = table_sorted.iloc[0].to_dict()
    summary = {
        "best_model_by_cost_on_test": best.get("model"),
        "best_cost": float(best.get("cost", float("nan"))),
        "best_recall": float(best.get("recall", float("nan"))),
        "note": "Limiar por validação (20% do treino) minimizando 10*FP+500*FN; "
        "modelos retreinados em todo o treino antes do teste.",
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    append_experiment_row(
        OUTPUT_DIR,
        run_id,
        to_log_dict(cfg),
        {
            "best_model_test": best.get("model"),
            "best_cost_test": float(best.get("cost", float("nan"))),
            "best_recall_test": float(best.get("recall", float("nan"))),
        },
    )

    print(table_sorted.to_string(index=False))
    print("\nResumo:", json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Run registado: run_id={run_id} -> outputs/experiments_log.csv")


if __name__ == "__main__":
    import sys

    cfg = with_fast_debug() if "--fast" in sys.argv else APSConfig()
    main(cfg)
