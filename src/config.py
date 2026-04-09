"""Hiperparâmetros centralizados do pipeline APS (treino + notebook)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class APSConfig:
    """Configuração reprodutível; use `to_log_dict()` para serializar (JSON/CSV)."""

    random_state: int = 42
    fp_cost: float = 10.0
    fn_cost: float = 500.0
    validation_size: float = 0.2

    rf_n_estimators: int = 300
    rf_max_depth: int = 20
    rf_min_samples_leaf: int = 4

    lr_max_iter: int = 10_000
    lr_solver: str = "saga"
    lr_C: float = 1.0

    xgb_n_estimators_max: int = 3000
    xgb_early_stopping_rounds: int = 60
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_min_child_weight: int = 3

    mlp_hidden_layers: tuple[int, ...] = (128, 64)
    mlp_alpha: float = 1e-4
    mlp_batch_size: int = 256
    mlp_learning_rate_init: float = 1e-3
    mlp_max_iter: int = 200
    mlp_validation_fraction: float = 0.15
    mlp_n_iter_no_change: int = 20


def with_fast_debug(base: APSConfig | None = None) -> APSConfig:
    """Valores menores para depuração rápida (notebook / testes)."""
    b = base or APSConfig()
    return replace(
        b,
        xgb_n_estimators_max=800,
        xgb_early_stopping_rounds=20,
        rf_n_estimators=80,
        mlp_max_iter=50,
    )


def to_log_dict(cfg: APSConfig) -> dict[str, Any]:
    """Dict JSON-friendly (tuplos → listas)."""
    d = asdict(cfg)
    hl = d.pop("mlp_hidden_layers")
    d["mlp_hidden_layers"] = list(hl)
    return d
