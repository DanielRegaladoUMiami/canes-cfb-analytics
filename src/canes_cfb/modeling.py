"""Model zoo, Optuna search spaces and walk-forward predictions for the points model.

Tree models learn the **residual of the ratings formula** (``points - exp_points``), the
shape that won the feature ablation: the ratings carry the signal and the trees correct
what they miss. Linear models predict points directly from the scaled features.

Every model is scored the same way: mean MAE over the walk-forward CV folds
(``validation.walk_forward``), never on 2024 (validation) or 2025 (test).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from canes_cfb.features import cumulative_sets
from canes_cfb.validation import walk_forward

FEATURES = cumulative_sets()["+priors"]
SEED = 7


@dataclass(frozen=True)
class ModelSpec:
    name: str
    residual: bool  # learn points - exp_points instead of points
    suggest: Callable[[optuna.Trial], dict[str, Any]]
    build: Callable[[dict[str, Any]], Any]
    defaults: dict[str, Any]


def _linear(model) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), model
    )


SPECS: dict[str, ModelSpec] = {
    "ridge": ModelSpec(
        "ridge",
        residual=False,
        suggest=lambda t: {"alpha": t.suggest_float("alpha", 1e-2, 1e3, log=True)},
        build=lambda p: _linear(Ridge(**p)),
        defaults={"alpha": 10.0},
    ),
    "poisson": ModelSpec(
        "poisson",
        residual=False,
        suggest=lambda t: {"alpha": t.suggest_float("alpha", 1e-4, 1.0, log=True)},
        build=lambda p: _linear(PoissonRegressor(max_iter=5000, **p)),
        defaults={"alpha": 1e-2},
    ),
    "random_forest": ModelSpec(
        "random_forest",
        residual=True,
        suggest=lambda t: {
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 20, 400, log=True),
            "max_features": t.suggest_float("max_features", 0.1, 0.6),
            "max_depth": t.suggest_int("max_depth", 3, 12),
        },
        build=lambda p: _linear_free(
            RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED, **p)
        ),
        defaults={"min_samples_leaf": 100, "max_features": 0.3, "max_depth": 8},
    ),
    "lightgbm": ModelSpec(
        "lightgbm",
        residual=True,
        suggest=lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 100, 1500, log=True),
            "learning_rate": t.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": t.suggest_int("num_leaves", 2, 31, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 20, 500, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30, log=True),
        },
        build=lambda p: lgb.LGBMRegressor(
            subsample_freq=1, random_state=SEED, verbose=-1, objective="l1", **p
        ),
        defaults={
            "n_estimators": 300,
            "learning_rate": 0.02,
            "num_leaves": 7,
            "min_child_samples": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
    ),
    "xgboost": ModelSpec(
        "xgboost",
        residual=True,
        suggest=lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 100, 1500, log=True),
            "learning_rate": t.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": t.suggest_int("max_depth", 1, 6),
            "min_child_weight": t.suggest_float("min_child_weight", 1, 300, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30, log=True),
        },
        build=lambda p: XGBRegressor(
            objective="reg:absoluteerror", tree_method="hist", random_state=SEED, n_jobs=-1, **p
        ),
        defaults={
            "n_estimators": 300,
            "learning_rate": 0.02,
            "max_depth": 2,
            "min_child_weight": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
    ),
    "catboost": ModelSpec(
        "catboost",
        residual=True,
        suggest=lambda t: {
            "iterations": t.suggest_int("iterations", 200, 1500, log=True),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": t.suggest_int("depth", 2, 6),
            "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 30, log=True),
            "rsm": t.suggest_float("rsm", 0.2, 1.0),
        },
        build=lambda p: CatBoostRegressor(
            loss_function="MAE", random_seed=SEED, verbose=0, allow_writing_files=False, **p
        ),
        defaults={
            "iterations": 500,
            "learning_rate": 0.03,
            "depth": 3,
            "l2_leaf_reg": 5.0,
            "rsm": 0.8,
        },
    ),
}


def _linear_free(model) -> Any:
    # Random forests don't need scaling, but sklearn's RF can't take NaN in older versions.
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), model)


def fit_predict(
    spec: ModelSpec,
    params: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] = FEATURES,
) -> np.ndarray:
    """Fit on ``train`` and return point predictions for ``test``."""
    target = train["points"] - train["exp_points"] if spec.residual else train["points"]
    model = spec.build(params).fit(train[features], target.to_numpy())
    pred = model.predict(test[features])
    return pred + test["exp_points"].to_numpy() if spec.residual else pred


def oof_predictions(
    spec: ModelSpec, params: dict[str, Any], data: pd.DataFrame, features: list[str] = FEATURES
) -> pd.Series:
    """Walk-forward out-of-fold predictions for the CV seasons (NaN elsewhere)."""
    out = pd.Series(np.nan, index=data.index)
    for _, train, valid in walk_forward(data):
        out.iloc[valid] = fit_predict(spec, params, data.iloc[train], data.iloc[valid], features)
    return out


def cv_mae(pred: pd.Series, data: pd.DataFrame) -> float:
    """Mean of per-season MAE over the CV seasons (each fold weighs the same)."""
    scored = data.assign(err=(data["points"] - pred).abs())[pred.notna()]
    return float(scored.groupby("season")["err"].mean().mean())


def tune(
    spec: ModelSpec, data: pd.DataFrame, n_trials: int, timeout: float | None = None
) -> optuna.Study:
    """Optuna TPE search minimizing walk-forward CV MAE. Starts from the defaults."""
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(spec.defaults)
    study.optimize(
        lambda t: cv_mae(oof_predictions(spec, spec.suggest(t), data), data),
        n_trials=n_trials,
        timeout=timeout,
    )
    return study


def ensemble_predict(preds: pd.DataFrame, recipe: dict) -> pd.Series:
    """Apply the final recipe saved by notebooks/04_team_total/full_game.ipynb."""
    final = recipe["final"]
    if final in preds.columns:
        return preds[final]
    if final == "average (all)":
        return preds[recipe["models"]].mean(axis=1)
    if final == "average (top 3)":
        return preds[recipe["top3"]].mean(axis=1)
    if final == "stacking (NNLS)":
        weights = pd.Series(recipe["stack_weights"])
        return preds[list(weights.index)] @ weights
    raise ValueError(f"unknown recipe: {final}")
