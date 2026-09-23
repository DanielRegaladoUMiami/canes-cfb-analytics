"""Tune every model in the zoo with Optuna (walk-forward CV 2019-2023) and save the best
parameters to models/team_points_params.json.

    uv run python scripts/tune_team_points.py [--quick]
"""

from __future__ import annotations

import json
import sys
import time

import optuna
import pandas as pd

from canes_cfb.modeling import SPECS, tune
from canes_cfb.paths import PROCESSED, ROOT

TRIALS = {"ridge": 25, "poisson": 20, "random_forest": 15, "lightgbm": 60, "xgboost": 40,
          "catboost": 25}  # fmt: skip
TIMEOUT = 900  # seconds per model


def main() -> None:
    quick = "--quick" in sys.argv
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    f = pd.read_parquet(PROCESSED / "team_games.parquet")
    data = f[f.completed & f.season.between(2016, 2023) & ~f.shortened].reset_index(drop=True)
    out_path = ROOT / "models" / "team_points_params.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name, spec in SPECS.items():
        start = time.time()
        study = tune(spec, data, n_trials=2 if quick else TRIALS[name], timeout=TIMEOUT)
        default_mae = study.trials[0].value
        results[name] = {
            "params": study.best_params,
            "cv_mae": round(study.best_value, 4),
            "cv_mae_defaults": round(default_mae, 4),
            "trials": len(study.trials),
        }
        print(f"{name:14s} defaults {default_mae:.3f} -> tuned {study.best_value:.3f} "
              f"({len(study.trials)} trials, {time.time() - start:.0f}s)", flush=True)  # fmt: skip
        if not quick:
            out_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
