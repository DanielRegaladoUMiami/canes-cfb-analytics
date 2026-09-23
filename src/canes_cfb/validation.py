"""Time-based split and walk-forward cross-validation (docs/pipeline.md §3)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

TRAIN_SEASONS = range(2015, 2024)
VALIDATION_SEASON = 2024
TEST_SEASON = 2025
CV_SEASONS = (2019, 2020, 2021, 2022, 2023)


def walk_forward(
    df: pd.DataFrame, cv_seasons: tuple[int, ...] = CV_SEASONS, first_season: int = 2016
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (season, train_idx, valid_idx): train on seasons before ``season``, validate on it.

    ``first_season`` skips 2015, whose features have no prior season to lean on.
    """
    seasons = df["season"].to_numpy()
    for season in cv_seasons:
        train = np.flatnonzero((seasons >= first_season) & (seasons < season))
        valid = np.flatnonzero(seasons == season)
        yield season, train, valid
