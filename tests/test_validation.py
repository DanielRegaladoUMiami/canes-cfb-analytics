import pandas as pd

from canes_cfb.validation import CV_SEASONS, TEST_SEASON, VALIDATION_SEASON, walk_forward


def test_walk_forward_never_trains_on_the_future():
    df = pd.DataFrame({"season": [s for s in range(2015, 2026) for _ in range(3)]})
    for season, train, valid in walk_forward(df):
        assert df.season.iloc[train].max() < season
        assert (df.season.iloc[valid] == season).all()
        assert df.season.iloc[train].min() == 2016


def test_cv_stays_inside_the_train_window():
    assert max(CV_SEASONS) < VALIDATION_SEASON < TEST_SEASON
