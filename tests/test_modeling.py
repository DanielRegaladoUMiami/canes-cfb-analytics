import pandas as pd
import pytest

from canes_cfb.modeling import ensemble_predict

PREDS = pd.DataFrame({"a": [10.0, 20.0], "b": [20.0, 30.0], "c": [30.0, 40.0]})
RECIPE = {"models": ["a", "b", "c"], "top3": ["a", "b", "c"],
          "stack_weights": {"a": 0.5, "b": 0.5, "c": 0.0}}  # fmt: skip


@pytest.mark.parametrize(
    ("final", "expected"),
    [
        ("b", [20.0, 30.0]),
        ("average (all)", [20.0, 30.0]),
        ("stacking (NNLS)", [15.0, 25.0]),
    ],
)
def test_ensemble_recipes(final, expected):
    out = ensemble_predict(PREDS, {**RECIPE, "final": final})
    assert out.tolist() == expected


def test_unknown_recipe_fails():
    with pytest.raises(ValueError):
        ensemble_predict(PREDS, {**RECIPE, "final": "magic"})
