import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "build_site", Path(__file__).parents[1] / "scripts" / "build_site.py"
)
build_site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_site)


def test_split_whole_adds_up():
    assert sum(build_site.split_whole(55, [31.7, 22.8])) == 55
    assert build_site.split_whole(7, [0.0, 0.0]) in ([4, 3], [3, 4])
    assert build_site.split_whole(0, [3.0, 4.0]) == [0, 0]


def test_period_points_are_whole_and_consistent():
    row = SimpleNamespace(pred=31.7, pred_away=22.8, pred_total=54.46)
    raw = {
        "1H": (27.1, 6.4),
        "2H": (24.1, 4.5),
        "Q1": (11.2, 2.6),
        "Q2": (16.7, 2.1),
        "Q3": (11.0, 2.8),
        "Q4": (13.9, 0.1),
    }
    for p, (t, m) in raw.items():
        setattr(row, f"pred_total_{p}", t)
        setattr(row, f"pred_margin_{p}", m)
    P = build_site.period_points(row)
    assert P["Game"]["home"] + P["Game"]["away"] == 55  # 54.46 shows as 54.5 → 55
    for s in ("home", "away"):
        assert P["1H"][s] + P["2H"][s] == P["Game"][s]
        assert P["Q1"][s] + P["Q2"][s] == P["1H"][s]
        assert P["Q3"][s] + P["Q4"][s] == P["2H"][s]
        assert all(isinstance(P[p][s], int) for p in P)


def test_grade_totals_and_spreads():
    g = build_site.grade
    assert g("under", 50.5, 48) == "win"
    assert g("over", 50.5, 48) == "loss"
    assert g("over", 48, 48) == "push"
    # spread: line is the home margin needed (home -7 -> 7); home wins by 10 covers
    assert g("home", 7, 10) == "win"
    assert g("away", 7, 10) == "loss"
    assert g(None, 7, 10) is None
