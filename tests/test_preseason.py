import pandas as pd

from canes_cfb.features import preseason_table

TEAMS = pd.DataFrame({"team_id": [1, 2], "team": ["Miami", "Florida"]})


def test_preseason_table():
    portal = pd.DataFrame(
        {"season": [2024, 2024, 2024], "origin": ["Florida", "Florida", "Miami"],
         "destination": ["Miami", None, "Florida"], "stars": [4, 3, None],
         "rating": [None] * 3, "transferDate": ["2024-01-05"] * 3}
    )  # fmt: skip
    recruiting = pd.DataFrame(
        {"season": [2021, 2022, 2023, 2024], "team": ["Miami"] * 4,
         "recruit_points": [200.0, 220.0, 240.0, 260.0]}
    )  # fmt: skip
    coaches = pd.DataFrame(
        {"season": [2024, 2024, 2024], "team_id": [1, 2, 2],
         "coach": ["Old Coach", "New Coach", "Interim"],
         "hire_date": ["2021-12-06", "2023-11-30", "2024-10-20"], "games": [12, 8, 4]}
    )  # fmt: skip
    ap = pd.DataFrame({"season": [2024], "team_id": [1], "ap_points": [492], "ap_rank": [19]})
    t = preseason_table(portal, recruiting, coaches, TEAMS, {1, 2}, ap).set_index(
        ["season", "team_id"]
    )
    miami, florida = t.loc[(2024, 1)], t.loc[(2024, 2)]
    assert (miami.portal_in, miami.portal_out) == (1, 1)
    assert miami.portal_net_stars == 4 - 2  # unknown stars count as 2
    assert florida.portal_net_stars == 2 - (4 + 3)
    assert miami.recruit_4yr == 230.0
    assert miami.new_coach == 0  # hired 2021
    assert florida.new_coach == 1  # starter hired Nov 2023; the mid-season interim is ignored
    assert (miami.ap_points, florida.ap_points) == (492, 0)  # unranked = 0
