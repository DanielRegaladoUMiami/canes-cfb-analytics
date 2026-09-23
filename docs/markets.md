# Market definitions

Every notebook uses these rules. The code version lives in `src/canes_cfb/markets.py`.

## Periods

| Period | Points counted | Notes |
|---|---|---|
| `full_game` | Q1 + Q2 + Q3 + Q4 + OT | Standard for moneyline, spread, total |
| `1H` | Q1 + Q2 | |
| `2H` | Q3 + Q4 + OT | Most US books include OT in the second half. Confirm your book's house rules. |
| `Q1`–`Q4` | That quarter only | Regulation only. OT never counts toward Q4. |

Check: `1H + 2H = full_game` for every game. The data notebooks assert this.

## Sign conventions

Everything is from the **home team's perspective**. At neutral sites, "home" is whoever
the data source lists as home.

- **Margin** = home points − away points.
- **Spread line** = the home team's spread. `-7` means home is favored by 7.
  - Home covers if `margin + line > 0`.
  - Push if `margin + line == 0` (only possible on whole-number lines).
- **Total** = home points + away points. Over if `total > line`.
- **Team total** = one team's points. The feature table stores one row per team per game
  (`is_home` flag), so a single model covers both sides.

## Targets and metrics

| Market | Target | Kind | Metrics |
|---|---|---|---|
| Winner | `home_win` | binary | log loss, Brier, accuracy, ROI vs close |
| Spread | `home_margin` | continuous → P(cover) | MAE, RMSE, cover rate, CLV, ROI |
| Total | `total_points` | continuous → P(over) | MAE, RMSE, over rate, CLV, ROI |
| Team total | `team_points` | continuous → P(over) | MAE, RMSE, over rate, CLV, ROI |

Continuous models become probabilities by putting a distribution on the prediction.
Start with a normal distribution using the residual SD from the walk-forward fit.
Football scores cluster on key numbers (3, 7, 10, 14), so check the empirical margin
distribution before trusting the normal assumption on spreads.

## ROI accounting

- Assume −110 pricing unless the actual price was recorded.
- Pushes return the stake and count as neither win nor loss.
- **CLV** (closing line value) = how much the line moved toward our pick between when
  we'd bet and kickoff. It's the most stable early signal of real edge.
