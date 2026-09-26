# Model card: capacity

**Simulated data only.** Every cell and its hourly traffic comes from the synthetic generator
in `telecom_ml_capacity.generator`. No real network data is used, and the results say nothing
about any real network.

## Task

Forecast each cell's hourly traffic 7 days ahead. Scored by mean absolute percentage error
(MAPE), against both the live model and a seasonal naive forecast.

## Data

- 60 cells x 24 hours per day. Traffic follows the cell's area and a diurnal x weekly shape,
  with per-hour noise, short autocorrelated noise, special-event bursts, rare outages and a
  slow per-cell level that wanders by day.
- Each row carries the same cell and hour 7, 8 and 14 days earlier and the mean over days 7 to
  13: everything a forecast made 7 days ahead may know.
- Label delay: none. Actual traffic arrives each hour.

## Models

| | Model | Inputs |
|---|---|---|
| live and candidate | LightGBM (63 leaves, 300 trees, learning rate 0.05) | the four history columns, hour, weekday, cell type and area |
| baseline | seasonal naive: the same cell and hour 7 days earlier | one history column |

Candidates retrain from scratch on the last 28 days; every model is scored on the last 14.
Splits are chronological only.

## Drift calendar

180 simulated days from 2026-01-01 (`scenario.yaml`):

| Day | Event | Kind | Change |
|---|---|---|---|
| 0 | steady growth | trend | every cell +2% a month, +12% by day 180 |
| 70 | fast-growth cells | trend | 15 cells grow at 5% a month from here |
| 130 | holiday week (benign) | covariate | 30% of cells at 1.8 times their traffic for 7 days, then back |

## What the loop shows here

A forecaster built on same-hour lags tracks growth through its lags: the higher level is
already in the inputs. The growth events are detected (drift and error) and trigger retrains,
but a new model is not better, so none is promoted. This is a finding, not a gap: not every
drift needs a new model, and the loop's job there is to refuse the wrong one.

The holiday week shows the other side. Models retrained with the holiday in their window
forecast worse once it is over: on day 151 such a candidate scored MAPE 25.5 against the live
model's 19.0. The promotion gate refused every one, and the day-0 model stayed live through
all 180 days.

## Loop rules

- Drift is checked on one row per cell per day: the day's traffic over the same weekday 7
  and 14 days earlier. The ratios do not move with the weekly pattern; raw daily means of
  traffic, users, PRB and latency, which all follow traffic, flagged dataset drift on 136 of
  180 scenario-off days.
- Retrain on dataset drift of that frame, or when MAPE rises 2 points above its value at
  promotion.
- Promote when MAPE beats live by 0.315 points (twice the largest improvement seen from noise
  alone in a scenario-off run that retrained every day, 0.157) and beats the seasonal naive.

## Full calendar runs (local, 2026-09-26)

| Run | Retrains | Dataset-drift days | Promotions |
|---|---|---|---|
| scenario off, 180 days | 14 | 14 (7.8%) | 0 |
| scenario on, 180 days | 62 | 58 | 0 |

- Steady growth: first drift and retrain on day 14; never promoted.
- Fast-growth cells (day 70): drift and retrain from day 72; never promoted.
- Holiday week (day 130, benign): drift and retrain from day 130 to day 157; never promoted.
- Day 179: live model v1, MAPE 14.9.

## Day-0 anchor to the earlier work

Source: capacity-forecasting at `8249252`. Its evidence (MAPE 14.51) reproduces exactly on
current libraries, but its features include the previous hour's traffic and rolling means
shifted by one hour: it is a 1-hour nowcast. The loop forecasts 7 days ahead, so the anchor is
a fresh run of the earlier generator (seed 42, 60 cells, 30 days, the last 20% of time as
test) with inputs at least 7 days old:

| | Earlier generator, 7 days ahead | This generator | Tolerance |
|---|---|---|---|
| LightGBM MAPE | 16.05 | 14.50 | 2 points |
| seasonal naive MAPE | 20.98 | 19.58 | 2 points |
| mean traffic, rural / suburban / urban | 2.645 / 7.148 / 13.324 | 2.660 / 6.962 / 13.316 | 5% |

The earlier generator's hourly random-walk drift has no bound in time; over 180 days it would
sit at its clip limits. It is replaced by a slow per-cell daily level, which makes these data
slightly easier to forecast than the earlier 30 days.

## Limits

- Growth here is smooth and multiplicative, which is exactly what lag features absorb; real
  demand shifts can be abrupt and shaped differently by hour.
