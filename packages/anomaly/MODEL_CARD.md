# Model card: anomaly

**Simulated data only.** Every cell, hourly KPI and anomaly comes from the synthetic
generator in `telecom_ml_anomaly.generator`. No real network data is used, and the results
say nothing about any real network.

## Task

Flag the cell-hours that hold a network anomaly, from hourly KPIs of 50 cells.

## Data

- 50 cells x 24 hours per day: traffic, SINR, throughput, latency, packet loss, connected
  users and PRB utilization, each following the cell's type, area and the hour's congestion.
- 5% of cell-hours carry one of four injected anomaly signatures (traffic spike, SINR drop,
  latency surge, throughput collapse); each moves several KPIs at once.
- **Labels come from triage of alerts plus a 20% audit.** An hour is labelled only if the
  Isolation Forest or the live detector alerted on it, or if it falls in a random 20% audit
  sample. Labels are released one day later. Every other hour stays unlabelled and never
  enters training or scoring.

## Models

| | Model | Trained on |
|---|---|---|
| live and candidate | gradient-boosted classifier (scikit-learn HistGradientBoosting) on the 16 features plus the Isolation Forest score | triaged hours only |
| baseline | the earlier repo's Isolation Forest (200 trees, contamination 0.05) on the 16 features | every hour, no labels |

The Isolation Forest is refit on each 30-day training window. Every model is scored on the
audit sample of the last 14 days only: the one set of hours labelled regardless of what any
model alerted on.

The synthetic anomalies are cleanly separable: once labelled, the detector averages F1 0.98
on the audit sample before any event (days 0 to 49). Real faults overlap normal behaviour far
more.

An unsupervised detector cannot learn a new anomaly type from retraining. Measured on this
generator, the Isolation Forest's recall on the day-110 outage type was 0.024 before
retraining and 0.048 after. That is why the loop runs a supervised detector on triage labels,
and keeps the earlier unsupervised model as its baseline.

## Drift calendar

180 simulated days from 2026-01-01 (`scenario.yaml`):

| Day | Event | Kind | Change |
|---|---|---|---|
| 50 | demand growth | covariate | 20% of cells: traffic, users and latency x1.8, the normal baseline moving toward congestion |
| 110 | intermittent outages | new class | a fifth anomaly type at 1% of hours: traffic and users fall away while radio conditions look normal |
| 155 | weekend profile smoothed (benign) | covariate | the weekend traffic dip goes away |

## Loop rules

- Drift is checked on one row per cell per day (the mean of each KPI), with the KS test:
  the hours of one cell and day share its load and conditions, so they are not independent
  samples. Latency flags on its own on most days: it follows the hour's congestion, not the
  cell, so one day's 50 cell means form a tight cluster that differs from a reference mixing
  weekdays and weekends. It never reaches dataset drift on its own (1 of 7 columns).
- Retrain on dataset drift of that frame, when F1 falls 0.05 below its value at promotion,
  or when alerts on normal hours rise 0.005 above it.
- Promote when F1 beats live by 0.043 and the rate of alerts on normal hours rises by no more
  than 0.0038. Both margins are twice the largest change seen from noise alone in a
  scenario-off run at a 20% audit that retrained every day (F1 +0.0215, normal alerts
  +0.0019).

## Full calendar runs (local, 2026-09-26)

| Run | Retrains | Dataset-drift days | Promotions |
|---|---|---|---|
| scenario off, 180 days | 0 | 0 | 0 |
| scenario on, 180 days | 24 | 0 | 2, on days 63 and 129 |

- Demand growth (day 50): alerts on normal hours rose and triggered a retrain on day 54;
  promoted on day 63 (F1 +0.060, normal-hour alerts 1.17% to 0.38%).
- Intermittent outages (day 110): F1 fell and triggered a retrain on day 115; promoted on
  day 129 (F1 +0.044).
- Weekend smoothing (day 155, benign): no retrain, never promoted.
- Day 179: live model v3, F1 0.915, recall on the outage type 0.36.

With a 10% audit the noise was twice as large (F1 +0.036, margin 0.072). The outage was
learnable there (candidate recall on it 0.38 to 0.63) but its F1 gain of about 0.04 never
cleared that margin, so the audit was raised to 20%.

## Day-0 anchor to the earlier work

The earlier anomaly-detection repo has no evidence file. The anchor is a fresh run of its
code at `e2c311f`: its own notebook at seed 42, in a separate environment on its own pinned
versions (python 3.13.12, numpy 1.26.4, scikit-learn 1.9.1, pandas 3.0.6). Isolation Forest
trained and scored on all 36,000 cell-hours:

| | Earlier code, fresh run | This generator | Tolerance |
|---|---|---|---|
| F1 at contamination 0.05 | 0.698 | 0.724 | 0.05 |
| ROC AUC | 0.969 | 0.972 | 0.01 |

The earlier README reported precision 0.95, recall 0.38 and F1 0.70 together. The first two
are the 2nd-percentile row of its notebook's threshold table (F1 0.545 there); the F1 is its
default threshold, where precision and recall are both 0.698.

## Limits

- Anomalies are injected into single hours with fixed multi-KPI signatures, which makes them
  far easier to separate than real faults.
- The live detector at retraining time stands in for the detector that was live when an
  hour happened, when deciding which past hours were triaged.
