# Model card: QoE

**Simulated data only.** Every session, radio reading and reported MOS comes from the
synthetic generator in `telecom_ml_qoe.generator`. No real user data is used, and the results
say nothing about any real network or its users.

## Task

Predict the MOS (1 to 5) a user reports for a session, from its radio conditions, device and
app. Regression, scored by mean absolute error (MAE).

## Data

- 1,500 simulated sessions per day, each with network type, device class, app, SINR,
  throughput (capped by the device), latency (from the hour's congestion), packet loss,
  duration and data volume.
- MOS follows throughput, latency and loss per app, plus perception noise, user bias and
  content quality.
- Label delay: none. MOS is measured with the session.

## Models

| | Model | Features |
|---|---|---|
| live and candidate | LightGBM regressor (31 leaves, 200 trees, learning rate 0.05) | 23: session measurements, temporal, interaction and one-hot features |
| baseline | the training window's mean MOS | none |

Candidates retrain from scratch on the last 14 days; every model is scored on the last 7.

## Drift calendar

180 simulated days from 2026-01-01 (`scenario.yaml`):

| Day | Event | Kind | Change |
|---|---|---|---|
| 40 | high-end device growth (benign) | covariate | high-end share 0.3 to 0.5 over 30 days |
| 90 | video codec change | concept | video sessions get the MOS of three times their throughput |
| 150 | cloud gaming launch | new class | an app at 10% of sessions that needs four times the throughput and is twice as latency-sensitive as local gaming |

The device mix is harmless to the model by design of the MOS curve: MOS saturates with
throughput, so more high-end devices change nothing the model gets wrong (measured: MAE
0.3602 stale against 0.3598 retrained). The draw that picks cloud gaming sessions comes from
a child random stream, so adding the event leaves every other session unchanged.

## Loop rules

- Drift is checked on one row per session (sessions are independent), with Evidently's
  default test. `latency_ms` and `congestion_level` flag on their own on most days: both
  follow the day of the week, so one day differs from a 14-day reference mixing weekdays and
  weekends. Two of nine columns never reach dataset drift.
- Retrain on dataset drift, or when MAE rises 0.02 above its value at promotion.
- Promote when MAE beats live by 0.0074: twice the largest improvement seen from noise alone
  in a scenario-off run that retrained every day (0.0037). At a codec gain of 2 the event's
  real gain (0.011) was under twice that margin, so the gain was set to 3 (0.015).

## Full calendar runs (local, 2026-09-26)

| Run | Retrains | Dataset-drift days | Promotions |
|---|---|---|---|
| scenario off, 180 days | 0 | 0 | 0 |
| scenario on, 180 days | 38 | 31 | 3, on days 101, 151 and 157 |

- Device growth (day 40, benign): dataset drift and a retrain from day 58 as the ramp built
  up; never promoted.
- Codec change (day 90): retrained from day 91, promoted on day 101 (MAE -0.0079).
- Cloud gaming (day 150): retrained from day 151. The day-151 promotion (MAE -0.0080) came
  from a candidate whose training window ended on day 144, before cloud gaming appeared: a
  refresh on recent data, not the loop learning the new class. The learning promotion is day
  157 (MAE -0.128), the first candidate trained on cloud sessions.
- Day 179: live model v4, MAE 0.370.

## Day-0 anchor to the earlier work

Source: qoe-prediction at `266b852`, `evidence/baseline_metrics.json` (2026-09-19): 10,000
sessions, random 80/20 split with seed 42.

| | Earlier | This generator | Tolerance |
|---|---|---|---|
| MOS mean, 10,000 sessions | 3.874 (fresh run) | 3.877 | 0.02 |
| MOS standard deviation | 0.704 (fresh run) | 0.716 | 0.02 |
| LightGBM MAE, evidence setup | 0.358 | 0.368 | 0.02 |
| mean baseline MAE, evidence setup | 0.546 | 0.572 | 0.03 |

The earlier generator's mean baseline scores MAE 0.559 over all 10,000 of its sessions, so
the evidence split's 0.546 sits below it by split noise.

## Limits

- MOS here is a smooth function of a few measurements plus independent noise; real reported
  MOS depends on content, expectations and context the data does not carry.
