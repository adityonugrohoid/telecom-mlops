# Model card: root cause

**Simulated data only.** Every incident, alarm and KPI reading comes from the synthetic
generator in `telecom_ml_root_cause.generator`. No real network data is used, and the
results say nothing about any real network.

## Task

Given the 20 alarm events of a network incident, rank them by how likely each is the root
cause. The metric is top-1: the share of incidents whose true root the model ranks first.

## Data

- Incidents per day drawn from Poisson(30), each a cascade of 20 alarm events over 50 cells.
- Each event: position in the cascade, time lag, cell, event type, alarm severity, affected
  cells, and SINR, throughput and latency deltas.
- The root event carries the largest KPI impact and is usually, not always, first; cascade
  alarms decay in severity and impact with distance from it.
- Label delay: 1 to 7 days per incident, drawn per incident (the ticket closes on its root
  cause). The loop trains and judges only on closed tickets.

## Models

| | Model | Features |
|---|---|---|
| live and candidate | XGBoost, `multi:softprob` over the two classes of `is_root_cause` (depth 6, 200 trees, learning rate 0.1) | 24: raw event columns, one-hot type and severity, temporal and cascade features |
| baseline | first alarm: always rank position 0 first | position only |

Candidates retrain from scratch on incidents released in the 30 days before the evaluation
window. The evaluation window is the incidents released in the last 21 days.

## Drift calendar

180 simulated days from 2026-01-01 (`scenario.yaml`):

| Day | Event | Kind | Change |
|---|---|---|---|
| 30 | firmware rollout | concept | in 50% of config_error incidents the root is no longer the first alarm |
| 90 | power_supply faults | new class | a root cause the live model never saw, 10% of incidents: late in the cascade, small KPI dip, 3 to 8 cells |
| 150 | volume growth (benign) | prior | incidents per day 30 to 42, same mechanics |

## Loop rules

- Drift is checked on one row per incident, because the 20 events of an incident share one
  impact and are not independent samples.
- Retrain on dataset drift of that frame, on any event type the live model never saw, when
  top-1 falls 0.05 below its value at promotion, or when incidents per day over the
  evaluation window move more than 20% from their value at promotion.
- Promote when top-1 beats live by 0.043: twice the largest gain seen from noise alone in a
  scenario-off run that retrained every day (+0.0213).

## Full calendar runs (local, 2026-09-26)

| Run | Retrains | Dataset-drift days | Promotions |
|---|---|---|---|
| scenario off, 180 days | 0 | 0 | 0 |
| scenario on, 180 days | 59 | 1 | 2, on days 64 and 115 |

- Firmware rollout (day 30): top-1 fell and triggered a retrain on day 47, once post-event
  tickets had closed; promoted on day 64 (top-1 +0.050).
- power_supply (day 90): unseen event type from day 91; promoted on day 115 (+0.091).
- Volume growth (day 150, benign): volume trigger from day 164, retrained, never promoted.
- Day 179: live model v3, top-1 0.894, top-3 1.0.

At 15 incidents per day and a 30% firmware share, the gain from retraining on noise alone
(+0.047) was larger than the firmware event's real gain (+0.031). Doubling the volume halved
the noise and the 50% share made the event learnable.

## Day-0 anchor to the earlier work

Source: root-cause-analysis at `6bcf53b`, `evidence/baseline_metrics.json` (2026-09-19): 500
incidents, incident-grouped 80/20 split with seed 42, model top-1 0.89, first-alarm 0.58.

The earlier first-alarm figure of 0.58 came from that one 100-incident test split. The
earlier generator puts 70% of roots at the first alarm (a fresh run at its seed, 5,000
incidents: 0.702); that split happened to hold 58. The anchor is therefore:

| | Earlier | This generator | Tolerance |
|---|---|---|---|
| roots at the first alarm, 5,000 incidents | 0.702 (fresh run) | 0.699 | 0.02 |
| KPI column means, 5,000 incidents | fresh run | all within 1% | 3% |
| model top-1, evidence setup, seed 42 | 0.89 | 0.93 | 0.05 |

Over eight split seeds of the evidence setup, the model scores top-1 0.94 +- 0.02 and the
first-alarm baseline 0.73 +- 0.03. The earlier code on current libraries scores 0.925 +-
0.018 on its own data (0.92 at seed 42), so about 0.03 of the gap to the published 0.89 is the
move from xgboost 1.7 to 3.4.

## Limits

- Every incident has exactly one root and exactly 20 alarms; real incidents vary in both.
- Twenty events per incident make the model's top-3 trivially perfect here.
