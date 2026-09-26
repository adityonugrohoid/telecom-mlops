# Model card: churn

**Simulated data only.** Every customer, charge, network reading and churn label comes
from the synthetic generator in `telecom_ml_churn.generator`. No real subscriber data is
used, and the results say nothing about any real network or customer base.

## Task

Predict whether a customer churns in the 30 days after their observation window. Binary
classification, scored as a probability.

## Data

- 1,000 simulated customers per day, each with a 30-day observation window ending that day.
- Inputs: tenure, contract, payment method, monthly charges, network type, device class,
  and averages of SINR, throughput, latency, packet loss and QoE MOS; support tickets and
  sessions.
- Label: a structural logit on QoE MOS, tenure, charges, tickets and contract, with an
  intercept solved once for a 15% churn rate on the scenario-off distribution.
- Label delay: 30 days. The loop trains and judges only on released labels.

## Models

| | Model | Features |
|---|---|---|
| live and candidate | XGBoost (depth 6, 200 trees, learning rate 0.1) | 27: raw plus temporal and interaction features |
| baseline | logistic regression with standard scaling | 17 raw features |

Candidates retrain from scratch on labels released in the 30 days before the evaluation
window. The evaluation window is the labels released in the last 14 days.

## Drift calendar

180 simulated days from 2026-01-01 (`scenario.yaml`):

| Day | Event | Kind | Change |
|---|---|---|---|
| 60 | price rise | covariate | month-to-month charges +15% over 14 days |
| 60 | price rise | concept | charge coefficient 0.4 to 2.0 over 14 days |
| 150 | 5G growth (benign) | covariate | 5G share 0.4 to 0.7 over 20 days |

## Loop rules

- Retrain when any monitored input drifts (Evidently, per column), AUROC falls 0.02 below
  its value at promotion, or the churn rate moves more than 0.03.
- Promote only when AUROC and Brier are both no worse than live, and one of them is better
  by its margin: AUROC +0.012 or Brier -0.0025. The margins are twice the largest gain seen
  from noise alone: a scenario-off run that retrained every day for 180 days found AUROC
  +0.0057 and Brier -0.0012 at most.

## Full calendar runs (local, 2026-09-26)

| Run | Retrains | Promotions |
|---|---|---|
| scenario off, 180 days | 3 | 0 |
| scenario on, 180 days | 89 | 6, on days 110, 114, 117, 122, 127 and 140 |

- Price rise (day 60): inputs drift from day 65. A candidate first sees post-event labels
  around day 104 (30-day label delay plus the 14-day window) and is first promoted on day 110.
- 5G growth (day 150, benign): drift from day 154, retrained, never promoted.
- Day 179: live model v7, AUROC 0.879, predicted churn rate 0.204 against 0.207 observed.

## Day-0 anchor to the earlier work

Source: churn-prediction at `8f3735c`, `evidence/baseline_metrics.json` (2026-09-19).
Ten scenario-off days (10,000 customers), stratified 80/20 split with seed 42:

| | Earlier evidence | This generator | Tolerance |
|---|---|---|---|
| test churn rate | 0.151 | 0.142 | 0.015 |
| logistic baseline AUROC | 0.8727 | 0.8752 | 0.02 |
| XGBoost AUROC | 0.8548 | 0.8542 | 0.02 |

The logistic baseline scores above XGBoost, as in the earlier work: the label is itself a
logit on the raw inputs, which a linear model fits directly.

## Limits

- The label mechanism is known and smooth; real churn is noisier and partly unobserved.
- Drift in the charges is detected against a reference that lags by the label delay plus
  the evaluation window, so the loop retrains on most days while the inputs sit away from it.
