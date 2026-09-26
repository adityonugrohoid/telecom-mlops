**All data and environments in this summary are simulated.** Every use case runs on a synthetic generator and a scripted drift calendar; the numbers say nothing about any real network, customer or operator.

# Results

Learned: the candidate's training data held rows from after the event began. Refresh: it did not, and won on recent data alone.

## anomaly

180 simulated days from 2026-01-01; 24 retrains, 2 promotions, dataset drift on 0 days.

| Event | Day | Kind | First dataset drift | First retrain | Promotions |
|---|---|---|---|---|---|
| demand_growth | 50 | covariate | none | 54 | 63 (refresh) |
| intermittent_outages | 110 | new_class | none | 115 | 129 (learned) |
| weekend_profile_smoothed (benign) | 155 | covariate | none | none | none |

Benign event: not detected, 0 promotions.

Day 179: live model v3, f1 0.9153, precision 0.9454, recall 0.8872, normal_alert_rate 0.0031, new_type_recall 0.3636, audited_hours 3380.
Details: packages/anomaly/MODEL_CARD.md.

## capacity

180 simulated days from 2026-01-01; 62 retrains, 0 promotions, dataset drift on 58 days.

| Event | Day | Kind | First dataset drift | First retrain | Promotions |
|---|---|---|---|---|---|
| steady_growth | 0 | trend | 14 | 14 | none |
| fast_growth_cells | 70 | trend | 72 | 72 | none |
| holiday_week (benign) | 130 | covariate | 130 | 130 | none |

Benign event: detected, 0 promotions.

No promotion is expected for the real events here. A forecaster on same-hour lags tracks growth through its lags, so the growth events are detected and may retrain but a new model is not expected to win. The use case shows the other side: models retrained on the holiday week forecast worse after it, and the gate refuses them.

Day 179: live model v1, mape 14.8643.
Details: packages/capacity/MODEL_CARD.md.

## churn

180 simulated days from 2026-01-01; 89 retrains, 6 promotions, dataset drift on 0 days.

| Event | Day | Kind | First dataset drift | First retrain | Promotions |
|---|---|---|---|---|---|
| price_rise_charges + price_rise_sensitivity | 60 | covariate + concept | none | 65 | 110 (learned), 114 (learned), 117 (learned), 122 (learned), 127 (learned), 140 (learned) |
| fiveg_share_growth (benign) | 150 | covariate | none | 154 | none |

Benign event: detected, 0 promotions.

Day 179: live model v7, auroc 0.8790, brier 0.1027, churn_rate 0.2067, predicted_rate 0.2038.
Details: packages/churn/MODEL_CARD.md.

## netopt

180 simulated days from 2026-01-01; 0 retrains, 0 promotions, dataset drift on 0 days.

| Event | Day | Kind | First dataset drift | First retrain | Promotions |
|---|---|---|---|---|---|
| load_growth | 60 | dynamics | none | none | none |
| neighbour_outage | 120 | dynamics | none | none | none |
| sinr_measurement_noise (benign) | 160 | dynamics | none | none | none |

Benign event: not detected, 0 promotions.

No promotion is expected for the real events here. Tested before building, warm-start retraining did not beat the live policy by twice its own noise on either real event, with two learner settings each set once. In the loop the events moved the live reward by at most 0.013, inside the retrain trigger, so nothing was retrained or promoted.

Day 179: live model v1, reward 0.3004, reward_sd 0.1483.
Details: packages/netopt/MODEL_CARD.md.

## qoe

180 simulated days from 2026-01-01; 38 retrains, 3 promotions, dataset drift on 31 days.

| Event | Day | Kind | First dataset drift | First retrain | Promotions |
|---|---|---|---|---|---|
| high_end_device_growth (benign) | 40 | covariate | 58 | 58 | none |
| video_codec_change | 90 | concept | 91 | 91 | 101 (learned) |
| cloud_gaming_launch | 150 | new_class | none | 151 | 151 (refresh), 157 (learned) |

Benign event: detected, 0 promotions.

Day 179: live model v4, mae 0.3696, rmse 0.4586.
Details: packages/qoe/MODEL_CARD.md.

## root-cause

180 simulated days from 2026-01-01; 59 retrains, 2 promotions, dataset drift on 1 day.

| Event | Day | Kind | First dataset drift | First retrain | Promotions |
|---|---|---|---|---|---|
| firmware_rollout | 30 | concept | 47 | 47 | 64 (learned) |
| power_supply_faults | 90 | new_class | none | 91 | 115 (learned) |
| incident_volume_growth (benign) | 150 | prior | none | 164 | none |

Benign event: detected, 0 promotions.

Day 179: live model v3, top1 0.8943, top3 1.0000, mrr 0.9454, unknown_signature_share 0.0000, incidents_per_day 42.7390.
Details: packages/root-cause/MODEL_CARD.md.
