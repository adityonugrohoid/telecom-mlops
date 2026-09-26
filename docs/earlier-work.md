# Earlier work

This repo replaces eight earlier repos: six single-use-case projects, the portfolio that
collected them, and the framework they were built from. Each was a notebook-first project on
its own synthetic data, with no loop, no drift and no promotion gate. The generators and
models here are ported from them; this page records what came from where, and what the port
found.

Porting rule: each use case's day 0, with its drift calendar off, has to reproduce the earlier
repo's measured result within a stated tolerance, held by a test (`tests/test_anchor.py` in
each package). Reproducing those results is how the findings below came to light.

## The six use cases

### telecom-churn-prediction

- Source: [telecom-churn-prediction](https://github.com/adityonugrohoid/telecom-churn-prediction)
  at `8f3735c`.
- Ported: the structural churn model (a logit on QoE MOS, tenure, charges, tickets and
  contract, with an intercept for a 15% churn rate), the 17 raw and 27 engineered features,
  XGBoost with its hyperparameters, and the logistic regression baseline.
- Found:
  - The generator built its timestamps from the current clock (`pd.Timestamp.now()`), so no
    two runs produced the same data. Here every day is seeded by its date.
  - It solved the intercept again for every sample, which would erase any drift in the churn
    rate. Here it is solved once, on the scenario-off distribution.
  - Its own evidence has the logistic baseline ahead of XGBoost (AUROC 0.8727 against
    0.8548), while its description quotes XGBoost at 0.86. The anchor reproduces both, and
    the baseline still leads: the label is itself a logit on the raw inputs.

### telecom-root-cause-analysis

- Source: [telecom-root-cause-analysis](https://github.com/adityonugrohoid/telecom-root-cause-analysis)
  at `6bcf53b`.
- Ported: the alarm-cascade generator event by event, the 24 features, the two-class XGBoost
  ranker and the first-alarm baseline.
- Found:
  - The generator read the clock for its timestamps, as churn did.
  - Its evidence reports the first-alarm baseline at top-1 0.58. That figure comes from one
    100-incident test split: its generator puts 70% of roots on the first alarm (a fresh run
    at its seed over 5,000 incidents: 0.702), and that split happened to hold 58. The anchor
    is the distribution at scale plus the evidence setup.
  - About 0.03 of the gap between the evidence's model score (0.89) and this port's (0.93)
    is the move from xgboost 1.7 to 3.4: the earlier code on current libraries scores 0.92
    at the same seed.

### telecom-anomaly-detection

- Source: [telecom-anomaly-detection](https://github.com/adityonugrohoid/telecom-anomaly-detection)
  at `e2c311f`.
- Ported: the per-cell hourly profiles, the four anomaly signatures, the 16 features and the
  Isolation Forest, kept here as the baseline.
- Found:
  - It had no evidence file. A fresh run of its own notebook, on its own pinned versions,
    gives F1 0.698 and ROC AUC 0.969; that run is the anchor.
  - Its README reports precision 0.95, recall 0.38 and F1 0.70 together, which cannot hold
    at one threshold. The first two are the 2nd-percentile row of its threshold table; the
    F1 is its default threshold, where precision and recall are both 0.698.
  - An unsupervised detector cannot learn a new anomaly type from retraining. The loop here
    runs a supervised detector on triage labels (alerts reviewed plus a 20% audit) and keeps
    the Isolation Forest as the baseline.

### telecom-qoe-prediction

- Source: [telecom-qoe-prediction](https://github.com/adityonugrohoid/telecom-qoe-prediction)
  at `266b852`.
- Ported: the session model (device, app and radio KPIs to MOS), the 23 features, LightGBM
  with its hyperparameters, and the mean-MOS baseline.
- Found: the evidence's baseline MAE (0.546) sits 0.013 below what the same predictor scores
  over all of its generator's sessions (0.559): split noise, noted and allowed for in the
  anchor's tolerance.

### telecom-capacity-forecasting

- Source: [telecom-capacity-forecasting](https://github.com/adityonugrohoid/telecom-capacity-forecasting)
  at `8249252`.
- Ported: the diurnal and weekly traffic model with bursts and outages, LightGBM with its
  forecaster's hyperparameters, and chronological splits.
- Found:
  - Its reported MAPE of 14.51 reproduces exactly, but its features include the previous
    hour's traffic and rolling means shifted by one hour: it is a 1-hour nowcast. The loop
    forecasts 7 days ahead; on the earlier generator's data that scores 16.05, against 20.98
    for the seasonal naive forecast, and that is the anchor.
  - Its model base class defaults to a random train/test split, which leaks the future into
    a forecast. Only chronological splits are used here.
  - Its hourly random-walk drift has no bound in time and would sit at its clip limits over
    a 180-day calendar. A slow, bounded per-cell daily level replaces it.

### telecom-network-optimization

- Source: [telecom-network-optimization](https://github.com/adityonugrohoid/telecom-network-optimization)
  at `360a9fa`.
- Ported: the random initial state, the five actions' physics and the reward.
- Found:
  - Its reported result, a Q-learning agent 61% better than random (60.2% in a fresh run of
    its notebook), comes from a replay environment. The next state is the recorded next row
    whatever action the agent takes, the reward depends on matching the recorded (randomly
    drawn) action, and every episode is the same 50 rows. The agent memorised 50 actions,
    which is why its reward has zero variance. The result is recorded, not anchored to.
  - Load entered neither the transitions nor the reward.
  - Here every step is simulated, with load couplings each stated with its reason, and the
    anchor is the earlier generator's one-step dynamics. On that environment a Q-learning
    policy scores 0.30 to 0.33 per episode against 0.20 to 0.22 for a rule-based policy and
    0.22 to 0.24 for random.

## Shared

- [telecom-ml-portfolio](https://github.com/adityonugrohoid/telecom-ml-portfolio) at
  `02840ff`: the umbrella that collected the six. Its role is taken by this repo's README and
  results.
- [telecom-ml-framework](https://github.com/adityonugrohoid/telecom-ml-framework) at `e0b1bd9`:
  the project template and generator base the six grew from. The base's shared radio helpers
  (SINR, throughput, congestion, latency, MOS) are copied into each use case here rather than
  shared, so a change to one use case's physics can never move another's results.
- All six pinned `numpy<2` and `xgboost<2` for an old SHAP incompatibility. Those pins are
  obsolete: this repo runs numpy 2.5, xgboost 3.4, lightgbm 4.7 and SHAP 0.52, and SHAP was
  re-run on the two-class root-cause model.
