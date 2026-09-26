# telecom-mlops: synthetic data generators, the spec

Owned by the one-ring control session (owner's instruction 2026-09-26:
the control session handles the generators). The build session
implements this; the control session reviews every generator PR against
it. Copy it into the repo as `docs/generators.md` in the first use-case
PR. Questions on generators go to the control session, not the owner.

## Rules

1. One generator per use case, owned by that use case's package. No
   generator imports another use case's code. The radio helpers the old
   repos share (SINR, SINR to throughput, congestion, latency, MOS) are
   copied into each use case, on purpose: a change to one use case's
   physics must never move another's results (owner ruling).
2. Deterministic by date. `generate(day)` returns that simulated day's
   batch; its seed derives from (base seed, use case, day), for example
   `numpy.random.default_rng([base_seed, usecase_id, day.toordinal()])`.
   Nothing reads the clock. The old churn and root-cause generators build
   timestamps from `pd.Timestamp.now()`; that is a reproducibility bug
   and is not ported.
3. Day 0 anchors to the earlier work. With the scenario off, the day-0
   distribution and the model's result stay within a stated tolerance of
   the old repo's measured evidence (its evidence/*.json, 2026-09-19).
   A test holds each anchor. Anomaly and netopt have no evidence file
   (and the anomaly README's P 0.95, R 0.38, F1 0.70 cannot hold at one
   threshold): their anchor is a fresh run of the old repo's code at its
   HEAD commit and default seed, in a throwaway environment on that
   repo's own pins, with the numbers, versions and commit recorded
   (reviewed 2026-09-26).
4. Ground truth is recorded. Each batch returns its data plus a manifest:
   the scenario events active that day and their strength. The manifest
   never reaches the model; the report uses it to show when drift was
   injected against when the loop detected it (detection delay).
5. Labels arrive late where they would in operations. Each use case
   declares a label delay; the core releases a day's labels only after
   it. Training and promotion use released labels only.
6. Scenarios are data. Each use case has `scenario.yaml`: a list of
   events, each with `day`, `ramp_days`, `kind`
   (covariate | concept | prior | new_class | dynamics | trend), the
   parameters it moves and their end values, plus `name`, `benign`
   (the rule 7 event) and `duration_days` (a temporary event such as
   the capacity holiday week; empty means the change stays). The
   scenario mechanics in core (Scenario.value, Event.strength,
   rng_for_day) decide every use case's data, so they are frozen after
   #1: a change needs control review and a test that every day-0 anchor
   is unchanged. Default calendar: 180
   simulated days from 2026-01-01.
7. Timing: after each real event, leave at least label delay + eval
   window + 30 days before the next event, so the loop can learn and
   promote before the calendar moves on.
8. Promotion margins. Each use case's promotion test requires the new
   model to win by a margin, never by any amount. The margin is set from
   a measured scenario-off run: at least twice the largest noise gain
   seen there, recorded with the run. With the scenario off, a full
   calendar must promote nothing; with it on, the benign event must
   promote nothing. Both are checked on a full local run before the use
   case's PR merges, and the numbers go in the PR body.
9. Every calendar holds one benign event: a real input shift that does
   not hurt the model. The loop should detect it, may retrain, and
   should not promote. That shows the promotion gate working.
10. Invented names only. No operator, vendor, city, region or holiday
   name in data, scenarios or reports.

## Per use case

### churn

- Batch: 1,000 customers per day. Label delay: 30 days (churn is known
  after the observation window).
- Source: the old structural model (logit on QoE MOS, tenure, charges,
  tickets, contract; intercept solved for a 15% rate). Keep it; expose
  the coefficients and input distributions as scenario parameters.
- Events: day 60, price rise: month-to-month charges +15% over 14 days
  (covariate) and the charge coefficient 0.4 to 2.0 (concept; 0.9 was
  measured too small to learn, 2026-09-26). Day 150,
  benign: 5G share 0.4 to 0.7 over 20 days, no change to the label
  mechanism.
- Promotion: AUROC and Brier score both no worse than the live model,
  and one better by a margin: AUROC +0.005 or Brier -0.002 (measured
  noise maxima +0.002 and -0.0011), on released labels from the last
  14 days.
- Timing (reviewed after #3): the holdout trains on labels released
  before the evaluation window, so a new model first sees post-drift
  labels at about event day + 30 (label delay) + 14 (window). The price
  rise at day 60 is learnable from about day 104; the benign event sits
  at day 150 so the two do not overlap.

### root-cause

- Batch: 15 incidents per day, about 20 alarm events each, 50 cells.
  Label delay: 1 to 7 days per incident (ticket closure), drawn per
  incident.
- Source: the old cascade model (root event, cascading alarms with
  severity decay, five event types).
- Events: day 45, firmware rollout: in 30% of config_error incidents the
  first alarm is no longer the root (concept). Day 100, a new root cause
  class `power_supply`, unseen by the live model, at 10% of incidents
  (new_class); the unknown-signature share is the drift signal. Day 140,
  benign: incident volume +40% with unchanged mechanics.
- Promotion: top-1 hit rate beats the live model on released incidents
  from the last 21 days.

### anomaly

- Batch: 50 cells x 24 hours per day. Label delay: 1 day (NOC triage).
- Source: the old per-cell hourly profiles with injected anomalies
  (traffic_spike, sinr_drop, latency_surge, throughput_collapse, 5%).
- Events: day 50, capacity upgrade on 20% of cells: throughput baseline
  +30%, latency baseline -20% (covariate; raises false alerts). Day 110,
  a new anomaly type `intermittent_outage` (short repeated drops) at 1%
  (new_class). Day 150, benign: weekend traffic profile smoothed.
- Promotion: F1 on released labels from the last 14 days beats the live
  model, and the alert rate on normal hours does not rise.

### qoe

- Batch: 1,500 sessions per day. Label delay: 0 (MOS measured with the
  session).
- Source: the old session model (app mix, device class, radio KPIs to
  MOS).
- Events: day 40, device mix: high-end share 0.3 to 0.5 over 30 days
  (covariate). Day 90, a video codec change: the same throughput gives
  higher MOS for video (concept). Day 150, benign: gaming share +5
  points.
- Promotion: MAE on the last 7 days beats the live model.

### capacity

- Batch: 60 cells x 24 hours per day. Label delay: 0 (actual traffic
  arrives each hour); forecasts are made 7 days ahead.
- Source: the old diurnal x weekly x growth model with special events.
- Events: continuous growth 2% per month (trend). Day 70, 15 cells move
  to 5% per month growth (trend, new demand in their area). Day 130, a
  holiday week: traffic x1.8 on 30% of cells for 7 days, then back
  (benign if the model is not retrained on it).
- Promotion: MAPE on the last 14 days beats the live model and the
  seasonal naive forecast. Chronological splits only.

### netopt

- Batch: an environment version per day: the transition dynamics and
  reward of the old generator (load, SINR, interference, throughput,
  latency; five actions), with scenario parameters on the dynamics.
  Label delay: none; the policy is judged by rollouts.
- Events: day 60, mean load +20% over 21 days (dynamics). Day 120, a
  neighbour outage on 10% of cells: interference and load up there,
  load_balance worth more (dynamics). Day 160, benign: measurement noise
  on SINR doubles.
- Evaluation: 50 fixed-seed episodes per environment version; the live
  policy, the new policy and a static rule-based baseline all run the
  same episodes.
- Retraining continues from the live Q-table (warm start).
- Promotion: mean episode reward beats the live policy on the fixed
  episodes.

## Tests every generator carries

- Same (seed, day) gives identical data; different days differ.
- Scenario off: day 0 and day 179 come from the same distribution.
- Each event moves its parameter by the stated amount on its ramp, and
  nothing else.
- Day 0 anchor: distribution and model result within tolerance of the
  old evidence.
- The manifest lists exactly the events active that day.
- No clock reads (a test patches the clock and checks output is equal).
