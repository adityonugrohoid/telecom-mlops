# Synthetic data generators

Every use case in this repo runs on simulated data. Each one owns a generator that
produces one simulated day at a time, and a drift calendar (`scenario.yaml`) that changes
the data on known days. This document states the rules every generator follows and what
each use case simulates.

## Rules

1. One generator per use case, owned by that use case's package. No generator imports
   another use case's code. The radio helpers the earlier repos shared (SINR, SINR to
   throughput, congestion, latency, MOS) are copied into each use case on purpose: a
   change to one use case's physics must never move another's results.
2. Deterministic by date. `generate(day)` returns that simulated day's batch; its seed
   derives from (base seed, use case, day), as
   `numpy.random.default_rng([base_seed, usecase_id, day.toordinal()])`. Nothing reads the
   clock. The earlier churn and root-cause generators built timestamps from
   `pd.Timestamp.now()`; that reproducibility bug is not ported.
3. Day 0 anchors to the earlier work. With the scenario off, the day-0 distribution and
   the model's result stay within a stated tolerance of the earlier repo's measured
   evidence (its `evidence/*.json`). A test holds each anchor. Anomaly and netopt have no
   evidence file (and the anomaly README's precision 0.95, recall 0.38 and F1 0.70 cannot
   hold at one threshold), so their anchor is a fresh run of the earlier repo's code at its
   last commit and default seed, in a separate environment on that repo's own pinned
   versions, with the numbers, versions and commit recorded. When an evidence figure proves
   to be a small-split artifact (root cause's first-alarm 0.58 came from one 100-incident
   test split), the anchor is the generator's distribution at scale against a fresh run of
   the earlier generator, plus the evidence setup within a stated tolerance; the model card
   states the earlier figure's origin and the spread across split seeds.
4. Ground truth is recorded. Each batch returns its data plus a manifest: the scenario
   events active that day and their strength. The manifest never reaches the model; the
   report uses it to show when drift was injected against when the loop detected it
   (detection delay).
5. Labels arrive late where they would in operations. Each use case declares a label
   delay; the core releases a day's labels only after it. Training and promotion use
   released labels only.
6. Scenarios are data. Each use case has `scenario.yaml`: a list of events, each with
   `name`, `day`, `ramp_days`, `kind` (covariate, concept, prior, new_class, dynamics or
   trend), the parameters it moves and their end values, `benign` (the rule 9 event) and
   an optional `duration_days` (a temporary event such as the capacity holiday week;
   without it the change stays). The scenario mechanics in the core (`Scenario.value`,
   `Event.strength`, `rng_for_day`) decide every use case's data, so a change to them must
   show every day-0 anchor unchanged. Default calendar: 180 simulated days from 2026-01-01.
7. Timing: after each real event, leave at least the label delay plus the evaluation
   window plus 30 days before the next event, so the loop can learn and promote before the
   calendar moves on.
8. Promotion margins. Each use case's promotion test requires the new model to win by a
   margin, never by any amount. The margin is set from a measured scenario-off run that
   retrains every day: at least twice the largest gain seen there from noise alone. With
   the scenario off, a full calendar must promote nothing; with it on, the benign event
   must promote nothing. Each use case's model card records these runs.

   Drift checks run on a frame whose rows are close to independent: one row per customer,
   per incident (not per alarm event), per cell per day (not per cell-hour), per session,
   per environment episode. A value that is constant within a day, such as a daily count,
   is a use-case metric, never a drift column. The scenario-off run reports its drift-flag
   days, which must stay at or below 10% of the calendar, or detection delays mean nothing.
   Input drift triggers a retrain on dataset drift (at least half the monitored columns
   drifted) unless the use case states another rule and its reason.
9. Every calendar holds one benign event: a real input shift that does not hurt the
   model. The loop should detect it, may retrain, and should not promote. That shows the
   promotion gate working.
10. Invented names only. No operator, vendor, city, region or holiday name in data,
    scenarios or reports.

## Per use case

### churn

- Batch: 1,000 customers per day. Label delay: 30 days (churn is known after the
  observation window).
- Model of the data: the earlier structural model, a logit on QoE MOS, tenure, charges,
  tickets and contract, with an intercept solved for a 15% churn rate. The coefficients and
  input distributions are scenario parameters.
- Events: day 60, price rise: month-to-month charges +15% over 14 days (covariate) and the
  charge coefficient 0.4 to 2.0 (concept; an end value of 0.9 was measured too small to
  learn). Day 150, benign: 5G share 0.4 to 0.7 over 20 days, with no change to the label
  mechanism.
- Promotion: AUROC and Brier score both no worse than the live model, and one better by a
  margin: AUROC +0.012 or Brier -0.0025 (twice the largest noise gains measured, AUROC
  +0.0057 and Brier -0.0012), on labels released in the last 14 days.
- Timing: the holdout trains on labels released before the evaluation window, so a new
  model first sees post-drift labels at about event day + 30 (label delay) + 14 (window).
  The price rise at day 60 is learnable from about day 104; the benign event sits at day
  150 so the two do not overlap.
- Input drift: any single monitored column drifting triggers a retrain, not dataset drift,
  because the price rise moves one input (charges) out of thirteen. The scenario-off run
  retrained on 3 of 180 days.

### root-cause

- Batch: incidents per day drawn from Poisson(30), about 20 alarm events each, 50 cells.
  Label delay: 1 to 7 days per incident (ticket closure), drawn per incident.
- Model of the data: the earlier cascade model (a root event, cascading alarms with
  severity decay, five event types).
- Events: day 30, firmware rollout: in 50% of config_error incidents the first alarm is no
  longer the root (concept). Day 90, a new root cause class `power_supply` at 10% of
  incidents (new_class): its root alarm arrives late (sequence position 2 to 4, after the
  first symptoms), with small KPI deltas (about 30% of a normal root), 3 to 8 affected
  cells and severity major. Its one-hot column exists from day 0 and is 0 before day 90.
  Day 150, benign: the incident rate rises 40% (30 to 42 per day) with unchanged mechanics.
- Volume and noise: at 15 incidents per day and a 30% firmware share, the noise from
  retraining alone (top-1 +0.047) swamped the firmware event's real gain (+0.031). Doubling
  the volume halved the noise (+0.0213), and the 50% share raised the firmware gain to
  about +0.06.
- Drift: checked on one row per incident (strongest KPI impact, spread, mean lag, first
  alarm type and severity), since the 20 events of an incident share one impact. Retrain
  triggers: dataset drift on that frame; any event type the live model never saw
  (`unknown_signature_share` above 0, the new-class signal); top-1 falling 0.05 below its
  value at promotion; incidents per day over the evaluation window moving more than 20%
  from their value at promotion (the daily count is a metric, not a drift column).
- Promotion: top-1 hit rate beats the live model by 0.043 (twice the largest noise gain) on
  incidents released in the last 21 days.

### anomaly

- Batch: 50 cells x 24 hours per day. Label delay: 1 day (network operations triage).
- Model of the data: the earlier per-cell hourly profiles with injected anomalies
  (traffic_spike, sinr_drop, latency_surge, throughput_collapse; 5% of hours).
- Events: day 50, capacity upgrade on 20% of cells: throughput baseline +30%, latency
  baseline -20% (covariate; raises false alerts). Day 110, a new anomaly type
  `intermittent_outage` (short repeated drops) at 1% (new_class). Day 150, benign: the
  weekend traffic profile is smoothed.
- Promotion: F1 on labels released in the last 14 days beats the live model, and the alert
  rate on normal hours does not rise.

### qoe

- Batch: 1,500 sessions per day. Label delay: 0 (MOS is measured with the session).
- Model of the data: the earlier session model (app mix, device class, radio KPIs to MOS).
- Events: day 40, device mix: high-end share 0.3 to 0.5 over 30 days (covariate). Day 90,
  a video codec change: the same throughput gives higher MOS for video (concept). Day 150,
  benign: gaming share +5 points.
- Promotion: MAE on the last 7 days beats the live model.

### capacity

- Batch: 60 cells x 24 hours per day. Label delay: 0 (actual traffic arrives each hour);
  forecasts are made 7 days ahead.
- Model of the data: the earlier diurnal x weekly x growth model with special events.
- Events: continuous growth of 2% per month (trend). Day 70, 15 cells move to 5% per month
  growth (trend: new demand in their area). Day 130, a holiday week: traffic x1.8 on 30% of
  cells for 7 days, then back (benign if the model is not retrained on it).
- Promotion: MAPE on the last 14 days beats the live model and the seasonal naive forecast.
  Chronological splits only.

### netopt

- Batch: one environment version per day: the earlier generator's transition dynamics and
  reward (load, SINR, interference, throughput, latency; five actions), with scenario
  parameters on the dynamics. Label delay: none; the policy is judged by rollouts.
- Events: day 60, mean load +20% over 21 days (dynamics). Day 120, a neighbour outage on
  10% of cells: interference and load rise there, and load_balance is worth more
  (dynamics). Day 160, benign: measurement noise on SINR doubles.
- Evaluation: 50 fixed-seed episodes per environment version; the live policy, the new
  policy and a static rule-based baseline all run the same episodes.
- Retraining continues from the live Q-table (warm start).
- Promotion: mean episode reward beats the live policy on the fixed episodes.

## Tests every generator carries

- The same (seed, day) gives identical data; different days differ.
- Scenario off: day 0 and day 179 come from the same distribution.
- Each event moves its parameters by the stated amount on its ramp, and nothing else.
- Day 0 anchor: distribution and model result within tolerance of the earlier evidence.
- The manifest lists exactly the events active that day.
- No clock reads: a test moves the clock and checks the output is unchanged.
