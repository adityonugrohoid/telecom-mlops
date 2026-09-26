# Model card: netopt

**Simulated environments only.** `telecom_ml_netopt.generator` is a stylised cell-optimisation
simulator, not a network model: five actions move a cell's SINR, interference, throughput,
latency and load by simple rules, and the reward is a weighted KPI improvement. The results
say nothing about any real network.

## Task

Learn a policy that picks one of five actions (increase power, decrease power, adjust tilt,
balance load, do nothing) for a cell at each of 50 steps, to maximise the episode's total
reward.

## The earlier work, and why it is not the anchor

The earlier network-optimization repo reported a Q-learning agent 61% better than random. A
fresh run of its own notebook (`360a9fa`, seed 42) gives -1.5547 +- 0.0000 against random
-3.9054, 60.2%. That result is an artifact of its environment:

- it replays pre-recorded rows, so the next state is the recorded next row whatever action
  the agent takes;
- the reward is the recorded reward when the agent's action matches the recorded one (itself
  drawn at random) and half of it minus 0.1 otherwise;
- every episode restarts at row 0 and runs the same 50 rows.

The agent memorised 50 random actions, which is why its reward has zero variance. Here every
step is simulated from the earlier generator's action physics.

## Environment

- The earlier generator's random initial state, the five actions' effects and its reward,
  plus load couplings, each set once from its physical reason:
  - raising power in a loaded network raises interference at neighbours more: a power
    increase adds interference x (0.5 + load)^2;
  - load balancing matters more under congestion: its latency cut scales x (load / 0.55)^2;
  - congestion builds latency: each step adds 20 x (load - 0.5) ms;
  - a cell next to an outage carries interference and load +0.3, and moving load off it pays
    1.5 times more.
- The agent observes SINR (with 1 dB measurement noise), interference and latency, in five
  bins each. Load moves the dynamics underneath but is not observed.
- Each day is an environment version: its load level, outage share and SINR noise.

## Policies

| | Policy |
|---|---|
| live and candidate | tabular Q-learning (discount 0.95); step size per entry 1/(1 + visits since this training run), floored at 0.02; first training 2,000 episodes with exploration 1.0 decaying to 0.01; retraining warm-starts from the live table for 500 episodes, exploration 0.2 decaying to 0.01 |
| baseline | rules: decrease power when interference is in the top two bins; otherwise balance load when latency is in the top two bins; otherwise increase power when SINR is in the bottom two bins; otherwise adjust tilt |

Every policy is scored on the same 50 fixed-seed episodes of the day's environment.

On a day-0 environment, over those 50 episodes (episode standard deviation about 0.14):
Q-learning 0.30 to 0.33, rule-based 0.20 to 0.22, random 0.22 to 0.24. That is the honest
replacement for the earlier 60.2%.

## Drift calendar

180 simulated days from 2026-01-01 (`scenario.yaml`):

| Day | Event | Kind | Change |
|---|---|---|---|
| 60 | load growth | dynamics | mean load x1.2 over 21 days |
| 120 | neighbour outage | dynamics | 10% of episodes on a cell next to an outage |
| 160 | SINR measurement noise (benign) | dynamics | measurement noise on SINR 1 dB to 2 dB |

## Before building: can retraining help here?

Each real event was tested before the package was built: the day-0 policy against a
warm-started retrain on the event's environment, on 50 fixed episodes, with the bar at twice
the noise of retraining on an unchanged environment. Two learner settings were tried, each set
once:

| Attempt | Unchanged-environment retrain gain, 10 seeds | Load +20% | Outage | Benign |
|---|---|---|---|---|
| 1: constant step 0.1, retrain exploration 0.01 | -0.001 to -0.041, all negative | -0.067 | -0.015 | -0.008 |
| 2: step 1/(1 + visits) floored at 0.02, retrain exploration 0.2 to 0.01 | mean -0.011 (se 0.0045), largest 0.029 | +0.026 | -0.046 | +0.013 |

Neither passed: no event's gain reached twice the retraining noise (0.058 in attempt 2), and
the first attempt's retrains were worse than live even with nothing changed. The loop runs the
second setting. Under the promotion margin below, a retrain like the outage one (-0.046) would
be refused.

## Loop rules

- Drift is checked on one row per episode: 50 episodes of the rule-based policy on the day's
  environment, the mean observed bin of SINR, interference and latency, with the KS test.
- Retrain on dataset drift of that frame, or when the live reward falls 0.03 below its value
  at promotion.
- Promote when the mean episode reward beats live by 0.0824: twice the largest gain seen from
  noise alone in a scenario-off run that retrained every day (0.0412).

## Full calendar runs (local, 2026-09-26)

| Run | Retrains | Dataset-drift days | Promotions |
|---|---|---|---|
| scenario off, 180 days | 0 | 0 | 0 |
| scenario on, 180 days | 0 | 0 | 0 |

The drift events went undetected. The live policy's mean reward on the fixed episodes was
0.300 on days 0 to 59, 0.287 under load growth, 0.295 with the outage and 0.300 with the extra
measurement noise: the largest dip (0.013) is inside the 0.03 retrain trigger. The drift
frame did not move either: load is hidden from the agent, and the rule-based policy's visited
bins barely shift when load rises or an outage hits 10% of cells. So the loop detected
nothing, retrained nothing and promoted nothing, and correctly left alone a policy the events
did not hurt.

The gate's refusal of a worse retrain is shown in the acceptance experiment above (outage
-0.046, load +0.026 against a 0.058 bar), not in the loop, where no retrain was triggered.

## Day-0 anchor to the earlier work

The dynamics, not the artifact: per action, the mean one-step reward and state changes from a
fresh run of the earlier generator (seed 42, 2,000 episodes of 50 random actions), against
the same process here with every coupling at its neutral value. Every reward is within 0.0002
and every state change within 4%; the tolerance is 0.002 on reward and 10% (or 0.05) on each
change. The couplings are tested separately: at load 0.9 a power increase adds 1.96 times the
interference, at load 0.5 exactly as much as without coupling.

## Limits

- The dynamics are a handful of rules with independent noise; real cells interact, and a
  tabular agent on 125 observed states is a teaching example, not a controller.
