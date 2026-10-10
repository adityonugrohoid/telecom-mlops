# telecom-mlops

Six telecom machine-learning use cases (churn, root cause, anomaly detection, QoE, capacity
forecasting and network optimization) on one MLOps pipeline that runs a daily drift loop:
validate the day's data, detect drift, retrain, and promote a new model only when it beats
the live one by more than noise. The data and environments are simulated, each use case on
its own generator with a scripted drift calendar; the automation is real. It is for anyone
who wants to see a model's life after deployment end to end, on data they can rebuild
exactly. What sets it apart is that every result says what the loop did, including where it
learned nothing, refused a worse model, or correctly left a model alone.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12 or newer if you do not have
it).

```bash
git clone https://github.com/adityonugrohoid/telecom-mlops.git
cd telecom-mlops
uv sync --locked
uv run tml list
uv run tml loop root-cause --from 2026-01-01 --days 70 --state /tmp/telecom-mlops-state
uv run tml report --state /tmp/telecom-mlops-state --out /tmp/telecom-mlops-summary.md
```

This replays 70 simulated days of the root-cause use case in about a minute. On day 30 a
firmware rollout changes which alarm is the root in half of one fault type; the loop starts
retraining on day 47, once enough tickets have closed, and promotes one model, on day 64,
which the summary marks **learned** (trained on data from after the change). Run state lives
only in the folder you pass to `--state`.

## How the loop works

Every use case runs the same six stages for each simulated day:

1. **Generate** the day's batch from the use case's generator and drift calendar, seeded by
   the date, so any day can be rebuilt exactly.
2. **Validate** it against the use case's schema (pandera).
3. **Check drift** against the live model's reference data (Evidently), on rows that are close
   to independent: one per customer, incident, cell-day, session or episode.
4. **Score** the live model on labels released so far. Labels arrive late where they would in
   operations: 30 days for churn, 1 to 7 days per incident for root cause, a day of triage for
   anomalies.
5. **Retrain** when drift or the use case's own score trigger fires.
6. **Promote** only if the candidate beats the live model by a margin set at twice the gain
   that retraining produced from noise alone, measured on a run with the calendar switched
   off. Every decision is logged, including "retrained, not promoted".

The pipeline (`packages/core`) holds only this machinery. Each use case owns its generator,
schema, features, model, baseline and calendar, and imports nothing from another, so tuning
one never moves another's results. Live models, per-day runs and the promotion log live in a
local MLflow store under the state folder, which is never committed.

## The moving split

Each day, a use case trains on older answers and tests on the most recent ones, and both
windows move forward one day with the calendar. Windows count by when an answer arrived, so a
use case whose answers come late (churn waits 30 days) learns from older data than one whose
answers come at once. The full picture, including why promotions lag the events behind them,
is in [docs/moving-split.md](docs/moving-split.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/split-day-110-dark.svg">
  <img alt="Train and test windows of the five holdout use cases on day 110, by answer release day and by the days the rows were generated." src="docs/figures/split-day-110-light.svg">
</picture>

## The six use cases

Results of the full calendar, 180 simulated days from 2026-01-01:
[results/calendar-2026-01-01-180-days.md](results/calendar-2026-01-01-180-days.md).

| Use case | What drifts | What the loop did |
|---|---|---|
| [churn](packages/churn/MODEL_CARD.md) | a price rise makes customers more sensitive to charges; benign: 5G share grows | learned the price rise (promoted six times from day 110); detected the benign shift, never promoted |
| [root cause](packages/root-cause/MODEL_CARD.md) | a firmware rollout moves the root alarm; a new fault class appears; benign: incident volume grows | learned both (promoted on days 64 and 115); detected the benign growth, never promoted |
| [anomaly](packages/anomaly/MODEL_CARD.md) | new demand congests some cells; a new anomaly type appears; benign: weekend traffic smooths | learned the new type (day 129); the day-63 promotion was a refresh on older data, and the demand growth was never learned |
| [qoe](packages/qoe/MODEL_CARD.md) | benign: more high-end devices; a video codec change; cloud gaming arrives | learned the codec (day 101) and cloud gaming (day 157, after a refresh on day 151); the benign shift was detected and retrained on, never promoted |
| [capacity](packages/capacity/MODEL_CARD.md) | steady and faster growth; benign: a holiday week | nothing promoted: the lag features already track growth, and the gate refused every model retrained on the holiday week |
| [netopt](packages/netopt/MODEL_CARD.md) | load grows; a neighbour outage; benign: noisier SINR readings | nothing detected or retrained: the events barely moved the live policy's reward |

## What the results show

Not every drift needs a new model, and the loop is judged on getting that right as much as on
learning. Across the six use cases the outcomes spread over four kinds:

- **Learned**: a candidate trained on post-event data won by the margin (churn, root cause,
  the anomaly outage, both QoE events).
- **Refresh**: a candidate won on recent data before the event reached its training window.
  The results summary checks every promotion's training rows and labels these, rather than
  crediting them to the event.
- **Refused**: retraining happened, and the gate kept a worse or no-better model out (the
  capacity holiday week, every benign event that was detected).
- **Left alone**: the event did not hurt the live model enough to act on (network
  optimization).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/events-to-promotions-dark.svg">
  <img alt="For each event with a promotion: the day it began, the wait until a model could learn it, and the days a model was promoted, filled for learned and hollow for refresh." src="docs/figures/events-to-promotions-light.svg">
</picture>

Two use cases state in their model cards why no promotion was expected. Each use case's day 0
is also tested against the measured result of the earlier repo it was ported from; what those
ports found is in [docs/earlier-work.md](docs/earlier-work.md). The generator rules are in
[docs/generators.md](docs/generators.md).

## Run the full calendar

```bash
uv run tml loop all --from 2026-01-01 --days 180 --state /tmp/telecom-mlops-full
uv run tml report --state /tmp/telecom-mlops-full --out /tmp/telecom-mlops-full.md
```

About 15 minutes on a 16-core machine. Use cases run one after another; running several at once
oversubscribes the gradient-boosting libraries' threads. A later run with the same `--state`
continues from the day the last one stopped.

## Repository layout

```
packages/
  core/          the pipeline: contract, runner, validate, drift, evaluators, registry, report, CLI
  churn/ root-cause/ anomaly/ qoe/ capacity/ netopt/
    src/telecom_ml_<name>/   generator, scenario.yaml, schema, features, model, baseline, usecase
    notebooks/               a report notebook per use case (jupytext-paired, run in CI)
    tests/                   generator rules, day-0 anchor, use-case binding
    MODEL_CARD.md
results/         committed summaries, chosen runs only
docs/            generators.md, earlier-work.md, moving-split.md, figures/ (drawn by a script)
```

## Tests and CI

```bash
uv run pytest -q
uv run pytest -q --nbmake packages/*/notebooks/*.ipynb
```

Every push runs ruff, strict mypy across all seven packages, the tests, the notebooks, and a
stateless replay of 14 simulated days for all six use cases.

## License

MIT, see [LICENSE](LICENSE).

## Author

Adityo Nugroho ([adityonugroho.com](https://adityonugroho.com)),
building with a Claude Code agentic workflow.
