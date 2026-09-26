# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Root cause: the dated alarm cascades, their drift calendar and the day-0 ranker
#
# **All data here is simulated.** Incidents, alarms and KPI readings come from the synthetic
# generator in `telecom_ml_root_cause.generator`; no real network data is used.
#
# Each incident is a cascade of 20 alarm events from one root event. The model ranks the
# events of an incident by how likely each is the root; the metric is the share of
# incidents whose true root it ranks first (top-1). The notebook shows:
#
# 1. one simulated day of incidents and its per-incident label delay
# 2. the drift calendar
# 3. what the firmware rollout and the new power_supply class change
# 4. the day-0 ranker against the first-alarm baseline
# 5. what drives the ranker's root score (SHAP)

# %%
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.model_selection import GroupShuffleSplit
from telecom_ml_root_cause import generator
from telecom_ml_root_cause.baseline import FirstAlarm
from telecom_ml_root_cause.features import INCIDENT, TARGET, engineered_features
from telecom_ml_root_cause.model import fit_model
from telecom_ml_root_cause.usecase import RootCauseUseCase, top_k

usecase = RootCauseUseCase()
scenario = usecase.scenario
off = scenario.without_events()


def day(index: int):
    return scenario.start + timedelta(days=index)


def days(first: int, count: int, calendar) -> pd.DataFrame:
    return pd.concat(
        [generator.generate(day(first + i), calendar).data for i in range(count)],
        ignore_index=True,
    )


def roots(data: pd.DataFrame) -> pd.DataFrame:
    return data[data[TARGET] == 1]


# %% [markdown]
# ## 1. One simulated day
#
# Incidents per day follow a Poisson rate. Each incident's label (the ticket closing on its
# root cause) is released 1 to 7 days later; the loop only learns from closed tickets.

# %%
batch = generator.generate(scenario.start, scenario)
delays = (
    pd.to_datetime(batch.data.groupby(INCIDENT)["label_released_on"].first())
    - pd.Timestamp(batch.day)
).dt.days
print(f"{batch.data[INCIDENT].nunique()} incidents, {len(batch.data)} alarm events on {batch.day}")
print(f"label delay in days: {delays.value_counts().sort_index().to_dict()}")
batch.data.head(8)

# %% [markdown]
# ## 2. The drift calendar
#
# Day 30, a firmware rollout: in 30% of config_error incidents the first alarm is no longer
# the root (a concept change). Day 90, a new root cause class, power_supply, at 10% of
# incidents: its root alarm arrives after the first symptoms, with a small KPI dip spread
# over several cells. Day 150, the benign event: incident volume rises 40% with the same
# mechanics.

# %%
calendar = pd.DataFrame([generator.params_on(day(i), scenario) for i in range(scenario.days)])
fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharex=True)
for ax, (param, title) in zip(
    axes,
    [
        ("config_error_late_root_share", "config_error with a late root (concept)"),
        ("power_supply_share", "power_supply share (new class)"),
        ("incident_rate", "incidents per day (benign)"),
    ],
    strict=True,
):
    ax.plot(calendar.index, calendar[param])
    ax.set_title(title)
    ax.set_xlabel("simulated day")
fig.tight_layout()

# %% [markdown]
# ## 3. What the events change
#
# Where the root sits in the cascade, by root cause type, before the firmware rollout and
# after the power_supply class arrives.

# %%
before, after = roots(days(0, 20, scenario)), roots(days(100, 20, scenario))
pd.DataFrame(
    {
        "root is first alarm, days 0-19": before.groupby("event_type")[
            "event_sequence_position"
        ].apply(lambda s: (s == 0).mean()),
        "root is first alarm, days 100-119": after.groupby("event_type")[
            "event_sequence_position"
        ].apply(lambda s: (s == 0).mean()),
    }
).round(2)

# %% [markdown]
# ## 4. Day-0 ranker against the first-alarm baseline
#
# 500 scenario-off incidents, split 80/20 by incident with seed 42 as the earlier work did.
# The earlier evidence reported top-1 0.89 for the model and 0.58 for the baseline; that
# baseline figure came from one 100-incident test split, while the generator puts about
# 70% of roots at the first alarm.

# %%
data = days(0, 30, off)
data = data[data[INCIDENT].isin(data[INCIDENT].drop_duplicates().iloc[:500])]
train, test = next(
    GroupShuffleSplit(1, test_size=0.2, random_state=42).split(data, groups=data[INCIDENT])
)
model = fit_model(data.iloc[train], seed=42)
holdout = data.iloc[test]
pd.Series(
    {
        "first-alarm baseline": top_k(FirstAlarm().root_score(holdout), holdout, 1),
        "xgboost ranker": top_k(model.root_score(holdout), holdout, 1),
    },
    name="top-1",
).round(3)

# %% [markdown]
# ## 5. What drives the root score
#
# SHAP values for the root class of the two-class `multi:softprob` model, on 1,000 test
# events.

# %%
features = engineered_features(holdout).iloc[:1000]
explanation = shap.TreeExplainer(model.classifier)(features)
shap.plots.beeswarm(explanation[:, :, 1], max_display=12, show=False)
plt.tight_layout()
