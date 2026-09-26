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
# # Anomaly: hourly cell KPIs, triage labels and the drift calendar
#
# **All data here is simulated.** Cells, hourly KPIs and anomalies come from the synthetic
# generator in `telecom_ml_anomaly.generator`; no real network data is used.
#
# The notebook shows:
#
# 1. one simulated day of 50 cells x 24 hours, and which hours ever get a label
# 2. the drift calendar
# 3. the source's unsupervised Isolation Forest (the baseline and the anchor to the earlier
#    work) against the triage-trained detector the loop runs
# 4. what the demand growth and the new outage type do to a detector that is not retrained

# %%
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from telecom_ml_anomaly import generator
from telecom_ml_anomaly.features import TARGET
from telecom_ml_anomaly.model import fit_forest, triaged
from telecom_ml_anomaly.usecase import AnomalyUseCase

usecase = AnomalyUseCase()
scenario = usecase.scenario
off = scenario.without_events()


def day(index: int):
    return scenario.start + timedelta(days=index)


def days(first: int, count: int, calendar) -> pd.DataFrame:
    return pd.concat(
        [generator.generate(day(first + i), calendar).data for i in range(count)],
        ignore_index=True,
    )


# %% [markdown]
# ## 1. One simulated day, and who looks at it
#
# 5% of cell-hours carry an injected anomaly. A label exists only where someone looked: the
# network operations team triages every hour a detector alerted on, plus a random 10% audit.
# Labels are released a day later. Every other hour stays unlabelled.

# %%
batch = generator.generate(scenario.start, scenario)
forest = fit_forest(days(-30, 30, scenario), seed=42)
looked_at = triaged(batch.data, forest, None)
print(f"{len(batch.data)} cell-hours, {batch.data[TARGET].sum()} anomalies")
print(f"labelled after triage: {looked_at.sum()} hours ({looked_at.mean():.1%})")
print(batch.data["anomaly_type"].value_counts().to_dict())

# %% [markdown]
# ## 2. The drift calendar
#
# Day 50: new demand reaches 20% of cells; their traffic, users and latency rise 1.8x, so the
# normal baseline moves toward what used to look like congestion. Day 110: a new anomaly
# type, intermittent outages, at 1% of hours. Day 155, the benign event: the weekend traffic
# dip is smoothed away.

# %%
calendar = pd.DataFrame([generator.params_on(day(i), scenario) for i in range(scenario.days)])
fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharex=True)
for ax, (param, title) in zip(
    axes,
    [
        ("grown_share", "share of cells with new demand"),
        ("outage_rate", "intermittent outage rate (new type)"),
        ("weekend_factor", "weekend traffic factor (benign)"),
    ],
    strict=True,
):
    ax.plot(calendar.index, calendar[param])
    ax.set_title(title)
    ax.set_xlabel("simulated day")
fig.tight_layout()

# %% [markdown]
# ## 3. The source detector against the loop's detector
#
# The anchor: the earlier repo's Isolation Forest, trained and scored on 30 scenario-off
# days (36,000 cell-hours), scored F1 0.698 and ROC AUC 0.969 in a fresh run of its code.
# The loop's detector is trained on triage labels only and scored here on the audit sample.

# %%
history, window = days(-30, 30, off), days(0, 14, off)
anchor_data = days(0, 30, off)
anchor_forest = fit_forest(anchor_data, seed=42)
detector = usecase.fit(history, None)
pd.DataFrame(
    {
        "source forest, all hours (anchor)": {
            "f1": f1_score(anchor_data[TARGET], anchor_forest.alerts(anchor_data)),
            "roc_auc": roc_auc_score(anchor_data[TARGET], -anchor_forest.score(anchor_data)),
        },
        "source forest, audit sample": {
            "f1": usecase.score(usecase.fit_baseline(history), window)["f1"]
        },
        "triage detector, audit sample": {"f1": usecase.score(detector, window)["f1"]},
    }
).round(3)

# %% [markdown]
# ## 4. What the events do to a detector that is not retrained
#
# The day-0 detector, scored on 14-day audit windows through the calendar.

# %%
rows = []
for start in range(0, 180 - 14, 14):
    scores = usecase.score(detector, days(start, 14, scenario))
    rows.append({"window start": start, **scores})
decay = pd.DataFrame(rows).set_index("window start")
decay[["f1", "normal_alert_rate", "new_type_recall"]].plot(subplots=True, figsize=(10, 6))
plt.tight_layout()
