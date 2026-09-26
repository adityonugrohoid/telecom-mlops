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
# # QoE: dated sessions, their drift calendar and the day-0 MOS model
#
# **All data here is simulated.** Sessions, radio readings and the MOS each user reports come
# from the synthetic generator in `telecom_ml_qoe.generator`; no real user data is used.
#
# The notebook shows:
#
# 1. one simulated day of sessions
# 2. the drift calendar
# 3. what each event does to MOS
# 4. the day-0 model against the mean-MOS baseline (the anchor to the earlier work)
# 5. how a model that is not retrained fares through the calendar

# %%
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from telecom_ml_qoe import generator
from telecom_ml_qoe.features import TARGET
from telecom_ml_qoe.model import fit_baseline, fit_model
from telecom_ml_qoe.usecase import QoEUseCase

usecase = QoEUseCase()
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
# ## 1. One simulated day
#
# 1,500 sessions a day. MOS is measured with the session, so each label is released at once.

# %%
batch = generator.generate(scenario.start, scenario)
print(f"{len(batch.data):,} sessions on {batch.day}; mean MOS {batch.data[TARGET].mean():.2f}")
batch.data.head()

# %% [markdown]
# ## 2. The drift calendar
#
# Day 40, the benign event: the share of high-end devices grows from 30% to 50% over 30 days.
# MOS saturates with throughput, so the model does not get these sessions wrong. Day 90: a
# video codec change gives the same picture from half the bits (a concept change). Day 150:
# cloud gaming arrives at 10% of sessions, an app type that needs four times the throughput
# and is twice as latency-sensitive as local gaming.

# %%
calendar = pd.DataFrame([generator.params_on(day(i), scenario) for i in range(scenario.days)])
fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharex=True)
for ax, (param, title) in zip(
    axes,
    [
        ("high_device_share", "high-end device share (benign)"),
        ("video_codec_gain", "video codec gain (concept)"),
        ("cloud_gaming_share", "cloud gaming share (new class)"),
    ],
    strict=True,
):
    ax.plot(calendar.index, calendar[param])
    ax.set_title(title)
    ax.set_xlabel("simulated day")
fig.tight_layout()

# %% [markdown]
# ## 3. What each event does to MOS
#
# Mean MOS by app type, a week before each event and a week after it has fully ramped in.

# %%
windows = {"days 30-36": 30, "days 80-86": 80, "days 95-101": 95, "days 155-161": 155}
pd.DataFrame(
    {
        label: days(start, 7, scenario).groupby("app_type")[TARGET].mean()
        for label, start in windows.items()
    }
).round(2)

# %% [markdown]
# ## 4. Day-0 model against the mean-MOS baseline
#
# 10,000 scenario-off sessions, split 80/20 with seed 42 as the earlier work did. The earlier
# evidence: LightGBM MAE 0.358, mean baseline 0.546.

# %%
data = days(0, 7, off).iloc[:10_000]
train, test = train_test_split(data.index, test_size=0.2, random_state=42)
model, baseline = fit_model(data.loc[train], seed=42), fit_baseline(data.loc[train])
truth = data.loc[test, TARGET].to_numpy()
pd.Series(
    {
        "mean baseline": float(np.mean(np.abs(baseline.predict(data.loc[test]) - truth))),
        "lightgbm": float(np.mean(np.abs(model.predict(data.loc[test]) - truth))),
    },
    name="MAE",
).round(4)

# %% [markdown]
# ## 5. A model that is not retrained
#
# The day-0 model scored on 7-day windows through the calendar.

# %%
day0 = usecase.fit(days(-14, 14, scenario), None)
decay = pd.DataFrame(
    [
        {"window start": start, **usecase.score(day0, days(start, 7, scenario))}
        for start in range(0, 180 - 7, 14)
    ]
).set_index("window start")
decay["mae"].plot(figsize=(10, 3), title="MAE of the day-0 model")
plt.tight_layout()
