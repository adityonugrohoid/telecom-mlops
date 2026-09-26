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
# # Capacity: hourly cell traffic, growth and a 7-day forecast
#
# **All data here is simulated.** Cells and their hourly traffic come from the synthetic
# generator in `telecom_ml_capacity.generator`; no real network data is used.
#
# The notebook shows:
#
# 1. one simulated day and the history a 7-day-ahead forecast may use
# 2. the drift calendar
# 3. the day-0 forecaster against the seasonal naive forecast (the anchor to the earlier work)
# 4. why the growth events do not need a new model, and why the holiday week needs the gate

# %%
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
from telecom_ml_capacity import generator
from telecom_ml_capacity.model import TARGET, SeasonalNaive, fit_forecaster, mape_pct
from telecom_ml_capacity.usecase import CapacityUseCase

usecase = CapacityUseCase()
scenario = usecase.scenario
off = scenario.without_events()


def day(index: int):
    return scenario.start + timedelta(days=index)


def days(first: int, count: int, calendar) -> pd.DataFrame:
    return pd.concat(
        [generator.generate(day(first + i), calendar).data for i in range(count)],
        ignore_index=True,
    )


def mape(model, data: pd.DataFrame) -> float:
    return mape_pct(data[TARGET].to_numpy(), model.predict(data))


# %% [markdown]
# ## 1. One simulated day
#
# 60 cells x 24 hours. Each row carries the same cell and hour 7, 8 and 14 days earlier and
# the mean over days 7 to 13: everything a forecast made 7 days ahead may know.

# %%
batch = generator.generate(scenario.start, scenario)
batch.data[["cell_id", "timestamp", TARGET, "lag_7d", "lag_14d", "mean_same_hour_7_13d"]].head()

# %% [markdown]
# ## 2. The drift calendar
#
# Traffic grows 2% a month on every cell. From day 70, 15 cells grow at 5% a month (new
# demand in their area). Days 130 to 136, the benign event: a holiday week lifts 30% of cells
# to 1.8 times their traffic, then they return.

# %%
calendar = pd.DataFrame([generator.params_on(day(i), scenario) for i in range(scenario.days)])
fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharex=True)
for ax, (param, title) in zip(
    axes,
    [
        ("growth_level", "growth, every cell"),
        ("fast_growth_extra", "extra growth, 15 cells"),
        ("holiday_mult", "holiday week (benign)"),
    ],
    strict=True,
):
    ax.plot(calendar.index, calendar[param])
    ax.set_title(title)
    ax.set_xlabel("simulated day")
fig.tight_layout()

# %% [markdown]
# ## 3. Day-0 forecaster against the seasonal naive forecast
#
# 30 scenario-off days, the last 20% of time as test. The earlier work's figure (MAPE 14.5)
# came from a 1-hour-ahead nowcast; on its own data at 7 days ahead it scores 16.1 against
# the naive 21.0.

# %%
data = days(0, 30, off)
cut = data["timestamp"].sort_values().iloc[int(len(data) * 0.8)]
train, test = data[data["timestamp"] < cut], data[data["timestamp"] >= cut]
pd.Series(
    {
        "seasonal naive": mape(SeasonalNaive(), test),
        "lightgbm, 7 days ahead": mape(fit_forecaster(train, seed=42), test),
    },
    name="MAPE %",
).round(2)

# %% [markdown]
# ## 4. Growth needs no new model; the holiday week needs the gate
#
# A forecaster trained before the fast growth began scores almost the same on those cells
# as one retrained after it: its lags already carry the higher level. A model retrained on
# the holiday week, by contrast, forecasts worse once the holiday is over.

# %%
fast = set(generator.cell_profiles(scenario.base_seed).query("fast_growth")["cell_id"])
before_growth = fit_forecaster(days(42, 28, scenario), seed=42)
after_growth = fit_forecaster(days(88, 28, scenario), seed=42)
growth_test = days(116, 14, scenario)
on_fast = growth_test[growth_test["cell_id"].isin(fast)]
after_holiday = days(137, 14, scenario)
on_holiday = fit_forecaster(days(110, 28, scenario), seed=42)
pd.DataFrame(
    {
        "fast-growth cells, days 116-129": {
            "trained before growth": mape(before_growth, on_fast),
            "retrained after": mape(after_growth, on_fast),
        },
        "all cells, days 137-150": {
            "trained before growth": mape(before_growth, after_holiday),
            "retrained on the holiday": mape(on_holiday, after_holiday),
        },
    }
).round(2)
