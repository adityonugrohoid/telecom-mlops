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
# # Churn: the dated generator, its drift calendar and the day-0 model
#
# **All data here is simulated.** Customers, charges, network readings and churn come from
# the synthetic generator in `telecom_ml_churn.generator`; no real subscriber data is used.
#
# The notebook is a report on the package. It shows:
#
# 1. one simulated day of customers and its label delay
# 2. the drift calendar: which parameters move, when, and by how much
# 3. what the price rise does to the inputs and the churn rate
# 4. the day-0 model against its baseline (the anchor to the earlier work)
# 5. what drives the model's churn scores (SHAP)

# %%
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from telecom_ml_churn import generator
from telecom_ml_churn.baseline import fit_baseline
from telecom_ml_churn.features import TARGET, engineered_features
from telecom_ml_churn.model import fit_model
from telecom_ml_churn.usecase import ChurnUseCase

usecase = ChurnUseCase()
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
# Each day brings 1,000 customers whose 30-day observation window ends that day. Their
# churn label is released 30 days later, and only then can the loop train or judge on it.

# %%
batch = generator.generate(scenario.start, scenario)
print(f"{len(batch.data):,} customers on {batch.day}; churn rate {batch.data[TARGET].mean():.3f}")
print(f"labels released on {batch.data['label_released_on'].iloc[0].date()}")
batch.data.head()

# %% [markdown]
# ## 2. The drift calendar
#
# Three events over 180 simulated days. The price rise at day 60 raises month-to-month
# charges by 15% and makes customers more sensitive to charges (a concept change). The 5G
# share growth at day 150 is the benign event: the inputs shift, the label mechanism does not.

# %%
calendar = pd.DataFrame([generator.params_on(day(i), scenario) for i in range(scenario.days)])
fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharex=True)
for ax, (param, title) in zip(
    axes,
    [
        ("mtm_charge_mult", "month-to-month charge multiplier"),
        ("charge_coef", "charge coefficient (concept)"),
        ("share_5g", "5G share (benign)"),
    ],
    strict=True,
):
    ax.plot(calendar.index, calendar[param])
    ax.set_title(title)
    ax.set_xlabel("simulated day")
fig.tight_layout()

# %% [markdown]
# ## 3. What the price rise does
#
# Ten days before the rise against ten days after it has fully ramped in.

# %%
before, after = days(40, 10, scenario), days(80, 10, scenario)
mtm = "month-to-month"
summary = pd.DataFrame(
    {
        "before (days 40-49)": [
            before.loc[before.contract_type == mtm, "monthly_charges"].mean(),
            before[TARGET].mean(),
        ],
        "after (days 80-89)": [
            after.loc[after.contract_type == mtm, "monthly_charges"].mean(),
            after[TARGET].mean(),
        ],
    },
    index=["month-to-month charge", "churn rate"],
)
summary.round(3)

# %% [markdown]
# ## 4. Day-0 model against its baseline
#
# Ten scenario-off days (10,000 customers), split 80/20 with seed 42 as the earlier work
# did. The earlier evidence: logistic regression AUROC 0.8727, XGBoost 0.8548. The
# logistic baseline scoring higher is expected on this generator, whose label is itself a
# logit on the raw inputs.

# %%
data = days(0, 10, off)
train, test = train_test_split(data.index, test_size=0.2, random_state=42, stratify=data[TARGET])
labels = data.loc[test, TARGET]
baseline = fit_baseline(data.loc[train])
model = fit_model(data.loc[train], seed=42)
pd.Series(
    {
        "logistic baseline": roc_auc_score(labels, baseline.predict_proba(data.loc[test])),
        "xgboost model": roc_auc_score(labels, model.predict_proba(data.loc[test])),
    },
    name="AUROC",
).round(4)

# %% [markdown]
# ## 5. What drives the churn score
#
# SHAP values of the XGBoost model on 500 test customers.

# %%
features = engineered_features(data.loc[test]).iloc[:500]
explanation = shap.TreeExplainer(model.classifier)(features)
shap.plots.beeswarm(explanation, max_display=12, show=False)
plt.tight_layout()
