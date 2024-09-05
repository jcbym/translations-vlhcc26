# %% Import

import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import pymc as pm
import arviz as az

import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

import logging

# %% Load (wide) data

wide = pd.read_csv(
    "output.csv",
    dtype=object,
    keep_default_na=False,
)[
    [
        "userID",
        "interface",
        "timesRan1",
        "timesRan2",
        "timesRan3",
        "timesUsed1",
        "timesUsed2",
        "timesUsed3",
        "taskTime1",
        "taskTime2",
        "taskTime3",
        "correct1",
        "correct2",
        "correct3",
    ]
]

wide["interface"] = wide["interface"].replace(
    {
        "canonicalization": "Probe–Components",
        "control": "Unfamiliar Only",
        "llmBasic": "NL",
        "llmTranslation": "Probe–NL Translation Explanation",
        "sequence": "Probe–Translation Steps",
        "translation": "Translation",
    }
)

for c in [
    "timesRan1",
    "timesRan2",
    "timesRan3",
    "timesUsed1",
    "timesUsed2",
    "timesUsed3",
]:
    wide[c] = wide[c].replace("", 0).astype(int)

# Express task time in minutes
for c in [
    "taskTime1",
    "taskTime2",
    "taskTime3",
]:
    wide[c] = (wide[c].astype(float) / 1000) / 60

# Treat "skipped" as incorrect
for c in [
    "correct1",
    "correct2",
    "correct3",
]:
    wide[c] = (
        wide[c]
        .replace(
            {
                "": False,
                "FALSE": False,
                "TRUE": True,
            }
        )
        .astype(bool)
    )

wide

# %% Convert to long format

long = pd.wide_to_long(
    wide,
    ["timesRan", "timesUsed", "taskTime", "correct"],
    i=["userID", "interface"],
    j="task",
)

long

# %% Aggregate over tasks

data = (
    long.groupby(["userID", "interface"])
    .agg({"timesRan": "sum", "timesUsed": "sum", "taskTime": "sum", "correct": "all"})
    .reset_index()
)

correct = data[data["correct"]].copy()

INTERFACE_LIST = list(correct["interface"].unique())
correct["i"] = correct["interface"].apply(lambda i: INTERFACE_LIST.index(i))

# %% Bayes


def model(times, interfaces):
    # y = mx + b
    # m = pyro.sample("m", dist.Normal(0, 10))
    # b = pyro.sample("b", dist.Normal(0, 10))
    # with pyro.plate("N", len(interfaces)):
    #     pyro.sample("y_i", dist.Normal(m * interfaces + b, 5), obs=times)
    beta_0 = pyro.sample("beta_0", dist.Gamma(30, 1))
    sigma_y = pyro.sample("sigma_y", dist.Uniform(1, 50))

    with pyro.plate("J", len(INTERFACE_LIST)):
        sigma_beta = pyro.sample("sigma_beta", dist.Gamma(10, 1))
        beta_j = pyro.sample("beta_j", dist.Normal(0, sigma_beta))
    with pyro.plate("N", len(interfaces)):
        mu_i = beta_0 + beta_j[interfaces]
        pyro.sample("y_i", dist.Normal(mu_i, sigma_y), obs=times)


times = torch.tensor(correct["taskTime"].values)
interfaces = torch.tensor(correct["i"].values)

pyro.render_model(
    model,
    model_args=(times, interfaces),
    filename="model.pdf",
    render_distributions=True,
)

pyro.clear_param_store()

auto_guide = pyro.infer.autoguide.AutoNormal(model)
adam = pyro.optim.Adam({"lr": 0.02})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

losses = []
for step in range(2000):
    loss = svi.step(times, interfaces)
    losses.append(loss)
    if step % 200 == 0:
        print("Elbo loss: {}".format(loss))

predictive = pyro.infer.Predictive(model, guide=auto_guide, num_samples=800)
samples = predictive(None, interfaces)

# %%

u = samples["beta_j"]
deflection = pd.DataFrame(
    {
        "avg": u.mean(0).detach().cpu().numpy(),
        "lo": u.kthvalue(int(len(u) * 0.05), dim=0)[0].detach().cpu().numpy(),
        "hi": u.kthvalue(int(len(u) * 0.95), dim=0)[0].detach().cpu().numpy(),
    }
)
deflection["interface"] = deflection.index.map(lambda i: INTERFACE_LIST[i])

# %% Distribution of

interfaces = [
    "Unfamiliar Only",
    "Translation",
    "Probe–Mapping",
    "Probe–Components",
    "Probe–Translation Steps",
    "Probe–NL Translation Explanation",
    "NL",
]

alt.layer(
    alt.Chart()
    .mark_point(color="black", filled=True)
    .encode(
        x="mean(taskTime):Q",
        y=alt.Y("interface:N", sort=order),
    ),
    alt.Chart()
    .mark_errorbar(extent="ci")
    .encode(
        x="taskTime:Q",
        y=alt.Y("interface:N", sort=order),
    ),
    data=correct,
).interactive().save("chart1.html")

# %% Distribution of

(
    alt.Chart(correct)
    .mark_bar()
    .encode(
        x=alt.X("taskTime", bin=True),
        y="count()",
    )
    .facet(row="interface")
    .interactive()
    .save("chart2.html")
)

# %%

order = [
    "Unfamiliar Only",
    "Translation",
    "Probe–Mapping",
    "Probe–Components",
    "Probe–Translation Steps",
    "Probe–NL Translation Explanation",
    "NL",
]

alt.layer(
    alt.Chart()
    .mark_point(color="black", filled=True)
    .encode(
        x="avg:Q",
        y=alt.Y("interface:N", sort=order),
    ),
    alt.Chart()
    .mark_errorbar(extent="ci")
    .encode(
        x="lo:Q",
        x2="hi:Q",
        y=alt.Y("interface:N", sort=order),
    ),
    data=deflection,
).interactive().save("chart3.html")
