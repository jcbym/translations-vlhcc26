# %% Import

import lib

import arviz as az
import importlib
import numpy as np
import polars as pl

from cmdstanpy import CmdStanModel

importlib.reload(lib)

# Color-blind-friendly color schemes: https://personal.sron.nl/~pault/

# % % Load metadata

TASKS = [1, 2, 3]

interface_groups = pl.read_csv("data/interface_groups.csv")

interfaces = (
    pl.read_csv("data/interfaces.csv")
    .with_row_index(
        "interface_id",
        offset=1,
    )
    .join(
        interface_groups,
        on="interface_group",
        validate="m:1",
    )
)

participants = pl.read_csv("data/participants.csv")

# % % Load and process data

wide = pl.read_csv("data/data.csv")[
    [
        "userID",
        "interface",
    ]
    + [f"correct{task}" for task in TASKS]
    + [f"taskTime{task}" for task in TASKS]
]

for task in TASKS:
    wide = wide.with_columns(
        # Consider "skip" unsuccessful
        pl.col(f"correct{task}").fill_null(pl.lit(False)),
        # Express task time in minutes
        pl.col(f"taskTime{task}").cast(float) / (1000 * 60),
    )

wide = wide.join(
    participants,
    left_on="userID",
    right_on="id",
    validate="1:1",
).join(
    interfaces,
    left_on="interface",
    right_on="interface_tag",
    validate="m:1",
)

# % % Descriptive plots

lib.distribution_comparison_plot(
    wide,
    group_feature="interface_label",
    value_feature="exp",
    sort_feature="interface_id",
    color_feature="interface_color",
    yticks=np.arange(0, 11, 1),
    figsize=(6, 4),
)[0].save("output/01-exp.pdf")

importlib.reload(lib)
for task in TASKS:
    lib.count_comparison_plot(
        wide,
        group_feature="interface_label",
        value_feature=f"correct{task}",
        sort_feature="interface_id",
        color_feature="interface_color",
        step=2,
        figsize=(6, 4),
    )[0].save(f"output/02-correct{task}.pdf")

    lib.distribution_comparison_plot(
        wide.filter(pl.col(f"correct{task}")),
        group_feature="interface_label",
        value_feature=f"taskTime{task}",
        sort_feature="interface_id",
        color_feature="interface_color",
        yticks=np.arange(0, 51, 5),
        figsize=(6, 4),
    )[0].save(f"output/03-time_taken{task}.pdf")

# %% Run Bayesian inference


def get_var(fit, var):
    x = fit.stan_variable(var)
    return x.reshape(1, *x.shape)


correct_model = CmdStanModel(stan_file="stan/correct.stan")
time_taken_model = CmdStanModel(stan_file="stan/time_taken.stan")

posterior = {}
posteriorES = {}

for task in TASKS:
    correct_fit = correct_model.sample(
        data={
            "N": len(wide),
            "I": len(interfaces),
            "interface": wide["interface_id"].to_numpy(),
            "correct": wide[f"correct{task}"].to_numpy(),
        },
        chains=8,
        iter_sampling=10_000,
    )

    print(f"correct{task}")
    print(correct_fit.diagnose())
    print(correct_fit.summary())
    posterior[f"theta{task}"] = get_var(correct_fit, "theta")
    posteriorES[f"theta{task}"] = get_var(correct_fit, "thetaES")

    time_taken_fit = time_taken_model.sample(
        data={
            "N": len(wide),
            "I": len(interfaces),
            "interface": wide["interface_id"].to_numpy(),
            "timeTaken": wide[f"taskTime{task}"].to_numpy(),
        },
        chains=8,
        iter_sampling=10_000,
    )

    print(f"taskTime{task}")
    print(time_taken_fit.diagnose())
    print(time_taken_fit.summary())
    posterior[f"mu{task}"] = get_var(time_taken_fit, "mu")
    posteriorES[f"logmu{task}"] = get_var(time_taken_fit, "logmuES")

# %% Plot effect size posterior distributions

importlib.reload(lib)
for task in TASKS:
    lib.es_plot(
        posteriorES[f"theta{task}"],
        labels=interfaces["interface_label"],
        colors=interfaces["interface_color"],
        measure="Probability of success",
        better="greater",
        bins=np.arange(-3, 3.1, 0.05),
        step=1,
        figsize=(6, 4),
    )[0].save(f"output/04-theta{task}.pdf")

    lib.es_plot(
        posteriorES[f"logmu{task}"],
        labels=interfaces["interface_label"],
        colors=interfaces["interface_color"],
        measure="Time taken",
        better="less",
        bins=np.arange(-3, 3, 0.05),
        step=1,
        figsize=(6, 4),
    )[0].save(f"output/04-mu{task}.pdf")

# %% Output summary statistics of the posteriors


def summary(x, es, es_criteria, *, prefix, hdi_prob=0.95):
    x_hdi = az.hdi(x, hdi_prob=hdi_prob)
    es_hdi = az.hdi(es, hdi_prob=hdi_prob)

    return {
        f"{prefix}_mean": x.mean(axis=(0, 1)),
        f"{prefix}_lo": x_hdi[:, 0],
        f"{prefix}_hi": x_hdi[:, 1],
        f"{prefix}_es_mean": es.mean(axis=(0, 1)),
        f"{prefix}_es_lo": es_hdi[:, 0],
        f"{prefix}_es_hi": es_hdi[:, 1],
        f"{prefix}_p": es_criteria(es).mean(axis=(0, 1)),
    }


data = {"interface_id": np.arange(1, len(interfaces) + 1)}

for task in TASKS:
    data |= summary(
        posterior[f"theta{task}"],
        posteriorES[f"theta{task}"][:, :, :, 0],
        lambda x: x > 0,
        prefix=f"theta{task}",
    )

    data |= summary(
        posterior[f"mu{task}"],
        posteriorES[f"logmu{task}"][:, :, :, 0],
        lambda x: x < 0,
        prefix=f"mu{task}",
    )

stats = interfaces.join(
    pl.DataFrame(data=data),
    on="interface_id",
    validate="1:1",
)

stats.write_csv("output/stats.csv")
