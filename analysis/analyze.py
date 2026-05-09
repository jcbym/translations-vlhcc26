# %% Import

import lib

import arviz_stats as azs
import importlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cmdstanpy import CmdStanModel

importlib.reload(lib)

# % % Global matplotlib configuration

plt.rcParams["font.family"] = "PT Sans"

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

# %% Descriptive plots

figsize = (2.75, 2)
compressed_figsize = (1.4, 1.7)
label_fontsize = 5

importlib.reload(lib)
lib.distribution_comparison_plot(
    wide,
    group_feature="interface_label",
    value_feature="exp",
    sort_feature="interface_id",
    color_feature="interface_color",
    yticks=np.arange(0, 11, 2),
    figsize=(3, 1.2),
    label_fontsize=label_fontsize,
    caption="Figure 3.",
    compressed=False,
)[0].save("output/01-exp.pdf")

for task in TASKS:
    lib.count_comparison_plot(
        wide,
        group_feature="interface_label",
        value_feature=f"correct{task}",
        sort_feature="interface_id",
        color_feature="interface_color",
        step=2,
        figsize=(3, 1.3),
        label_fontsize=label_fontsize,
        compressed=False,
    )[0].save(f"output/02-correct{task}.pdf")

    lib.distribution_comparison_plot(
        wide.filter(pl.col(f"correct{task}")),
        group_feature="interface_label",
        value_feature=f"taskTime{task}",
        sort_feature="interface_id",
        color_feature="interface_color",
        yticks=np.arange(0, 61, 10),
        figsize=(3, 1.3),
        label_fontsize=label_fontsize,
        caption=f"Figure B2.{task}.",
        compressed=False,
    )[0].save(f"output/03-correct_time_taken{task}.pdf")

    lib.distribution_comparison_plot(
        wide.filter(~pl.col(f"correct{task}")),
        group_feature="interface_label",
        value_feature=f"taskTime{task}",
        sort_feature="interface_id",
        color_feature="interface_color",
        yticks=np.arange(0, 61, 10),
        figsize=compressed_figsize,
        label_fontsize=label_fontsize,
        caption=f"Figure B3.{task}.",
        compressed=False,
    )[0].save(f"output/03-incorrect_time_taken{task}.pdf")

import sys

sys.exit(0)

# %% Run Bayesian inference


def get_var(fit, var):
    x = fit.stan_variable(var)
    return x.reshape(1, *x.shape)


correct_model = CmdStanModel(stan_file="stan/correct.stan")
time_taken_model = CmdStanModel(stan_file="stan/time_taken.stan")

posterior = {}
posteriorES = {}

with open("output/model_diagnostics.txt", "w") as f:
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

        print("========================================", file=f)
        print(f"correct ({task})", file=f)
        print("========================================", file=f)
        print(correct_fit.diagnose(), file=f)
        print(correct_fit.summary().to_string(), file=f)
        posterior[f"theta{task}"] = get_var(correct_fit, "theta")
        posteriorES[f"theta{task}"] = get_var(correct_fit, "thetaES")

        df = wide.filter(pl.col(f"correct{task}"))

        correct_time_taken_fit = time_taken_model.sample(
            data={
                "N": len(df),
                "I": len(interfaces),
                "interface": df["interface_id"].to_numpy(),
                "timeTaken": df[f"taskTime{task}"].to_numpy(),
            },
            chains=8,
            iter_sampling=10_000,
        )

        print("========================================", file=f)
        print(f"correct_time_taken ({task})", file=f)
        print("========================================", file=f)
        print(correct_time_taken_fit.diagnose(), file=f)
        print(correct_time_taken_fit.summary().to_string(), file=f)
        posterior[f"correct_mu{task}"] = get_var(
            correct_time_taken_fit,
            "mu",
        )
        posteriorES[f"correct_logmu{task}"] = get_var(
            correct_time_taken_fit,
            "logmuES",
        )

        df = wide.filter(~pl.col(f"correct{task}"))

        incorrect_time_taken_fit = time_taken_model.sample(
            data={
                "N": len(df),
                "I": len(interfaces),
                "interface": df["interface_id"].to_numpy(),
                "timeTaken": df[f"taskTime{task}"].to_numpy(),
            },
            chains=8,
            iter_sampling=20_000,
        )

        print("========================================", file=f)
        print(f"incorrect_time_taken ({task})", file=f)
        print("========================================", file=f)
        print(incorrect_time_taken_fit.diagnose(), file=f)
        print(incorrect_time_taken_fit.summary().to_string(), file=f)
        posterior[f"incorrect_mu{task}"] = get_var(
            incorrect_time_taken_fit,
            "mu",
        )
        posteriorES[f"incorrect_logmu{task}"] = get_var(
            incorrect_time_taken_fit,
            "logmuES",
        )

# %% Plot effect size posterior distributions

importlib.reload(lib)
for task in TASKS:
    lib.es_plot(
        posteriorES[f"theta{task}"],
        labels=interfaces["interface_label"],
        colors=interfaces["interface_color"],
        measure="Probability of success",
        better_notion="better",
        worse_notion="worse",
        better="greater",
        bins=np.arange(-1, 1.0001, 0.01),
        step=1,
        figsize=(4, 2.5),
        fontsize=7,
        xticks=np.arange(-1, 1.001, 0.25),
        round_amount=2,
    )[0].save(f"output/04-theta{task}.pdf")

    lib.es_plot(
        posteriorES[f"correct_logmu{task}"],
        labels=interfaces["interface_label"],
        colors=interfaces["interface_color"],
        measure="Time taken",
        better_notion="faster",
        worse_notion="slower",
        better="less",
        bins=np.arange(-20, 20.0001, 0.1),
        step=1,
        figsize=(4, 2.5),
        fontsize=7,
        xticks=np.arange(-20, 20.0001, 5).astype(int),
        round_amount=1,
    )[0].save(f"output/05-correct_mu{task}.pdf")

    lib.es_plot(
        posteriorES[f"incorrect_logmu{task}"],
        labels=interfaces["interface_label"],
        colors=interfaces["interface_color"],
        measure="Time taken",
        better_notion="faster",
        worse_notion="slower",
        better="less",
        bins=np.arange(-90, 90.0001, 0.1),
        step=1,
        figsize=(4, 2.5),
        fontsize=7,
        xticks=np.arange(-90, 90.0001, 15).astype(int),
        round_amount=1,
    )[0].save(f"output/06-incorrect_mu{task}.pdf")

# %% Output summary statistics of the posteriors


def hdi_many(x, *, prob):
    return np.array(
        [azs.hdi(x[0, :, c], prob=prob) for c in range(0, x.shape[2])]
    )


def summary(x, es, es_criteria, *, prefix, hdi_prob=0.95):
    x_hdi = hdi_many(x, prob=hdi_prob)
    es_hdi = hdi_many(es, prob=hdi_prob)

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
        posterior[f"correct_mu{task}"],
        posteriorES[f"correct_logmu{task}"][:, :, :, 0],
        lambda x: x < 0,
        prefix=f"correct_mu{task}",
    )

    data |= summary(
        posterior[f"incorrect_mu{task}"],
        posteriorES[f"incorrect_logmu{task}"][:, :, :, 0],
        lambda x: x < 0,
        prefix=f"incorrect_mu{task}",
    )

stats = interfaces.join(
    pl.DataFrame(data=data),
    on="interface_id",
    validate="1:1",
)

stats.write_csv("output/stats.csv")

# %% Make teaser figure graphs

importlib.reload(lib)
for row in stats.iter_rows(named=True):
    iid = row["interface_id"]
    label = row["interface_label"]
    lib.teaser_plot(
        [row[f"theta{task}_p"] for task in TASKS],
        color=row["interface_color"],
    )[0].save(f"output/teaser/{iid}-theta-{label}.svg")
