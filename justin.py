# %% Import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


# %% Define interfaces and order

INTERFACES = [
    ("control", "Unfamiliar Only"),
    ("translation", "Translation"),
    # "Probe–Mapping",
    ("canonicalization", "Probe–Components"),
    ("sequence", "Probe–Translation Steps"),
    ("llmTranslation", "Probe–NL Translation Explanation"),
    ("llmBasic", "NL"),
]

INTERFACE_ORDER = [label for (_, label) in INTERFACES]

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

wide["interface"] = wide["interface"].replace({k: v for (k, v) in INTERFACES})

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
    .agg(
        {
            "timesRan": "sum",
            "timesUsed": "sum",
            "taskTime": "sum",
            "correct": "all",
        }
    )
    .reset_index()
)

correct = data[data["correct"]]

# %% Distribution of features


def plot_hist(df, feature, lo, hi, step):
    fig, ax = plt.subplots(
        len(INTERFACE_ORDER),
        1,
        figsize=(7, 6),
        layout="constrained",
    )
    fig.get_layout_engine().set(hspace=0.05)

    yticks = [0, 0.25, 0.5, 0.75, 1]
    yticklabels = ["0%", "", "50%", "", "100%"]

    for i, interface in enumerate(INTERFACE_ORDER):
        vals = df[df["interface"] == interface][feature].astype(float)
        bins = np.arange(lo, hi + 0.0000001, step)
        counts, _, _ = ax[i].hist(
            vals,
            bins=bins,
            color="gray",
            edgecolor="black",
            weights=np.ones_like(vals) / len(vals),
        )

        median = vals.median()
        ax[i].axvline(x=median, c="#DD0000", lw=1.5, clip_on=False)
        ax[i].scatter([median], [0], c="#DD0000", marker="^", clip_on=False)

        mean = vals.mean()
        ax[i].axvline(x=mean, c="#0000DD", lw=1.5, clip_on=False)
        ax[i].scatter([mean], [0], c="#0000DD", marker="x", clip_on=False)

        ax[i].set_xlim(lo, hi)
        ax[i].set_xticks(bins)

        ax[i].set_ylim(0, 1)
        ax[i].set_yticks(yticks, labels=yticklabels)

        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)

        ax[i].text(
            -0.15,
            0.5,
            interface,
            ha="right",
            va="center",
            fontweight="bold",
            transform=ax[i].transAxes,
        )

    ax[-1].set_xlabel(r"$\bf{" + feature + r"}$", fontsize=10)
    # ax[1].set_ylabel(r"$\bf Relative\ frequency$", fontsize=10)

    fig.savefig(f"distribution-{feature}.pdf")


plot_hist(correct, "taskTime", 0, 90, 10)
plot_hist(data, "correct", 0, 1, 0.2)


# %% Bootstrap feature estimators


def bootstrap_forest(df, feature, estimator, lo, hi, step, spacing=0.5):
    estimator_name = estimator.__name__

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    coords = []
    interfaces = []
    lows = []
    highs = []
    estimates = []

    for i, interface in enumerate(reversed(INTERFACE_ORDER)):
        vals = df[df["interface"] == interface][feature].values
        result = stats.bootstrap(
            (vals,),
            estimator,
            n_resamples=99999,
            confidence_level=0.95,
            alternative="two-sided",
            method="BCa",
            random_state=0,
        )
        coords.append(i + spacing)
        interfaces.append(interface)
        lows.append(result.confidence_interval.low)
        highs.append(result.confidence_interval.high)
        estimates.append(estimator(vals))

    df = pd.DataFrame(
        {
            "coord": coords,
            "interface": interfaces,
            "low": lows,
            "high": highs,
            "estimate": estimates,
        }
    )

    ax.errorbar(
        df["estimate"],
        df["coord"],
        xerr=[df["estimate"] - df["low"], df["high"] - df["estimate"]],
        fmt="o",
        color="black",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(lo, hi)
    ax.set_xticks(np.arange(lo, hi + 0.0000001, step))
    ax.set_xlabel(f"{estimator_name} {feature}", fontweight="bold")

    ax.set_ylim(df["coord"].min() - spacing, df["coord"].max() + spacing)
    ax.set_yticks(df["coord"], labels=df["interface"])

    fig.tight_layout()
    fig.savefig(f"forest-{feature}-{estimator_name}.pdf")


bootstrap_forest(correct, "taskTime", np.median, 0, 90, 10)
bootstrap_forest(data, "correct", np.mean, 0, 1, 0.2)
