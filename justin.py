# %% Import

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats as stats


# %% Define interfaces and order

INTERFACES = [
    ("control", "Unfamiliar Only", 0),
    ("translation", "Translation", 0),
    # "Probe–Mapping",
    ("canonicalization", "Probe–Components", 1),
    ("sequence", "Probe–Translation Steps", 1),
    ("llmTranslation", "Probe–NL Translation Explanation", 1),
    ("llmBasic", "NL Explanation", 2),
]

INTERFACE_ORDER = [label for (_, label, _) in INTERFACES]
EXPERIMENTS = {i: e for (_, i, e) in INTERFACES}

# %% Load (wide) data

wide = pd.read_csv(
    "data.csv",
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

wide["interface"] = wide["interface"].replace({k: v for (k, v, _) in INTERFACES})

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
).reset_index()

long["score"] = long["correct"].astype(float) / 3

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
            "score": "sum",
        }
    )
    .reset_index()
)

# %% Distribution of features


def plot_hist(
    df,
    feature,
    lo,
    hi,
    step,
    *,
    correct_only,
):
    if correct_only:
        df = df[df["correct"]]

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

    fig.savefig(f"output/distribution-c{correct_only}-{feature}.pdf")


plot_hist(data, "taskTime", 0, 90, 10, correct_only=True)
plot_hist(data, "correct", 0, 1, 0.2, correct_only=False)
plot_hist(data, "score", 0, 1, 0.2, correct_only=False)


# %% Bootstrap feature estimators


# Color-blind-friendly color schemes: https://personal.sron.nl/~pault/


def bootstrap_forest(
    df,
    *,
    feature,
    estimator,
    lo,
    hi,
    step,
    correct_only,
    spacing=0.5,
    experiment_colors={
        0: "#004488",
        1: "#BB5566",
        2: "#DDAA33",
    },
    xlabel=None,
    xformatter=None,
    text_format=None,
):
    if correct_only:
        df = df[df["correct"]]

    estimator_name = estimator.__name__

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    coords = []
    interfaces = []
    lows = []
    highs = []
    estimates = []
    colors = []

    coord = 0
    prev_experiment = None
    for interface in reversed(INTERFACE_ORDER):
        experiment = EXPERIMENTS[interface]
        if experiment != prev_experiment:
            coord += spacing

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
        coords.append(coord)
        interfaces.append(interface)
        lows.append(result.confidence_interval.low)
        highs.append(result.confidence_interval.high)
        estimates.append(estimator(vals))
        colors.append(experiment_colors[experiment])

        coord += 1
        prev_experiment = experiment

    df = pd.DataFrame(
        {
            "coord": coords,
            "interface": interfaces,
            "low": lows,
            "high": highs,
            "estimate": estimates,
            "color": colors,
        }
    )

    for _, row in df.iterrows():
        ax.errorbar(
            [row["estimate"]],
            [row["coord"]],
            xerr=[[row["estimate"] - row["low"]], [row["high"] - row["estimate"]]],
            color=row["color"],
            fmt="o",
        )

        if text_format:
            hpad = text_format["hpad"]
            f = text_format["formatter"]
            alpha = 0.5

            ax.text(
                row["low"] - hpad,
                row["coord"],
                f(row["low"]),
                va="center",
                ha="right",
                color=row["color"],
                alpha=alpha,
            )
            ax.text(
                row["estimate"],
                row["coord"] + 0.15,
                f(row["estimate"]),
                va="bottom",
                ha="center",
                color=row["color"],
                alpha=alpha,
            )
            ax.text(
                row["high"] + hpad,
                row["coord"],
                f(row["high"]),
                va="center",
                ha="left",
                color=row["color"],
                alpha=alpha,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(lo, hi)
    ax.set_xticks(np.arange(lo, hi + 0.0000001, step))
    if xformatter:
        ax.xaxis.set_major_formatter(xformatter)
    ax.set_xlabel(xlabel if xlabel else f"$\\mathbf{{{estimator_name}\\ {feature}}}$")

    ax.set_ylim(df["coord"].min() - spacing, df["coord"].max() + spacing)
    ax.set_yticks(df["coord"], labels=df["interface"])
    for t in ax.yaxis.get_ticklabels():
        t.set_color(experiment_colors[EXPERIMENTS[t.get_text()]])

    fig.tight_layout()
    fig.savefig(f"output/forest-c{correct_only}-{feature}-{estimator_name}.pdf")


bootstrap_forest(
    data,
    feature="taskTime",
    estimator=np.median,
    lo=0,
    hi=60,
    step=10,
    correct_only=True,
    xlabel=r"$\mathbf{Median\ time\ taken}\ \text{(min.)},\, \it{lower}\ \text{is\ better}$",
    text_format={
        "hpad": 0.8,
        "formatter": lambda x: str(round(x, 1)),
    },
)

bootstrap_forest(
    data,
    feature="score",
    estimator=np.mean,
    lo=0,
    hi=1,
    step=0.1,
    correct_only=False,
    xlabel=r"$\mathbf{Percent\ correct},\, \it{higher}\ \text{is\ better}$",
    xformatter=ticker.PercentFormatter(xmax=1),
    text_format={
        "hpad": 0.01,
        "formatter": lambda x: str(int(round(x, 2) * 100)) + "%",
    },
)
