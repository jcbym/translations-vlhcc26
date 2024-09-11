# %% Import

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats as stats

pd.set_option("future.no_silent_downcasting", True)

# Color-blind-friendly color schemes: https://personal.sron.nl/~pault/

# %% Define interfaces and order

INTERFACES = [
    ("control", "Basic-Control", 0),
    ("translation", "Basic-Translation", 0),
    ("llmBasic", "Basic-NL", 1),
    ("highlighting", "Pointed-Highlight", 2),
    ("canonicalization", "Pointed-Individual", 2),
    ("sequence", "Pointed-StepByStep", 2),
    ("llmTranslation", "Pointed-Translation+NL", 2),
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

for c in [
    "correct1",
    "correct2",
    "correct3",
]:
    vals = wide[c]

    # Treat "skipped" as incorrect (put "skipped" in new column)
    wide[c] = vals.replace(
        {
            "": False,
            "FALSE": False,
            "TRUE": True,
        }
    ).astype(bool)

    skip_c = f"skipped{c[-1]}"
    wide[skip_c] = vals.replace(
        {
            "": True,
            "FALSE": False,
            "TRUE": False,
        }
    ).astype(bool)

wide

# %% Convert to long format

long = pd.wide_to_long(
    wide,
    ["timesRan", "timesUsed", "taskTime", "correct", "skipped"],
    i=["userID", "interface"],
    j="task",
).reset_index()

long["success_rate"] = long["correct"].astype(float) / 3

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
            "success_rate": "sum",
        }
    )
    .reset_index()
)

# %% Distribution of features


def plot_hist(
    df,
    *,
    feature,
    lo,
    hi,
    step,
    xlabel,
    large,
    correct_only,
    xformatter=None,
):
    if correct_only:
        df = df[df["correct"]]

    too_low = df[feature] < lo
    too_high = df[feature] > hi

    if too_low.sum() > 0:
        print(
            f"WARNING: {too_low.sum()} too low for distribution-c{correct_only}-{feature}.pdf"
        )
        print(df[too_low])

    if too_high.sum() > 0:
        print(
            f"WARNING: {too_high.sum()} too high for distribution-c{correct_only}-{feature}.pdf"
        )
        print(df[too_high])

    fig, ax = plt.subplots(
        len(INTERFACE_ORDER),
        1,
        figsize=(8, 6),
        layout="constrained",
    )
    fig.get_layout_engine().set(hspace=0.05)

    if large:
        yticks = [0, 2, 4, 6, 8, 10, 12]
        yticklabels = ["0", "", "4", "", "8", "", "12"]
    else:
        yticks = [0, 1, 2, 3, 4]
        yticklabels = ["0", "", "2", "", "4"]

    for i, interface in enumerate(INTERFACE_ORDER):
        vals = df[df["interface"] == interface][feature].astype(float)

        if step:  # continuous
            bins = np.arange(lo, hi + 0.0000001, step)
            ax[i].hist(
                vals,
                bins=bins,
                color="0.8",
                edgecolor="0.6",
            )
            ax[i].set_xlim(lo, hi)
            ax[i].set_xticks(bins)
        else:  # discrete
            xs, counts = np.unique(
                vals,
                return_counts=True,
            )
            spacing = 0.2
            ax[i].bar(
                xs,
                counts,
                color="0.8",
                edgecolor="0.6",
                width=spacing,
            )
            ax[i].set_xlim(-spacing, 1 + spacing)
            ax[i].set_xticks([0, 0.333, 0.667, 1])

        median = vals.median()
        ax[i].axvline(
            x=median,
            c="#009988",
            lw=2,
            clip_on=False,
            zorder=100,
        )
        ax[i].scatter(
            [median],
            [0],
            c="#009988",
            marker="^",
            clip_on=False,
            label="Median",
            zorder=100,
            s=50,
        )

        mean = vals.mean()
        ax[i].axvline(
            x=mean,
            c="#CC3311",
            lw=2,
            clip_on=False,
            zorder=100,
        )
        ax[i].scatter(
            [mean],
            [0],
            c="#CC3311",
            marker="x",
            clip_on=False,
            label="Mean",
            zorder=100,
            s=50,
        )

        if i == 0:
            ax[i].legend(bbox_to_anchor=(1.05, 1.05))

        if xformatter:
            ax[i].xaxis.set_major_formatter(xformatter)

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

    ax[-1].set_xlabel(
        xlabel,  #  + " – " + ("continuous" if step else "discrete") + " data",
        fontsize=10,
    )

    # ax[int((len(INTERFACES) - 1) / 2)].set_ylabel(
    #     "Relative frequency",
    #     fontsize=10,
    #     fontweight="bold",
    # )

    fig.savefig(
        f"output/distribution-c{correct_only}-{feature}.pdf",
        bbox_inches="tight",
    )


plot_hist(
    data,
    feature="taskTime",
    lo=0,
    hi=90,
    step=10,
    xlabel=r"$\mathbf{Time\ taken}$ (in minutes) among successful participants",
    correct_only=True,
    large=False,
)

plot_hist(
    data,
    feature="success_rate",
    lo=0,
    hi=1,
    step=None,
    xlabel=r"$\mathbf{Success\ rate}$ among all participants",
    correct_only=False,
    xformatter=ticker.PercentFormatter(xmax=1),
    large=True,
)

# %% Bootstrap feature estimators


def bootstrap_forest(
    df,
    *,
    feature,
    estimator,
    lo,
    hi,
    step,
    correct_only,
    spacing=1.5,
    padding=0.5,
    experiment_colors={
        0: "#004488",
        1: "#6699CC",
        2: "#994455",
    },
    xlabel=None,
    xformatter=None,
    text_format=None,
):
    if correct_only:
        df = df[df["correct"]]

    estimator_name = estimator.__name__

    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))

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
            coord += padding

        vals = df[df["interface"] == interface][feature].values
        estimate = estimator(vals)
        if len(vals) == 1:
            low = estimate
            high = estimate
            print(
                f"WARNING: Only 1 sample for {interface} for file 'forest-c{correct_only}-{feature}-{estimator_name}.pdf'"
            )
        else:
            result = stats.bootstrap(
                (vals,),
                estimator,
                n_resamples=99999,
                confidence_level=0.95,
                alternative="two-sided",
                method="percentile",
                random_state=0,
            )
            low = result.confidence_interval.low
            high = result.confidence_interval.high

        coords.append(coord)
        interfaces.append(interface)
        lows.append(low)
        highs.append(high)
        estimates.append(estimate)
        colors.append(experiment_colors[experiment])

        coord += spacing
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

    ax.set_ylim(df["coord"].min() - padding, df["coord"].max() + padding)
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
    hi=70,
    step=10,
    correct_only=True,
    xlabel=r"$\mathbf{Median\ time\ taken}$ (in minutes) among successful participants",
    text_format={
        "hpad": 0.8,
        "formatter": lambda x: str(round(x, 1)),
    },
)

bootstrap_forest(
    data,
    feature="success_rate",
    estimator=np.mean,
    lo=0,
    hi=1,
    step=0.1,
    correct_only=False,
    xlabel=r"$\mathbf{Mean\ success\ rate}$ among all participants",
    xformatter=ticker.PercentFormatter(xmax=1),
    text_format={
        "hpad": 0.01,
        "formatter": lambda x: str(int(round(x, 2) * 100)) + "%",
    },
)

# %% Look at participant information

import altair as alt

participants = pd.read_csv(
    "participants.csv",
    dtype=object,
)[["id", "exp"]]

assert sorted(data["userID"]) == sorted(participants["id"])

pdata = pd.merge(
    data,
    participants,
    left_on="userID",
    right_on="id",
)

pdata["exp"] = pdata["exp"].astype(int)

alt.Chart(pdata).mark_boxplot().encode(
    alt.X("exp:Q"),
    alt.Y("taskTime:Q"),
).save(
    "output/exp-taskTime.html",
)

alt.layer(
    alt.Chart()
    .mark_errorbar(extent="ci")
    .encode(
        alt.X("exp:Q"),
        alt.Y("success_rate:Q"),
    ),
    alt.Chart()
    .mark_point(
        filled=True,
        color="black",
    )
    .encode(
        alt.X("exp:Q"),
        alt.Y("mean(success_rate):Q"),
    ),
    data=pdata,
).save(
    "output/exp-success_rate.html",
)
