# %% Import

import lib

import importlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import polars as pl

from cmdstanpy import CmdStanModel

importlib.reload(lib)

# Color-blind-friendly color schemes: https://personal.sron.nl/~pault/

# % % Load interface metadata

interfaces = pl.read_csv("interfaces.csv").with_row_index(
    "interface_id",
    offset=1,
)

interface_groups = pl.read_csv("interface_groups.csv")
participants = pl.read_csv("participants.csv")

# % % Load (wide) data

wide = pl.read_csv("data.csv")[
    [
        "userID",
        "interface",
        "correct1",
        "correct2",
        "correct3",
        "taskTime1",
        "taskTime2",
        "taskTime3",
    ]
]

for task in [1, 2, 3]:
    wide = wide.with_columns(
        # Consider "skip" unsuccessful
        pl.col(f"correct{task}").fill_null(pl.lit(False)),
        # Express task time in minutes
        pl.col(f"taskTime{task}").cast(float) / (1000 * 60),
    )

wide = (
    wide.join(participants, left_on="userID", right_on="id")
    .join(interfaces, left_on="interface", right_on="interface_tag")
    .join(
        interface_groups,
        on="interface_group",
    )
)

# # % % Convert to long format
#
# long = lib.wide_to_long(
#     wide,
#     index=["userID", "interface"],
#     stubnames=["taskTime", "correct"],
#     suffixes=[1, 2, 3],
#     suffix_name="task",
# ).with_columns(
#     success_rate=pl.col("correct").cast(float) / 3,
# )
#
# # % % Aggregate over tasks
#
# data = (
#     long.group_by("userID", "interface")
#     .agg(
#         pl.col("taskTime").sum(),
#         pl.col("correct").all(),
#         pl.col("success_rate").sum(),
#     )
#     .join(participants, left_on="userID", right_on="id")
#     .join(interfaces, left_on="interface", right_on="interface_tag")
#     .join(
#         interface_groups,
#         on="interface_group",
#     )
# )
#
# # % % Distribution of features
#
# lib.feature_histogram(
#     data,
#     group_feature="interface_label",
#     value_feature="exp",
#     sort_feature="interface_order",
#     mode="discrete",
#     xticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     xlabel=r"$\mathbf{Self\!-\!reported\ experience}$ among all participants",
#     size="medium",
#     spacing=0.5,
# )[0].save(
#     "output/hist_exp.pdf",
#     bbox_inches="tight",
# )
#
# lib.feature_histogram(
#     data.filter(pl.col("correct")),
#     group_feature="interface_label",
#     value_feature="taskTime",
#     sort_feature="interface_order",
#     mode="continuous",
#     xticks=np.arange(0, 91, 10),
#     xlabel=r"$\mathbf{Time\ taken}$ (in minutes) among successful participants",
#     size="small",
# )[0].save(
#     "output/hist_tt.pdf",
#     bbox_inches="tight",
# )
#
# lib.feature_histogram(
#     data,
#     group_feature="interface_label",
#     value_feature="success_rate",
#     sort_feature="interface_order",
#     mode="discrete",
#     xticks=[0, 0.333, 0.667, 1],
#     xlabel=r"$\mathbf{Success\ rate}$ among all participants",
#     size="large",
#     xformatter=ticker.PercentFormatter(xmax=1),
# )[0].save(
#     "output/hist_sr.pdf",
#     bbox_inches="tight",
# )

# %% Bayes

model = CmdStanModel(stan_file="model.stan")

fit = model.sample(
    data={
        "N": len(wide),
        "T": 3,
        "I": len(wide["interface_id"].unique()),
        "interface": wide["interface_id"].to_numpy(),
        "correct": wide[["correct1", "correct2", "correct3"]].to_numpy(),
    }
)

fit.summary()

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
    experiment_colors={},
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
                method="BCa",
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
            xerr=[
                [row["estimate"] - row["low"]],
                [row["high"] - row["estimate"]],
            ],
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
    ax.set_xlabel(
        xlabel if xlabel else f"$\\mathbf{{{estimator_name}\\ {feature}}}$"
    )

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

# %% Calculate effect sizes

dfs = []

for int1, g1 in data.groupby("interface"):
    name2 = []
    success_rate_est = []
    success_rate_lo = []
    success_rate_hi = []
    task_time_est = []
    task_time_lo = []
    task_time_hi = []
    for int2, g2 in data.groupby("interface"):
        if int1 == int2:
            continue

        name2.append(int2)

        sr = stats.bootstrap(
            (g1["success_rate"], g2["success_rate"]),
            lambda s1, s2: np.mean(s2) - np.mean(s1),
            n_resamples=99999,
            confidence_level=0.95,
            alternative="two-sided",
            method="BCa",
            random_state=0,
        ).confidence_interval
        success_rate_est.append(
            np.mean(g2["success_rate"]) - np.mean(g1["success_rate"])
        )
        success_rate_lo.append(sr.low)
        success_rate_hi.append(sr.high)

        c1 = g1[g1["correct"]]
        c2 = g2[g2["correct"]]

        tt = stats.bootstrap(
            (c1["taskTime"], c2["taskTime"]),
            lambda s1, s2: np.median(s2) - np.median(s1),
            n_resamples=99999,
            confidence_level=0.95,
            alternative="two-sided",
            method="BCa",
            random_state=0,
        ).confidence_interval
        task_time_est.append(
            np.median(c2["taskTime"]) - np.median(c1["taskTime"])
        )
        task_time_lo.append(tt.low)
        task_time_hi.append(tt.high)

    dfs.append(
        pd.DataFrame(
            {
                "name1": int1,
                "name2": name2,
                "success_rate_est": success_rate_est,
                "success_rate_lo": success_rate_lo,
                "success_rate_hi": success_rate_hi,
                "task_time_est": task_time_est,
                "task_time_lo": task_time_lo,
                "task_time_hi": task_time_hi,
            }
        )
    )

effect_sizes = pd.concat(dfs)
effect_sizes

# %% Plot effect sizes

reference = "Basic-Control"
df = effect_sizes[effect_sizes["name1"] == reference].copy()
df["order"] = df["name2"].apply(lambda n: INTERFACE_ORDER.index(n))
df.sort_values(by="order", ascending=False, inplace=True)

for feature, flow, fhigh, fstep in [
    ("success_rate", -0.6, 0.6, 0.2),
    ("task_time", -40, 40, 5),
]:
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    yticks = np.arange(0, len(df))
    yticklabels = df["name2"]

    ax.errorbar(
        df[f"{feature}_est"],
        yticks,
        xerr=[
            df[f"{feature}_est"] - df[f"{feature}_lo"],
            df[f"{feature}_hi"] - df[f"{feature}_est"],
        ],
        color="black",
        fmt="o",
        zorder=10,
    )
    ax.set_yticks(yticks, labels=yticklabels)

    ax.set_xticks(np.arange(flow, fhigh + 0.0001, fstep))
    ax.set_xlabel(f"{feature} compared to {reference}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.axvline(0, ls="--", c="lightgray", zorder=1)
    fig.tight_layout()
    fig.savefig(f"output/ES_{feature}.pdf")
    plt.close(fig)


# %% Look at participant information

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

# alt.Chart(pdata).mark_boxplot().encode(
#     alt.X("exp:Q"),
#     alt.Y("taskTime:Q"),
# ).save(
#     "output/exp-taskTime.html",
# )
#
# alt.layer(
#     alt.Chart()
#     .mark_errorbar(extent="ci")
#     .encode(
#         alt.X("exp:Q"),
#         alt.Y("success_rate:Q"),
#     ),
#     alt.Chart()
#     .mark_point(
#         filled=True,
#         color="black",
#     )
#     .encode(
#         alt.X("exp:Q"),
#         alt.Y("mean(success_rate):Q"),
#     ),
#     data=pdata,
# ).save(
#     "output/exp-success_rate.html",
# )
