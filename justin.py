# %% Import

import altair as alt
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

# %% Distribution of task times
# Right skewed so we should use median.

(
    alt.Chart(correct)
    .transform_density(
        density="taskTime",
        groupby=["interface"],
        as_=["taskTime", "density"],
    )
    .mark_area(interpolate="monotone")
    .encode(
        alt.X("taskTime:Q"),
        alt.Y("density:Q"),
    )
    .facet(
        row=alt.Row("interface:N")
        .title(None)
        .header(labelAngle=0, labelAlign="left", format="%B")
    )
    # .properties(bounds="flush")
    # .configure_facet(spacing=0)
    # .configure_view(stroke=None)
    # .configure_title(anchor="end")
    .interactive()
    .save("taskTimes.html")
)


# %% Bootstrap

lows = []
highs = []
inters = []
meds = []

for i, g in correct.groupby("interface"):
    vals = g["taskTime"].values
    inters.append(i)
    res = stats.bootstrap((vals,), np.median, n_resamples=10_000)
    lows.append(res.confidence_interval.low)
    highs.append(res.confidence_interval.high)
    meds.append(np.median(vals))

df = pd.DataFrame({"interface": inters, "lo": lows, "hi": highs, "med": meds})

alt.layer(
    alt.Chart()
    .mark_point(color="black", filled=True)
    .encode(
        x="med:Q",
        y=alt.Y("interface:N", sort=INTERFACE_ORDER),
    ),
    alt.Chart()
    .mark_errorbar(extent="ci")
    .encode(
        x="lo:Q",
        x2="hi:Q",
        y=alt.Y("interface:N", sort=INTERFACE_ORDER),
    ),
    data=df,
).interactive().save("bootstrap_median.html")

# %%


# https://glowingpython.blogspot.com/2020/03/ridgeline-plots-in-pure-matplotlib.html
def ridgeline(ax, data, overlap=0, fill=True, labels=None, n_points=150):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError("overlap must be in [0 1]")
    xx = np.linspace(0, np.max(np.concatenate(data)), n_points)
    dom = np.arange(0, 91, 10)
    ax.set_xticks(dom)
    curves = []
    ys = []
    for i, d in enumerate(data):
        # pdf = stats.gaussian_kde(d)
        y = i * (1.0 - overlap)
        yNext = (i + 1) * (1.0 - overlap)
        ys.append(y)
        ax.hist(
            d,
            bins=dom,
            density=True,
            bottom=y,
            color="gray",  # (i / len(data), 0, 0),
            edgecolor="black",
            zorder=len(data) - i + 1,
        )
        # curve = pdf(xx)
        # if fill:
        #     ax.fill_between(
        #         xx,
        #         np.ones(n_points) * y,
        #         curve + y,
        #         zorder=len(data) - i + 1,
        #         color=fill,
        #     )
        # ax.plot(xx, curve + y, c="k", zorder=len(data) - i + 1)
        ax.axhline(y, c="black")
        med = np.median(d)
        mean = np.mean(d)
        ax.plot((med, med), (y, yNext), color="red")
        ax.plot((mean, mean), (y, yNext), color="blue")
    if labels:
        ax.set_yticks(ys, labels)


res = []
for i, g in correct.groupby("interface"):
    res.append((INTERFACE_ORDER.index(i), i, g["taskTime"]))

res.sort(reverse=True)
_, labels, d = zip(*res)
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ridgeline(ax, d, overlap=0.95, fill="gray", labels=labels)
fig.tight_layout()
ax.spines[["left", "right", "top"]].set_visible(False)
plt.savefig("ridge.pdf")
