# %% Import

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

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
        y=alt.Y("interface:N", sort=order),
    ),
    alt.Chart()
    .mark_errorbar(extent="ci")
    .encode(
        x="lo:Q",
        x2="hi:Q",
        y=alt.Y("interface:N", sort=order),
    ),
    data=df,
).interactive().save("chart4.html")
