import altair as alt
import pandas as pd

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

plot_hist(
    pdata,
    feature="exp",
    lo=1,
    hi=10,
    step=None,
    xticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    xlabel=r"$\mathbf{Self\!-\!reported\ experience}$ among all participants",
    correct_only=False,
    size="medium",
    spacing=0.5,
)
