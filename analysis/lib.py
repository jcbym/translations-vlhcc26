import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import os

################################################################################
## Monkey-patch matplotlib


def _save(self, filename, *args, **kwargs):
    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True,
    )
    self.savefig(filename, *args, **kwargs)
    plt.close(self)


matplotlib.figure.Figure.save = _save

################################################################################
## Polars helpers


def wide_to_long(df, *, index, stubnames, suffixes, suffix_name):
    dfs = []
    for suffix in suffixes:
        exprs = (
            [pl.col(col) for col in index]
            + [pl.lit(suffix).alias(suffix_name)]
            + [pl.col(f"{stub}{suffix}").alias(stub) for stub in stubnames]
        )
        dfs.append(df.select(*exprs))
    return pl.concat(dfs, how="vertical")


################################################################################
## Plots


def feature_histogram(
    df,
    *,
    group_feature,
    value_feature,
    sort_feature,
    mode,
    xticks,
    xlabel,
    size,
    xformatter=None,
    spacing=0.2,  # only applies if mode is "discrete"
    color="0.8",
    edgecolor="0.6",
    median_color="#009988",
    mean_color="#CC3311",
):
    lo = min(xticks)
    hi = max(xticks)

    too_low = df[value_feature] < lo
    too_high = df[value_feature] > hi

    if too_low.sum() > 0:
        print(f"WARNING: {too_low.sum()} too low:")
        print(df[value_feature].filter(too_low))

    if too_high.sum() > 0:
        print(f"WARNING: {too_high.sum()} too high:")
        print(df[value_feature].filter(too_high))

    groups = list(
        df.sort(
            by=sort_feature,
        ).group_by(
            group_feature,
            maintain_order=True,
        )
    )

    fig, ax = plt.subplots(
        len(groups),
        1,
        figsize=(8, 6),
        layout="constrained",
    )
    fig.get_layout_engine().set(hspace=0.05)

    if size == "large":
        yticks = [0, 2, 4, 6, 8, 10, 12]
        yticklabels = ["0", "", "4", "", "8", "", "12"]
    elif size == "medium":
        yticks = [0, 2, 4, 6, 8]
        yticklabels = ["0", "", "4", "", "8"]
    elif size == "small":
        yticks = [0, 1, 2, 3, 4]
        yticklabels = ["0", "", "2", "", "4"]
    else:
        raise ValueError(f"Unknown size '{size}'")

    for i, ((label,), group) in enumerate(groups):
        vals = group[value_feature]

        if mode == "continuous":
            ax[i].hist(
                vals,
                bins=xticks,
                color=color,
                edgecolor=edgecolor,
            )
            ax[i].set_xlim(lo, hi)
            ax[i].set_xticks(xticks)
        elif mode == "discrete":
            xs, counts = np.unique(
                vals,
                return_counts=True,
            )
            ax[i].bar(
                xs,
                counts,
                color=color,
                edgecolor=edgecolor,
                width=spacing,
            )
            ax[i].set_xlim(lo - spacing, hi + spacing)
            ax[i].set_xticks(xticks)
        else:
            print(f"WARNING: unknown mode '{mode}'")

        median = vals.median()
        ax[i].axvline(
            x=median,
            c=median_color,
            lw=2,
            clip_on=False,
            zorder=100,
        )
        ax[i].scatter(
            [median],
            [0],
            c=median_color,
            marker="^",
            clip_on=False,
            label="Median",
            zorder=100,
            s=50,
        )

        mean = vals.mean()
        ax[i].axvline(
            x=mean,
            c=mean_color,
            lw=2,
            clip_on=False,
            zorder=100,
        )
        ax[i].scatter(
            [mean],
            [0],
            c=mean_color,
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
            label,
            ha="right",
            va="center",
            fontweight="bold",
            transform=ax[i].transAxes,
        )

    ax[-1].set_xlabel(
        xlabel,
        fontsize=10,
    )

    return fig, ax
