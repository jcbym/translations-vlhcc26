import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import polars as pl
import os

################################################################################
## Monkey-patch matplotlib


def _save(self, filename, *args, **kwargs):
    d = os.path.dirname(filename)
    if d:
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


def es_plot(
    es_draws,
    *,
    better,
    labels,
    bins,
    step,
    figsize,
):
    assert better in {"greater", "less"}

    es = es_draws[:, :, :, 0].mean(axis=0).T[1:]
    labels = labels[1:]
    N = len(labels)
    fig, ax = plt.subplots(N, 1, figsize=figsize)

    for i in range(N):
        assert (es[i] < max(bins)).all()
        assert (es[i] > min(bins)).all()
        n, _, patches = ax[i].hist(es[i], bins=bins)
        ax[i].set_ylim(0, max(n))
        ax[i].set_yticks([max(n) / 2], labels=[labels[i]])
        for p in patches:
            if (better == "greater" and p.get_x() > 0) or (
                better == "less" and p.get_x() < 0
            ):
                p.set_facecolor("pink")
        ax[i].set_xticks(np.arange(min(bins), max(bins) + step / 2, step))
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        # ax[i].axes.get_yaxis().set_visible(False)
        ax[i].axvline(x=0, c="red")
        ax[i].tick_params(axis="both", which="both", length=0)

    return fig, ax


def count_comparison_plot(
    df,
    *,
    group_feature,
    value_feature,
    sort_feature,
    color_feature,
    step,
    figsize,
):
    data = (
        df.sort(by=sort_feature)
        .group_by(group_feature, maintain_order=True)
        .agg(
            pl.col(value_feature).sum().alias("sum"),
            pl.col(value_feature).count().alias("total"),
            pl.col(color_feature).first().alias("color"),
        )
    )
    yticks = np.arange(0, max(data["total"]) + step, step)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    xticks = np.arange(len(data))
    bars = ax.bar(xticks, data["sum"], color=data["color"])
    ax.bar_label(
        bars,
        [
            f"{r['sum']}\n({r['sum'] / r['total']:.0%})"
            for r in data.iter_rows(named=True)
        ],
        padding=3,
    )
    ax.set_xticks(
        xticks,
        labels=data[group_feature],
        fontsize=9,
    )
    for xt, bl, c in zip(ax.get_xticklabels(), ax.texts, data["color"]):
        bl.set_color(c)
        xt.set_color(c)
    ax.set_yticks(yticks)
    ax.set_ylim(min(yticks), max(yticks) + 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(color="1", linewidth=0.5)
    ax.tick_params(axis="both", which="both", length=0)
    fig.tight_layout()
    return fig, ax


def distribution_comparison_plot(
    df,
    *,
    group_feature,
    value_feature,
    sort_feature,
    color_feature,
    yticks,
    figsize,
):
    assert df[value_feature].is_between(min(yticks), max(yticks)).all()

    labels = []
    colors = []
    vals = []

    for (label,), g in df.sort(
        by=sort_feature,
    ).group_by(
        group_feature,
        maintain_order=True,
    ):
        labels.append(label)
        colors.append(g[color_feature].first())
        vals.append(g[value_feature])
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    xticks = np.arange(len(labels))
    bplot = ax.boxplot(
        vals,
        positions=xticks,
        patch_artist=True,
        showcaps=False,
        boxprops=dict(linewidth=1, fill=None),
        medianprops=dict(linewidth=2, color="0"),
    )
    ax.set_xticks(
        xticks,
        labels=labels,
        fontsize=9,
    )
    boxplot_alpha = 1
    for box, flier, med, xt, v, c in zip(
        bplot["boxes"],
        bplot["fliers"],
        bplot["medians"],
        ax.get_xticklabels(),
        vals,
        colors,
    ):
        box.set_edgecolor(c)
        box.set_alpha(boxplot_alpha)
        xt.set_color(c)
        med.set_color(c)
        med.set_alpha(boxplot_alpha)
        med.set_linewidth(2)
        flier.set(
            marker="",
            markerfacecolor=c,
            markeredgecolor="1",
            linewidth=0,
        )
        x = med.get_xdata().mean()
        y = med.get_ydata().mean()
        ax.text(
            x,
            y,
            "",  # f"{y:.1f}",
            ha="center",
            va="center",
            color="1",
            fontsize=7,
            bbox=dict(
                boxstyle="square,pad=0.05",
                fc=c,
                ec="none",
            ),
        )
        np.random.seed(0)
        spread = 0.3
        ax.scatter(
            x + np.random.uniform(low=-spread, high=spread, size=len(v)),
            v,
            color=c,
            zorder=10,
            s=20,
            alpha=1,
            lw=0,
            marker="",
            # ec="0",
        )
    for i, whis in enumerate(bplot["whiskers"]):
        c = colors[i // 2]
        whis.set_color(c)
        whis.set_alpha(boxplot_alpha)
        whis.set_linewidth(1)
    ax.set_yticks(yticks)
    ax.set_ylim(min(yticks), max(yticks))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    fig.tight_layout()
    return fig, ax


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
