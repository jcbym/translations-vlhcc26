import arviz_stats as azs
import matplotlib
import matplotlib.pyplot as plt
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
    measure,
    better_notion,
    xticks,
    worse_notion,
    better,
    labels,
    colors,
    bins,
    step,
    figsize,
    round_amount,
    hdi_prob=0.95,
    fontsize=9,
    xlabel_fontsize=14,
):
    assert better in {"greater", "less"}
    assert round_amount in {1, 2}

    es = es_draws[:, :, :, 0].mean(axis=0).T[1:]
    labels = labels[1:]
    colors = colors[1:]

    N = len(labels)
    fig, ax = plt.subplots(
        N,
        1,
        figsize=figsize,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    for i in range(N):
        n, _, patches = ax[i].hist(
            es[i],
            bins=bins,
            color="0.8",
            edgecolor="0",
            linewidth=0.00,
        )
        ymax = 1.1 * max(n)
        ax[i].set_ylim(0, ymax)
        ax[i].set_yticks([ymax / 2], labels=[labels[i]])
        ax[i].get_yticklabels()[0].set_color(colors[i])
        for p in patches:
            if (better == "greater" and p.get_x() > -0.00001) or (
                better == "less" and p.get_x() + p.get_width() < 0
            ):
                p.set_facecolor(colors[i])
        if xticks is not None:
            xt = xticks
        else:
            xt = np.arange(min(bins), max(bins) + step / 2, step)
        ax[i].set_xticks(xt, labels=xt, fontsize=fontsize)
        if i < N - 1:
            ax[i].set_xticks([])

        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        ax[i].axvline(x=0, c="black", lw=1)
        ax[i].tick_params(axis="both", which="both", length=0)

        if better == "greater":
            pb = (es[i] > 0).mean()
            pb_x = max(bins)
            pb_ha = "right"

            pw_x = min(bins)
            pw_ha = "left"
        else:
            pb = (es[i] < 0).mean()
            pb_x = min(bins)
            pb_ha = "left"

            pw_x = max(bins)
            pw_ha = "right"

        ax[i].text(
            pb_x,
            ymax / 2,
            r"$\mathbb{P}(\sf{"
            + better_notion
            + r"}) \approx $"
            + f"{pb:0.2f}",
            ha=pb_ha,
            va="bottom",
            color=colors[i],
            fontsize=fontsize,
        )

        ax[i].text(
            pw_x,
            ymax / 2,
            r"$\mathbb{P}(\sf{"
            + worse_notion
            + r"}) \approx $"
            + f"{1 - pb:0.2f}",
            ha=pw_ha,
            va="bottom",
            color="0.4",
            fontsize=fontsize,
        )

        am = np.argmax(n)
        mode = (bins[am] + bins[am + 1]) / 2
        # ax[i].text(
        #     mode - 0.4,
        #     0.9 * ymax,
        #     f"mode = {mode:0.2f}",
        #     fontsize=7,
        #     color=colors[i],
        #     va="top",
        #     ha="right",
        # )
        ax[i].text(
            mode,
            0.6 * ymax,
            f"{mode:+0.2f}" if round_amount == 2 else f"{mode:+0.1f}",
            fontsize=fontsize,
            color=colors[i],
            va="top",
            ha="center",
            bbox=dict(
                pad=0.5,
                facecolor="1",
                ec=colors[i],
                lw=0.5,
            ),
        )
        ax[i].axvline(x=mode, lw=0.5, c="1")

        lo, hi = azs.hdi(es[i], prob=hdi_prob)
        ax[i].plot(
            [lo, hi],
            [0.1 * ymax, 0.1 * ymax],
            color="1",
            lw=0.5,
        )

        ax[i].text(
            lo,
            0.2 * ymax,
            f"{lo:+0.2f}" if round_amount == 2 else f"{lo:+0.1f}",
            fontsize=fontsize,
            color=colors[i],
            va="bottom",
            ha="right",
            bbox=dict(
                pad=0.5,
                facecolor="1",
                ec=colors[i],
                lw=0.5,
            ),
        )

        ax[i].text(
            hi,
            0.2 * ymax,
            f"{hi:+0.2f}" if round_amount == 2 else f"{hi:+0.1f}",
            fontsize=fontsize,
            color=colors[i],
            va="bottom",
            ha="left",
            bbox=dict(
                pad=0.5,
                facecolor="1",
                ec=colors[i],
                lw=0.5,
            ),
        )

    # offset = 0.15 * (1 if better == "greater" else -1)

    # ax[-1].set_xlabel(
    #     # r"$\bf{" + measure.replace(" ", r"\ ") + r"}$ effect size"
    #     f"{measure} (effect size)",
    #     fontsize=xlabel_fontsize,
    # )
    # better_label = "    Better →" if better == "greater" else "← Better    "
    # ax[0].text(0, 1.1 * ymax, better_label, ha="center", va="bottom")

    # ax[-1].annotate(
    #     "Better",
    #     xy=(0.5 + offset, -0.5),
    #     xytext=(0.5, -0.5),
    #     xycoords="axes fraction",
    #     arrowprops=dict(
    #         arrowstyle="-|>",
    #         facecolor="black",
    #         relpos=(0.5 + offset / 2, 0.5),
    #         lw=3,
    #     ),
    #     va="center",
    #     ha="center",
    #     weight="bold",
    # )

    fig.tight_layout()
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
    label_fontsize,
    compressed=False,
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
    bars = ax.bar(
        xticks,
        data["sum"],
        color=[(c, 0.5) for c in data["color"]],
        edgecolor=[(c, 1) for c in data["color"]],
    )
    ax.bar_label(
        bars,
        [
            f"{r['sum']}\n({r['sum'] / r['total']:.0%})"
            for r in data.iter_rows(named=True)
        ],
        padding=2 if compressed else 3,
        fontsize=3 if compressed else 1.5 * label_fontsize,
    )
    ax.set_xticks(
        xticks,
        labels=data[group_feature],
        fontsize=label_fontsize,
        fontweight="bold",
        rotation="vertical" if compressed else None,
    )
    for xt, bl, c in zip(ax.get_xticklabels(), ax.texts, data["color"]):
        bl.set_color(c)
        xt.set_color(c)
    ax.set_yticks(yticks, labels=yticks, fontsize=label_fontsize * 1.5)
    ax.set_ylim(min(yticks), max(yticks) + 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # ax.yaxis.grid(color="1", linewidth=0.5)
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
    label_fontsize,
    caption=None,
    show_boxplots=False,
    compressed=False,
):
    assert df[value_feature].is_between(min(yticks), max(yticks)).all()

    np.random.seed(0)

    labels = []
    colors = []
    vals = []

    if caption:
        print(caption, end=" ")

    for (label,), g in df.sort(
        by=sort_feature,
    ).group_by(
        group_feature,
        maintain_order=True,
    ):
        labels.append(label)
        colors.append(g[color_feature].first())
        vals.append(g[value_feature])

        q1 = round(np.quantile(g[value_feature], 0.25), 1)
        q2 = round(np.quantile(g[value_feature], 0.5), 1)
        q3 = round(np.quantile(g[value_feature], 0.75), 1)

        if caption:
            print(f"{label}, median {q2}, (Q1 {q1}, Q3 {q3}).", end=" ")

    if caption:
        print()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    xticks = np.arange(len(labels))
    for i in range(len(labels)):
        spread = 0.1
        x = xticks[i] + np.random.uniform(-spread, spread, size=len(vals[i]))
        ax.scatter(
            x,
            vals[i],
            c=colors[i],
            alpha=0.3,
        )
        if not show_boxplots:
            med = vals[i].median()
            ax.hlines(
                y=med,
                xmin=xticks[i] - 0.3,
                xmax=xticks[i] + 0.3,
                color=colors[i],
                lw=2,
            )
            ax.annotate(
                f"{med:.1f}",
                xy=(xticks[i], med),
                xytext=(0, 1),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=colors[i],
                fontsize=4.5 if compressed else 7,
                bbox=dict(
                    boxstyle="square,pad=0.05",
                    fc="1",
                    ec=colors[i],
                    lw=0.5,
                    alpha=0.8,
                ),
            )

    ax.set_xticks(
        xticks,
        labels=labels,
        fontsize=label_fontsize,
        rotation="vertical" if compressed else None,
        fontweight="bold",
    )
    for xt, c in zip(ax.get_xticklabels(), colors):
        xt.set_color(c)

    if show_boxplots:
        bplot = ax.boxplot(
            vals,
            positions=xticks,
            patch_artist=True,
            showcaps=False,
        )
        boxplot_alpha = 1
        for box, flier, med, v, c in zip(
            bplot["boxes"],
            bplot["fliers"],
            bplot["medians"],
            vals,
            colors,
        ):
            box.set_facecolor((c, 0.5))
            box.set_edgecolor((c, 1))
            box.set_linewidth(1)
            med.set_alpha(boxplot_alpha)
            med.set_color(c)
            med.set_linewidth(2)
            flier.set(
                marker="o",
                markerfacecolor=c,
                markeredgecolor="1",
                linewidth=0,
            )
            x = med.get_xdata().mean()
            y = med.get_ydata().mean()
            ax.text(
                x,
                y,
                f"{y:.1f}",
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

    ax.set_yticks(yticks, labels=yticks, fontsize=label_fontsize * 1.5)
    ax.set_ylim(min(yticks), max(yticks))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    fig.tight_layout()
    return fig, ax


def teaser_plot(data, *, color):
    fig, ax = plt.subplots(1, 1, figsize=(5, 0.8))

    ax.set_yticks([0], labels=[""])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    xticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(
        xticks,
        labels=["0%", "25%", "50%", "75%", "100%"],
    )
    for left, right in zip(xticks, xticks[1:]):
        ax.plot(
            [left, right],
            [0, 0],
            marker="|",
            color="0.8",
        )

    y = np.zeros_like(data)
    ax.scatter(
        data,
        y,
        color=color,
        s=80,
        zorder=10,
    )

    fig.tight_layout()
    return fig, ax
