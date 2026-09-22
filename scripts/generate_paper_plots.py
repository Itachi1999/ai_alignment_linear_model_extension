from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    path = "paper_results/mallows_dispersion/20260922_001832/trials.csv"
    CSV_PATH = Path(path)
    OUTPUT_PATH = Path("paper/figures/mallows_dispersion")
    if Path(OUTPUT_PATH).exists() is False:
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    SELECTED_MODELS = [
        "LP2",
        "LP3",
        "Linear",
    ]

    df = pd.read_csv(CSV_PATH)
    print(df.head())
    df = df[
        df["model"].astype(str).str.startswith(
            tuple(SELECTED_MODELS)
        )
    ].copy()
    if df.empty:
        raise ValueError(
            f"No data found for selected models: {SELECTED_MODELS}"
        )

    # Optional: verify the resulting models
    print(df["model"].unique(), "unique models")
    print(df.head())
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=3.0,
        
    )

    palette = sns.color_palette(
        "colorblind",
        n_colors=len(SELECTED_MODELS),
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16.0, 4.0),
        sharex=True,
        # gridspec_kw={
        #     "hspace": 0.10,
        #     "wspace": 0.10,
        # }
    )

    plots = [
        ("violation_percentage", "Total Violations", "Violation rate (%)"),
        ("po_violation_percentage", "PO Violations", "Violation rate (%)"),
        ("pmc_violation_percentage", "Majority Violations", "Violation rate (%)"),
        ("epsilon_l1_norm", "Candidate-wise slack", r"$\|\varepsilon\|_1$"),
    ]

    for ax, (column, title, ylabel) in zip(axes, plots):

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found in CSV."
            )

        sns.lineplot(
            data=df,
            x="phi",
            y=column,
            hue="model",
            style="model",
            palette=palette,
            marker="o",
            markersize=4,
            linewidth=2.5,
            errorbar=("ci", 95),
            ax=ax,
            # legend=False,
        )
        ax.yaxis.get_offset_text().set_fontsize(12)
        ax.set_title(
            title,
            fontsize=18,
            pad=6,
        )

        ax.set_xlabel(
            r"Mallows dispersion $\phi$",
            fontsize=16,
        )

        ax.set_ylabel(
            ylabel,
            fontsize=16,
        )

        ax.tick_params(
            labelsize=10.5,
        )

        # ax.set_xticks(
        #     sorted(df["phi"].unique())
        # )

        ax.grid(
            True,
            which="major",
            axis="both",
            alpha=0.25,
            linewidth=0.7,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)   

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(SELECTED_MODELS),
        frameon=False,
        fontsize=16,
        handlelength=2.5,
        columnspacing=1.8,
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        top=0.88,
        bottom=0.23,
        wspace=0.25,
    )

    fig.savefig(
        OUTPUT_PATH / "mallows_dispersion_summary_3_6_5.pdf",
        bbox_inches="tight",
        dpi=1200
    )

    fig.savefig(
        OUTPUT_PATH / "mallows_selected.png",
        dpi=900,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()