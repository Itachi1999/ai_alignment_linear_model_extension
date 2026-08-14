from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt


def plot_violation_rates(
    labels: Sequence[str],
    po_rates: Sequence[float],
    pmc_rates: Sequence[float],
    total_rates: Sequence[float],
) -> plt.Figure:
    """
    Plot PO, PMC, and total violation percentages.
    """

    figure, axis = plt.subplots(figsize=(8, 5))

    x = range(len(labels))
    width = 0.25

    axis.bar(
        [i - width for i in x],
        po_rates,
        width=width,
        label="PO",
    )

    axis.bar(
        x,
        pmc_rates,
        width=width,
        label="PMC",
    )

    axis.bar(
        [i + width for i in x],
        total_rates,
        width=width,
        label="Total",
    )

    axis.set_xlabel("Model")
    axis.set_ylabel("Violation Rate (%)")
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels)
    axis.legend()

    figure.tight_layout()

    return figure