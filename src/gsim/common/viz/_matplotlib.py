"""Shared Matplotlib layout helpers."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def add_bottom_legend(figure: Figure, ax: Axes) -> None:
    """Add a deduplicated figure legend below an axes."""
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    seen_labels: set[str] = set()

    for handle, label in zip(handles, labels, strict=True):
        if label in seen_labels:
            continue
        unique_handles.append(handle)
        unique_labels.append(label)
        seen_labels.add(label)

    if not unique_handles:
        return

    figure.legend(
        unique_handles,
        unique_labels,
        loc="outside lower center",
        ncols=min(4, len(unique_labels)),
        fancybox=True,
        framealpha=1.0,
    )
