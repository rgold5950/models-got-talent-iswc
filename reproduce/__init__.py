"""Self-contained reproduction of the ISWC'26 paper results.

See reproduce/README.md. Nothing in this package imports torch or ray, so the
paper's tables and figure can be regenerated from a clone in a few seconds.
"""

from reproduce.metrics import (
    ORDERED_DATASETS,
    ZC_METRICS,
    ALL_PROXIES,
    compute_stats,
    compute_all_stats,
)

__all__ = [
    "ORDERED_DATASETS",
    "ZC_METRICS",
    "ALL_PROXIES",
    "compute_stats",
    "compute_all_stats",
]
