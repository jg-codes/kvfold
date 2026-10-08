"""kvfold: fold discarded KV-cache entries back into the kept ones (a wrapper for any kvpress press)."""
from .merging_any_press import (
    COMPENSATIONS,
    Compensation,
    CompensationResult,
    FoldRouting,
    KeptSet,
    MassFold,
    MergingAnyPress,
    NoCompensation,
    install_bias_clearing,
    map_scores,
    patch_attention_bias,
    scatter_add_atomic,
    scatter_add_sorted,
)

__version__ = "0.1.1"
__all__ = [
    "COMPENSATIONS", "Compensation", "CompensationResult", "FoldRouting", "KeptSet", "MassFold", "MergingAnyPress",
    "NoCompensation", "install_bias_clearing", "map_scores", "patch_attention_bias", "scatter_add_atomic", "scatter_add_sorted",
]
