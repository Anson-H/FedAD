"""Utility function module"""

from typing import Dict


def _normalize_probs_dict(d: Dict[str, float]) -> Dict[str, float]:
    """Normalize nonnegative dictionary values into a probability distribution."""
    s = float(sum(max(0.0, v) for v in d.values()))
    if s <= 1e-12:
        return {k: 0.0 for k in d}
    return {k: max(0.0, float(v)) / s for k, v in d.items()}
