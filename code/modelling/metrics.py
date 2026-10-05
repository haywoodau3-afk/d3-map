"""Validation metrics for leave-one-out predictions."""
from __future__ import annotations
import math
from collections.abc import Sequence

def rmse(observed: Sequence[float], predicted: Sequence[float]) -> float:
    if len(observed) != len(predicted) or not observed:
        raise ValueError("observed and predicted must have the same nonzero length")
    return math.sqrt(sum((float(a)-float(b))**2 for a,b in zip(observed,predicted))/len(observed))

def r2(observed: Sequence[float], predicted: Sequence[float]) -> float:
    if len(observed) != len(predicted) or len(observed)<2:
        raise ValueError("R2 requires matching vectors with at least two observations")
    mean=sum(map(float,observed))/len(observed)
    total=sum((float(y)-mean)**2 for y in observed)
    if total==0: return float("nan")
    residual=sum((float(y)-float(p))**2 for y,p in zip(observed,predicted))
    return 1.0-residual/total
