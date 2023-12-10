import numpy as np
import pytest

from src.metric.eer import compute_eer


def test_eer():

    bonafide_scores = np.array([1, 1, 1])
    other_scores = np.array([0, 0 ])

    eer, t = compute_eer(bonafide_scores, other_scores)
    print(f"eer: {eer}, t: {t}")