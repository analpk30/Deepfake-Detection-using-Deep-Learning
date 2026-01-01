import numpy as np
from eval import aggregate_scores


def test_aggregate_scores_mean_median_majority():
    scores = [0.1, 0.9, 0.8, 0.2]
    s_mean, l_mean = aggregate_scores(scores, method='mean', threshold=0.5)
    assert abs(s_mean - np.mean(scores)) < 1e-6
    assert l_mean == 1

    s_median, l_median = aggregate_scores(scores, method='median', threshold=0.5)
    assert abs(s_median - np.median(scores)) < 1e-6

    s_maj, l_maj = aggregate_scores(scores, method='majority', threshold=0.5)
    # two scores >= 0.5 out of four -> fraction 0.5 -> label 1
    assert abs(s_maj - 0.5) < 1e-6
    assert l_maj == 1
