import numpy as np

from analyze_epoch_count_stability import subsample_scores


def test_subsample_scores_is_reproducible():
    subjects = [np.arange(10, dtype=float), np.arange(10, 20, dtype=float)]

    first = subsample_scores(subjects, 5, 4, np.random.default_rng(3))
    second = subsample_scores(subjects, 5, 4, np.random.default_rng(3))

    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 2)
