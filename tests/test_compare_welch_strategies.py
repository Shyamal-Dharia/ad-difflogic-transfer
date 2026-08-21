import numpy as np

from compare_welch_strategies import epochwise_subject_features


def test_epochwise_subject_features_returns_channel_band_vector():
    sfreq = 100
    time = np.arange(4 * sfreq) / sfreq
    epoch = np.stack(
        [np.sin(2 * np.pi * 6 * time), np.sin(2 * np.pi * 10 * time)]
    )
    x = np.stack([epoch, epoch])

    features = epochwise_subject_features(x, sfreq, 2.0, 1.0)

    assert features.shape == (10,)
    assert np.isfinite(features).all()
