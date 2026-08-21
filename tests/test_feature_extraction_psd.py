import numpy as np

from feature_extraction_psd import select_and_rereference


def test_select_and_rereference_uses_only_selected_channels():
    x = np.array([[[1.0, 2.0], [4.0, 8.0], [10.0, 20.0]]], dtype=np.float32)

    rereferenced, channels = select_and_rereference(
        x, ["F7", "Cz", "F8"], ["F7", "F8"]
    )

    np.testing.assert_allclose(rereferenced.mean(axis=1), 0.0)
    np.testing.assert_allclose(rereferenced[:, 1] - rereferenced[:, 0], x[:, 2] - x[:, 0])
    assert channels == ["F7", "F8"]
