import numpy as np

from train_helpers import cap_subject_epochs


def test_cap_subject_epochs_is_reproducible_and_preserves_short_subjects():
    subjects = [
        {"participant_id": "long", "x": np.arange(20)[:, None], "n_epochs": 20},
        {"participant_id": "short", "x": np.arange(5)[:, None], "n_epochs": 5},
    ]

    first = cap_subject_epochs(subjects, 10, random_state=7)
    second = cap_subject_epochs(subjects, 10, random_state=7)

    np.testing.assert_array_equal(first[0]["x"], second[0]["x"])
    np.testing.assert_array_equal(first[1]["x"], subjects[1]["x"])
    assert [subject["n_epochs"] for subject in first] == [10, 5]
