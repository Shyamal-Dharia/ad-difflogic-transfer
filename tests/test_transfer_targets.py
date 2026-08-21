import numpy as np
import pandas as pd

from train_difflogic import make_epoch_predictions
from train_helpers import alz_c_vs_ad_ftd_label, ds007427_group_label


def test_transfer_target_labels_and_dataset_metadata():
    assert ds007427_group_label("G1") == (1, "G1")
    assert ds007427_group_label("G2") == (0, "G2")
    assert ds007427_group_label("CTR") == (0, "CTR")

    predictions = make_epoch_predictions(
        "transfer",
        1,
        42,
        np.array([[0.25, 0.75]]),
        np.array([1]),
        np.array(["sub-G1000"]),
        pd.DataFrame(
            [{"participant_id": "sub-G1000", "dataset": "DS007427", "group": "G1"}]
        ),
    )
    assert predictions.loc[0, "dataset"] == "DS007427"
    assert predictions.loc[0, "p_ad"] == 0.75


def test_combined_dementia_labels():
    assert alz_c_vs_ad_ftd_label("C") == (0, "C")
    assert alz_c_vs_ad_ftd_label("A") == (1, "A")
    assert alz_c_vs_ad_ftd_label("F") == (1, "F")
