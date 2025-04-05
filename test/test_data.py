"""Testing functions in data.py."""

from al_pipe.util.data import Data


def test_data_init(path="./dataset/random_promo", data_name="sub_sample_pTpA_All.csv"):  # noqa: D103
    # how do you test it with data
    data_class = Data(path, data_name)
    assert data_class is not None
