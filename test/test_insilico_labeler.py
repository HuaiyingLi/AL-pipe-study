"""Test the insilico labeler."""

import os

import pandas as pd
import pytest

from al_pipe.data.dna_dataset import DNADataset


@pytest.fixture
def test_insilico_labeler():
    # create a small test dataset
    test_data = pd.DataFrame({"sequences": ["ATCG", "ACCT", "ATTC"], "values": [1, 2, 3]})
    os.makedirs("./dataset/test", exist_ok=True)
    test_data.to_csv("./dataset/test/test_data.csv", index=False, sep="\t")
    return test_data


def test_insilico_labeler_returns_labels():
    # create an insilico labeler
    insilico_labeler = DNADataset(data_path="./dataset/test", data_name="test_data.csv", batch_size=1, train_val_test_pool_split=[0.5, 0.25, 0.25], max_length=4, embedding_model=None)
    labels = insilico_labeler.return_label(["ACCT", "ATCG", "ATTC"])
    assert labels == [2, 1, 3]
