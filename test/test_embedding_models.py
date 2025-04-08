"""Test embedding models."""

import pytest
import pandas as pd


@pytest.fixture
def sample_sequences():
    """Create sample DNA sequences for testing."""
    return pd.Series(["ATCG", "GCTA", "AAAA"])


# TODO: revisit these tasks
# def test_onehot_embedding_initialization(sample_sequences):
#     """Test OneHotEmbedding initialization."""
#     embedder = OneHotEmbedding(sample_sequences)
#     assert embedder.sequence_data.equals(sample_sequences)
#     assert embedder.device == "cuda" if torch.cuda.is_available() else "cpu"


# def test_onehot_embedding_output(sample_sequences):
#     """Test OneHotEmbedding output format and values."""
#     embedder = OneHotEmbedding(sample_sequences)
#     embeddings = embedder.get_embeddings()

#     # Check output is a pandas Series
#     assert isinstance(embeddings, pd.Series)
#     assert len(embeddings) == len(sample_sequences)

#     # Check each embedding tensor shape and values
#     for emb in embeddings:
#         assert isinstance(emb, torch.Tensor)
#         assert emb.shape == (4, 4)  # 4 nucleotides x sequence length
#         assert torch.all((emb == 0) | (emb == 1))  # Only contains 0s and 1s
#         assert torch.sum(emb, dim=0).eq(1).all()  # One-hot property: sum along each position = 1


# def test_onehot_embedding_invalid_sequence():
#     """Test OneHotEmbedding with invalid DNA sequence."""
#     invalid_sequences = pd.Series(["ATCG", "XYZW"])  # Contains invalid characters
#     embedder = OneHotEmbedding(invalid_sequences)

#     with pytest.raises(ValueError):
#         _ = embedder.get_embeddings()
