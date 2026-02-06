"""
Tests for ups_challenge.dataloaders.contrastive.

Each test is atomic and follows Arrange, Act, Assert.
Tests assert on the public interface of each function, not implementation details.
"""
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from ups_challenge.dataloaders.contrastive import (
    decode_contrastive,
    collate_contrastive,
    build_contrastive_dataset,
)


# -----------------------------------------------------------------------------
# decode_contrastive
# -----------------------------------------------------------------------------


def test_decode_contrastive_returns_none_when_decoder_raises():
    """decode_contrastive returns None when AudioDecoder fails."""
    # Arrange
    sample = (b"invalid_mp3", "key", "https://example.com/shard.tar")

    # Act
    with patch("ups_challenge.dataloaders.contrastive.AudioDecoder", side_effect=RuntimeError("decode failed")):
        result = decode_contrastive(sample)

    # Assert
    assert result is None


def test_decode_contrastive_returns_none_when_duration_less_than_chunk_sec():
    """decode_contrastive returns None when file is shorter than chunk_sec."""
    # Arrange: duration 5s, chunk_sec 10s
    chunk_sec = 10.0
    target_sr = 16000
    chunk_samples = int(chunk_sec * target_sr)
    fake_data = torch.zeros(1, chunk_samples)

    fake_metadata = MagicMock()
    fake_metadata.duration_seconds_from_header = 5.0

    fake_decoder = MagicMock()
    fake_decoder.metadata = fake_metadata
    fake_decoder.get_samples_played_in_range.return_value = MagicMock(data=fake_data)

    sample = (b"fake_mp3", "key", "https://example.com/shard.tar")

    # Act
    with patch("ups_challenge.dataloaders.contrastive.AudioDecoder", return_value=fake_decoder):
        result = decode_contrastive(sample, target_sr=target_sr, chunk_sec=chunk_sec)

    # Assert
    assert result is None


def test_decode_contrastive_returns_dict_with_expected_keys():
    """decode_contrastive returns dict with 'anchor', 'positive', and 'meta'."""
    # Arrange: duration long enough for two chunks
    chunk_sec = 10.0
    target_sr = 16000
    chunk_samples = int(chunk_sec * target_sr)
    fake_data = torch.zeros(1, chunk_samples)

    fake_metadata = MagicMock()
    fake_metadata.duration_seconds_from_header = 30.0

    fake_decoder = MagicMock()
    fake_decoder.metadata = fake_metadata
    fake_decoder.get_samples_played_in_range.return_value = MagicMock(data=fake_data)

    sample = (b"fake_mp3", "mykey", "https://example.com/1.tar")

    # Act
    with patch("ups_challenge.dataloaders.contrastive.AudioDecoder", return_value=fake_decoder):
        result = decode_contrastive(sample, target_sr=target_sr, chunk_sec=chunk_sec)

    # Assert
    assert result is not None
    assert "anchor" in result
    assert "positive" in result
    assert "meta" in result


def test_decode_contrastive_anchor_and_positive_are_numpy_arrays():
    """anchor and positive are numpy arrays of the same length."""
    # Arrange
    chunk_sec = 10.0
    target_sr = 16000
    chunk_samples = int(chunk_sec * target_sr)
    fake_data = torch.zeros(1, chunk_samples)

    fake_metadata = MagicMock()
    fake_metadata.duration_seconds_from_header = 30.0

    fake_decoder = MagicMock()
    fake_decoder.metadata = fake_metadata
    fake_decoder.get_samples_played_in_range.return_value = MagicMock(data=fake_data)

    sample = (b"fake_mp3", "key", "https://example.com/shard.tar")

    # Act
    with patch("ups_challenge.dataloaders.contrastive.AudioDecoder", return_value=fake_decoder):
        result = decode_contrastive(sample, target_sr=target_sr, chunk_sec=chunk_sec)

    # Assert
    assert isinstance(result["anchor"], np.ndarray)
    assert isinstance(result["positive"], np.ndarray)
    assert result["anchor"].shape == result["positive"].shape
    assert result["anchor"].shape == (chunk_samples,)


def test_decode_contrastive_meta_contains_key_url_duration():
    """meta dict contains key, url, and duration."""
    # Arrange
    chunk_sec = 10.0
    target_sr = 16000
    chunk_samples = int(chunk_sec * target_sr)
    fake_data = torch.zeros(1, chunk_samples)

    fake_metadata = MagicMock()
    fake_metadata.duration_seconds_from_header = 25.0

    fake_decoder = MagicMock()
    fake_decoder.metadata = fake_metadata
    fake_decoder.get_samples_played_in_range.return_value = MagicMock(data=fake_data)

    sample = (b"fake_mp3", "sample_key", "https://example.com/42.tar")

    # Act
    with patch("ups_challenge.dataloaders.contrastive.AudioDecoder", return_value=fake_decoder):
        result = decode_contrastive(sample, target_sr=target_sr, chunk_sec=chunk_sec)

    # Assert
    meta = result["meta"]
    assert meta["key"] == "sample_key"
    assert meta["url"] == "https://example.com/42.tar"
    assert meta["duration"] == 25.0


@pytest.mark.parametrize("chunk_sec,target_sr", [
    (10.0, 16000),
    (5.0, 16000),
    (10.0, 8000),
])
def test_decode_contrastive_chunk_length_matches_chunk_sec_and_sr(chunk_sec, target_sr):
    """anchor and positive length equals chunk_sec * target_sr."""
    # Arrange
    chunk_samples = int(chunk_sec * target_sr)
    fake_data = torch.zeros(1, chunk_samples)

    fake_metadata = MagicMock()
    fake_metadata.duration_seconds_from_header = 60.0

    fake_decoder = MagicMock()
    fake_decoder.metadata = fake_metadata
    fake_decoder.get_samples_played_in_range.return_value = MagicMock(data=fake_data)

    sample = (b"fake_mp3", "key", "https://example.com/shard.tar")

    # Act
    with patch("ups_challenge.dataloaders.contrastive.AudioDecoder", return_value=fake_decoder):
        result = decode_contrastive(sample, target_sr=target_sr, chunk_sec=chunk_sec)

    # Assert
    assert result["anchor"].shape == (chunk_samples,)
    assert result["positive"].shape == (chunk_samples,)


def test_decode_contrastive_returns_none_when_extract_fails():
    """decode_contrastive returns None if chunk extraction raises."""
    # Arrange: first call succeeds, second raises (e.g. corrupted segment)
    chunk_sec = 10.0
    target_sr = 16000
    chunk_samples = int(chunk_sec * target_sr)
    fake_data = torch.zeros(1, chunk_samples)

    fake_metadata = MagicMock()
    fake_metadata.duration_seconds_from_header = 30.0

    fake_decoder = MagicMock()
    fake_decoder.metadata = fake_metadata
    fake_decoder.get_samples_played_in_range.side_effect = [
        MagicMock(data=fake_data),
        RuntimeError("corrupted"),
    ]

    sample = (b"fake_mp3", "key", "https://example.com/shard.tar")

    # Act
    with patch("ups_challenge.dataloaders.contrastive.AudioDecoder", return_value=fake_decoder):
        result = decode_contrastive(sample, target_sr=target_sr, chunk_sec=chunk_sec)

    # Assert
    assert result is None


# -----------------------------------------------------------------------------
# collate_contrastive
# -----------------------------------------------------------------------------


def test_collate_contrastive_returns_none_when_batch_empty():
    """collate_contrastive([]) returns None."""
    # Arrange
    batch = []

    # Act
    result = collate_contrastive(batch)

    # Assert
    assert result is None


def test_collate_contrastive_returns_none_when_all_none():
    """collate_contrastive([None, None]) returns None."""
    # Arrange
    batch = [None, None]

    # Act
    result = collate_contrastive(batch)

    # Assert
    assert result is None


def test_collate_contrastive_returns_dict_with_anchor_positive_negative():
    """collate_contrastive returns dict with anchor, positive, negative as lists."""
    # Arrange
    batch = [
        {"anchor": np.zeros(100), "positive": np.ones(100)},
        {"anchor": np.ones(100) * 2, "positive": np.ones(100) * 3},
    ]

    # Act
    with patch("ups_challenge.dataloaders.contrastive.random.shuffle"):
        result = collate_contrastive(batch)

    # Assert
    assert "anchor" in result
    assert "positive" in result
    assert "negative" in result
    assert isinstance(result["anchor"], list)
    assert isinstance(result["positive"], list)
    assert isinstance(result["negative"], list)


def test_collate_contrastive_list_lengths_match_batch_size():
    """anchor, positive, and negative lists have length equal to non-None batch size."""
    # Arrange
    batch = [
        {"anchor": np.zeros(50), "positive": np.ones(50)},
        {"anchor": np.ones(50) * 2, "positive": np.ones(50) * 3},
        {"anchor": np.ones(50) * 4, "positive": np.ones(50) * 5},
    ]

    # Act
    with patch("ups_challenge.dataloaders.contrastive.random.shuffle"):
        result = collate_contrastive(batch)

    # Assert
    assert len(result["anchor"]) == 3
    assert len(result["positive"]) == 3
    assert len(result["negative"]) == 3


def test_collate_contrastive_negatives_are_permutation_of_anchors():
    """negative list contains the same anchor arrays as anchor list (possibly reordered)."""
    # Arrange: use distinct arrays so we can check identity/set
    a0 = np.array([1.0])
    a1 = np.array([2.0])
    a2 = np.array([3.0])
    batch = [
        {"anchor": a0, "positive": np.ones(1)},
        {"anchor": a1, "positive": np.ones(1) * 2},
        {"anchor": a2, "positive": np.ones(1) * 3},
    ]

    # Act (deterministic shuffle so we can assert on permutation)
    with patch("ups_challenge.dataloaders.contrastive.random.shuffle", side_effect=lambda x: x.reverse()):
        result = collate_contrastive(batch)

    # Assert: negatives are the same set of arrays as anchors (permutation)
    assert set(id(arr) for arr in result["negative"]) == set(id(arr) for arr in result["anchor"])


def test_collate_contrastive_no_negative_same_as_anchor_or_positive():
    """For each position i, negative[i] is not the same array as anchor[i] or positive[i]."""
    # Arrange: distinct arrays so we can check identity
    a0, a1, a2 = np.array([1.0]), np.array([2.0]), np.array([3.0])
    p0, p1, p2 = np.array([10.0]), np.array([20.0]), np.array([30.0])
    batch = [
        {"anchor": a0, "positive": p0},
        {"anchor": a1, "positive": p1},
        {"anchor": a2, "positive": p2},
    ]
    # Derangement: shift by 1 so perm[i] != i for all i -> negatives[i] != anchors[i]
    def shuffle_derangement(x):
        n = len(x)
        if n > 1:
            x[:] = [x[(i + 1) % n] for i in range(n)]

    # Act
    with patch("ups_challenge.dataloaders.contrastive.random.shuffle", side_effect=shuffle_derangement):
        result = collate_contrastive(batch)

    # Assert: no negative is the anchor or positive for that same index
    for i in range(len(result["anchor"])):
        assert result["negative"][i] is not result["anchor"][i]
        assert result["negative"][i] is not result["positive"][i]


def test_collate_contrastive_filters_out_none():
    """None entries in batch are removed; result length is count of non-None items."""
    # Arrange
    sample = {"anchor": np.zeros(10), "positive": np.ones(10)}
    batch = [None, sample, None]

    # Act
    with patch("ups_challenge.dataloaders.contrastive.random.shuffle"):
        result = collate_contrastive(batch)

    # Assert
    assert len(result["anchor"]) == 1
    assert len(result["positive"]) == 1
    assert len(result["negative"]) == 1


# -----------------------------------------------------------------------------
# build_contrastive_dataset
# -----------------------------------------------------------------------------


def test_build_contrastive_dataset_returns_iterable():
    """build_contrastive_dataset returns an object that can be iterated."""
    # Arrange: WebDataset requires at least one URL
    with patch("ups_challenge.dataloaders.contrastive.build_urls") as mock_build_urls:
        mock_build_urls.return_value = ["pipe:curl -s -L https://example.com/1.tar"]

        # Act
        dataset = build_contrastive_dataset(
            langs=[],
            index_path="/nonexistent/index.pkl",
            hf_token="dummy_token",
        )

    # Assert
    assert dataset is not None
    assert hasattr(dataset, "__iter__")


@pytest.mark.parametrize("max_samples", [1, 10, 100])
def test_build_contrastive_dataset_with_max_samples_returns_limited_dataset(max_samples):
    """When max_samples is set, returned dataset is a LimitedDataset with that limit."""
    from ups_challenge.utils import LimitedDataset

    # Arrange
    with patch("ups_challenge.dataloaders.contrastive.build_urls") as mock_build_urls:
        mock_build_urls.return_value = ["pipe:curl -s -L https://example.com/1.tar"]

        # Act
        dataset = build_contrastive_dataset(
            langs=[],
            index_path="/nonexistent/index.pkl",
            hf_token="dummy_token",
            max_samples=max_samples,
        )

    # Assert
    assert isinstance(dataset, LimitedDataset)
    assert dataset.max_samples == max_samples


def test_build_contrastive_dataset_without_max_samples_not_limited_dataset():
    """When max_samples is None, returned object is not LimitedDataset."""
    from ups_challenge.utils import LimitedDataset

    # Arrange
    with patch("ups_challenge.dataloaders.contrastive.build_urls") as mock_build_urls:
        mock_build_urls.return_value = ["pipe:curl -s -L https://example.com/1.tar"]

        # Act
        dataset = build_contrastive_dataset(
            langs=[],
            index_path="/nonexistent/index.pkl",
            hf_token="dummy_token",
            max_samples=None,
        )

    # Assert
    assert not isinstance(dataset, LimitedDataset)


def test_build_contrastive_dataset_default_langs_is_empty_list():
    """build_contrastive_dataset(langs=None) behaves like langs=[] (build_urls gets empty list)."""
    with patch("ups_challenge.dataloaders.contrastive.build_urls") as mock_build_urls:
        mock_build_urls.return_value = ["pipe:curl -s -L https://example.com/1.tar"]

        build_contrastive_dataset(
            langs=None,
            index_path="/nonexistent/index.pkl",
            hf_token="dummy_token",
        )

    mock_build_urls.assert_called_once()
    call_kwargs = mock_build_urls.call_args[1]
    assert call_kwargs.get("index_path") == "/nonexistent/index.pkl"
    # langs was passed as first kwarg or in *args; check call args
    assert mock_build_urls.call_args[0][0] == []
