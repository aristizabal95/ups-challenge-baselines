"""
Tests for ups_challenge.dataloaders.base.

Each test is atomic and follows Arrange, Act, Assert.
Tests assert on the public interface of each function, not implementation details.
"""
import torch
import pytest
from unittest.mock import MagicMock, patch

from ups_challenge.dataloaders.base import collate_fn, decode_and_normalize, build_wds_dataset


# -----------------------------------------------------------------------------
# collate_fn
# -----------------------------------------------------------------------------


def test_collate_fn_returns_none_when_batch_is_empty():
    """collate_fn([]) returns None."""
    # Arrange
    batch = []

    # Act
    result = collate_fn(batch)

    # Assert
    assert result is None


def test_collate_fn_returns_none_when_all_items_are_none():
    """collate_fn([None, None]) returns None."""
    # Arrange
    batch = [None, None]

    # Act
    result = collate_fn(batch)

    # Assert
    assert result is None


def test_collate_fn_returns_dict_with_expected_keys_for_single_sample():
    """collate_fn([sample]) returns dict with 'input_values' and 'attention_mask'."""
    # Arrange
    sample = {
        "input_values": torch.randn(2, 16000),
        "attention_mask": torch.ones(2, 16000, dtype=torch.long),
    }
    batch = [sample]

    # Act
    result = collate_fn(batch)

    # Assert
    assert result is not None
    assert "input_values" in result
    assert "attention_mask" in result
    assert result["input_values"].shape == (2, 16000)
    assert result["attention_mask"].shape == (2, 16000)
    assert result["attention_mask"].dtype == torch.long


@pytest.mark.parametrize("shapes,expected_first_dim", [
    ([(1, 100), (2, 100)], 3),
    ([(1, 50), (1, 50), (1, 50)], 3),
    ([(2, 16000), (3, 16000)], 5),
    ([(1, 200)], 1),
])
def test_collate_fn_concatenates_input_values_on_dim_zero(shapes, expected_first_dim):
    """collate_fn concatenates input_values along dim=0."""
    # Arrange
    batch = [
        {
            "input_values": torch.randn(n, t),
            "attention_mask": torch.ones(n, t, dtype=torch.long),
        }
        for n, t in shapes
    ]

    # Act
    result = collate_fn(batch)

    # Assert
    expected_t = shapes[0][1]
    assert result["input_values"].shape == (expected_first_dim, expected_t)
    assert result["attention_mask"].shape == (expected_first_dim, expected_t)


def test_collate_fn_filters_out_none_and_concatenates_rest():
    """collate_fn ignores None entries and collates the rest."""
    # Arrange
    sample = {
        "input_values": torch.randn(1, 50),
        "attention_mask": torch.ones(1, 50, dtype=torch.long),
    }
    batch = [None, sample, None]

    # Act
    result = collate_fn(batch)

    # Assert
    assert result is not None
    assert result["input_values"].shape == (1, 50)
    assert result["attention_mask"].shape == (1, 50)


# -----------------------------------------------------------------------------
# decode_and_normalize
# -----------------------------------------------------------------------------


def test_decode_and_normalize_returns_dict_with_expected_keys():
    """decode_and_normalize(sample) returns dict with 'input_values' and 'attention_mask'."""
    # Arrange: mock decoder for a short file (single chunk)
    chunk_samples = int(10.0 * 16000)
    fake_data = torch.zeros(1, chunk_samples)
    fake_samples = MagicMock()
    fake_samples.data = fake_data

    fake_metadata = MagicMock()
    fake_metadata.duration_seconds_from_header = 5.0

    fake_decoder = MagicMock()
    fake_decoder.metadata = fake_metadata
    fake_decoder.get_samples_played_in_range.return_value = fake_samples

    sample = (b"fake_mp3_bytes", "key", "https://example.com/shard.tar")

    # Act
    with patch("ups_challenge.dataloaders.base.AudioDecoder", return_value=fake_decoder):
        result = decode_and_normalize(sample, chunk_sec=10.0, target_sr=16000)

    # Assert
    assert "input_values" in result
    assert "attention_mask" in result
    assert isinstance(result["input_values"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)


def test_decode_and_normalize_attention_mask_matches_input_values_shape():
    """attention_mask has the same shape as input_values."""
    # Arrange
    chunk_samples = int(10.0 * 16000)
    fake_data = torch.zeros(1, chunk_samples)
    fake_samples = MagicMock()
    fake_samples.data = fake_data

    fake_metadata = MagicMock()
    fake_metadata.duration_seconds_from_header = 5.0

    fake_decoder = MagicMock()
    fake_decoder.metadata = fake_metadata
    fake_decoder.get_samples_played_in_range.return_value = fake_samples

    sample = (b"fake_mp3_bytes", "key", "https://example.com/shard.tar")

    # Act
    with patch("ups_challenge.dataloaders.base.AudioDecoder", return_value=fake_decoder):
        result = decode_and_normalize(sample, chunk_sec=10.0, target_sr=16000)

    # Assert
    assert result["attention_mask"].shape == result["input_values"].shape
    assert result["attention_mask"].dtype == torch.long


@pytest.mark.parametrize("duration_sec,chunk_sec,target_sr", [
    (5.0, 10.0, 16000),
    (1.0, 10.0, 16000),
    (10.0, 10.0, 16000),
    (3.0, 5.0, 8000),
])
def test_decode_and_normalize_short_file_returns_one_chunk(duration_sec, chunk_sec, target_sr):
    """For duration <= chunk_sec, output has one chunk (first dim = 1)."""
    # Arrange
    chunk_samples = int(chunk_sec * target_sr)
    fake_data = torch.zeros(1, chunk_samples)
    fake_samples = MagicMock()
    fake_samples.data = fake_data

    fake_metadata = MagicMock()
    fake_metadata.duration_seconds_from_header = duration_sec

    fake_decoder = MagicMock()
    fake_decoder.metadata = fake_metadata
    fake_decoder.get_samples_played_in_range.return_value = fake_samples

    sample = (b"fake_mp3_bytes", "key", "https://example.com/shard.tar")

    # Act
    with patch("ups_challenge.dataloaders.base.AudioDecoder", return_value=fake_decoder):
        result = decode_and_normalize(sample, chunk_sec=chunk_sec, target_sr=target_sr)

    # Assert
    assert result["input_values"].ndim == 2
    assert result["input_values"].shape[0] == 1
    assert result["input_values"].shape[1] == chunk_samples


@pytest.mark.parametrize("max_chunks_per_example", [1, 4, 8, 16])
def test_decode_and_normalize_long_file_returns_multiple_chunks(max_chunks_per_example):
    """For duration > chunk_sec, output has multiple chunks up to max_chunks_per_example."""
    # Arrange: duration 100s, chunk_sec 10s
    chunk_sec = 10.0
    target_sr = 16000
    chunk_samples = int(chunk_sec * target_sr)
    fake_data = torch.zeros(1, chunk_samples)
    fake_samples = MagicMock()
    fake_samples.data = fake_data

    fake_metadata = MagicMock()
    fake_metadata.duration_seconds_from_header = 100.0

    fake_decoder = MagicMock()
    fake_decoder.metadata = fake_metadata
    fake_decoder.get_samples_played_in_range.return_value = fake_samples

    sample = (b"fake_mp3_bytes", "key", "https://example.com/shard.tar")

    # Act
    with patch("ups_challenge.dataloaders.base.AudioDecoder", return_value=fake_decoder):
        result = decode_and_normalize(
            sample,
            chunk_sec=chunk_sec,
            target_sr=target_sr,
            max_chunks_per_example=max_chunks_per_example,
        )

    # Assert
    assert result["input_values"].ndim == 2
    assert result["input_values"].shape[0] == max_chunks_per_example
    assert result["input_values"].shape[1] == chunk_samples


# -----------------------------------------------------------------------------
# build_wds_dataset
# -----------------------------------------------------------------------------


def test_build_wds_dataset_returns_iterable():
    """build_wds_dataset returns an object that can be iterated (dataset)."""
    # Arrange: avoid real index and HF token; WebDataset requires at least one URL
    with patch("ups_challenge.dataloaders.base.build_urls") as mock_build_urls:
        mock_build_urls.return_value = ["pipe:curl -s -L https://example.com/1.tar"]
        # Act
        dataset = build_wds_dataset(
            langs=[],
            index_path="/nonexistent/index.pkl",
            hf_token="dummy_token",
        )

    # Assert
    assert dataset is not None
    assert hasattr(dataset, "__iter__")


@pytest.mark.parametrize("max_samples", [1, 10, 100])
def test_build_wds_dataset_with_max_samples_returns_limited_dataset(max_samples):
    """When max_samples is set, returned dataset is a LimitedDataset with that limit."""
    from ups_challenge.utils import LimitedDataset

    # Arrange: WebDataset requires at least one URL
    with patch("ups_challenge.dataloaders.base.build_urls") as mock_build_urls:
        mock_build_urls.return_value = ["pipe:curl -s -L https://example.com/1.tar"]

        # Act
        dataset = build_wds_dataset(
            langs=[],
            index_path="/nonexistent/index.pkl",
            hf_token="dummy_token",
            max_samples=max_samples,
        )

    # Assert
    assert isinstance(dataset, LimitedDataset)
    assert dataset.max_samples == max_samples


def test_build_wds_dataset_without_max_samples_returns_webdataset_pipeline():
    """When max_samples is None, returned object is not LimitedDataset."""
    from ups_challenge.utils import LimitedDataset

    # Arrange
    with patch("ups_challenge.dataloaders.base.build_urls") as mock_build_urls:
        mock_build_urls.return_value = ["pipe:curl -s -L https://example.com/1.tar"]

        # Act
        dataset = build_wds_dataset(
            langs=[],
            index_path="/nonexistent/index.pkl",
            hf_token="dummy_token",
            max_samples=None,
        )

    # Assert: pipeline (wds.WebDataset .to_tuple().map()) is not LimitedDataset
    assert not isinstance(dataset, LimitedDataset)
