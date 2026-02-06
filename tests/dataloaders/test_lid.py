"""
Tests for ups_challenge.dataloaders.lid.

Each test is atomic and follows Arrange, Act, Assert.
Tests assert on the public interface of each function, not implementation details.
"""
import pickle

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from ups_challenge.dataloaders.lid import (
    decode_with_lang,
    extract_embedding_from_sample,
    build_lid_dataset,
    collate_embeddings,
)


# -----------------------------------------------------------------------------
# decode_with_lang
# -----------------------------------------------------------------------------


def test_decode_with_lang_returns_none_when_url_has_no_audio_path():
    """decode_with_lang returns None when url does not contain 'audio/' or 'audio2/'."""
    # Arrange
    sample = (b"mp3_bytes", "key", "https://example.com/something.tar")
    lid_index = {("0001", "key.mp3"): "en"}

    # Act
    result = decode_with_lang(sample, lid_index)

    # Assert
    assert result is None


def test_decode_with_lang_returns_none_when_key_not_in_index():
    """decode_with_lang returns None when (tar_number, filename) not in lid_index."""
    # Arrange: url gives tar_number 0001, key "0001/foo" -> filename "foo.mp3"; not in index
    sample = (b"mp3_bytes", "0001/foo", "https://huggingface.co/.../audio/0001.tar?download=1")
    lid_index = {("0001", "other.mp3"): "en"}

    # Act (returns None before decoder is used)
    result = decode_with_lang(sample, lid_index)

    # Assert
    assert result is None


def test_decode_with_lang_returns_none_when_decoder_raises():
    """decode_with_lang returns None when AudioDecoder or extraction fails."""
    # Arrange: valid url and key in index
    sample = (b"bad_mp3", "0001/clip", "https://example.com/datasets/audio/0001.tar")
    lid_index = {("0001", "clip.mp3"): "en"}

    # Act
    with patch("ups_challenge.dataloaders.lid.AudioDecoder", side_effect=RuntimeError("decode failed")):
        result = decode_with_lang(sample, lid_index)

    # Assert
    assert result is None


def test_decode_with_lang_returns_none_when_duration_less_than_chunk_sec():
    """decode_with_lang returns None when file duration < chunk_sec."""
    # Arrange
    chunk_sec = 10.0
    target_sr = 16000
    chunk_samples = int(chunk_sec * target_sr)
    fake_data = torch.zeros(1, chunk_samples)

    fake_metadata = MagicMock()
    fake_metadata.duration_seconds_from_header = 5.0

    fake_decoder = MagicMock()
    fake_decoder.metadata = fake_metadata
    fake_decoder.get_samples_played_in_range.return_value = MagicMock(data=fake_data)

    sample = (b"mp3", "0001/clip", "https://example.com/audio/0001.tar")
    lid_index = {("0001", "clip.mp3"): "es"}

    # Act
    with patch("ups_challenge.dataloaders.lid.AudioDecoder", return_value=fake_decoder):
        with patch("ups_challenge.dataloaders.lid.np.random.uniform", return_value=0.0):
            result = decode_with_lang(sample, lid_index, target_sr=target_sr, chunk_sec=chunk_sec)

    # Assert
    assert result is None


def test_decode_with_lang_returns_dict_with_waveform_lang_key():
    """decode_with_lang returns dict with 'waveform', 'lang', and 'key' when successful."""
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

    sample = (b"mp3", "0001/mysample", "https://example.com/audio/0001.tar")
    lid_index = {("0001", "mysample.mp3"): "fr"}

    # Act
    with patch("ups_challenge.dataloaders.lid.AudioDecoder", return_value=fake_decoder):
        with patch("ups_challenge.dataloaders.lid.np.random.uniform", return_value=0.0):
            result = decode_with_lang(sample, lid_index, target_sr=target_sr, chunk_sec=chunk_sec)

    # Assert
    assert result is not None
    assert "waveform" in result
    assert "lang" in result
    assert "key" in result
    assert result["lang"] == "fr"
    assert result["key"] == "0001/mysample"


def test_decode_with_lang_waveform_is_numpy_with_expected_length():
    """waveform is a numpy array of length chunk_sec * target_sr."""
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

    sample = (b"mp3", "42/clip", "https://example.com/audio2/42.tar")
    lid_index = {("42", "clip.mp3"): "de"}

    # Act
    with patch("ups_challenge.dataloaders.lid.AudioDecoder", return_value=fake_decoder):
        with patch("ups_challenge.dataloaders.lid.np.random.uniform", return_value=0.0):
            result = decode_with_lang(sample, lid_index, target_sr=target_sr, chunk_sec=chunk_sec)

    # Assert
    assert isinstance(result["waveform"], np.ndarray)
    assert result["waveform"].shape == (chunk_samples,)


@pytest.mark.parametrize("url_path,tar_number", [
    ("https://example.com/audio/0001.tar", "0001"),
    ("https://huggingface.co/datasets/.../audio2/5001.tar?download=1", "5001"),
])
def test_decode_with_lang_parses_tar_number_from_audio_url(url_path, tar_number):
    """tar_number is parsed from url containing 'audio/' or 'audio2/'."""
    # Arrange: key without slash -> filename = key + .mp3
    sample = (b"mp3", "clip", url_path)
    lid_index = {(tar_number, "clip.mp3"): "en"}

    chunk_samples = int(10.0 * 16000)
    fake_data = torch.zeros(1, chunk_samples)
    fake_metadata = MagicMock()
    fake_metadata.duration_seconds_from_header = 20.0
    fake_decoder = MagicMock()
    fake_decoder.metadata = fake_metadata
    fake_decoder.get_samples_played_in_range.return_value = MagicMock(data=fake_data)

    # Act
    with patch("ups_challenge.dataloaders.lid.AudioDecoder", return_value=fake_decoder):
        with patch("ups_challenge.dataloaders.lid.np.random.uniform", return_value=0.0):
            result = decode_with_lang(sample, lid_index, target_sr=16000, chunk_sec=10.0)

    # Assert (success implies correct tar_number and filename)
    assert result is not None
    assert result["lang"] == "en"


# -----------------------------------------------------------------------------
# extract_embedding_from_sample
# -----------------------------------------------------------------------------


def test_extract_embedding_from_sample_returns_none_when_sample_none():
    """extract_embedding_from_sample returns None when sample is None."""
    # Arrange
    model = MagicMock()
    feature_extractor = MagicMock()
    device = torch.device("cpu")
    lang_to_idx = {"en": 0}

    # Act
    result = extract_embedding_from_sample(
        None, model, feature_extractor, device, lang_to_idx
    )

    # Assert
    assert result is None


def test_extract_embedding_from_sample_returns_none_when_lang_not_in_lang_to_idx():
    """extract_embedding_from_sample returns None when lang is not in lang_to_idx."""
    # Arrange
    sample = {"waveform": np.zeros(16000), "lang": "unknown", "key": "k"}
    model = MagicMock()
    feature_extractor = MagicMock()
    device = torch.device("cpu")
    lang_to_idx = {"en": 0, "es": 1}

    # Act
    result = extract_embedding_from_sample(
        sample, model, feature_extractor, device, lang_to_idx
    )

    # Assert
    assert result is None


def test_extract_embedding_from_sample_returns_embedding_and_label():
    """extract_embedding_from_sample returns dict with 'embedding' and 'label' tensors."""
    # Arrange
    sample = {"waveform": np.zeros(16000), "lang": "en", "key": "k"}
    feature_extractor = MagicMock()
    feature_extractor.sampling_rate = 16000
    feature_extractor.return_value = MagicMock(to=MagicMock(return_value={}))

    # model(**inputs) -> last_hidden_state (1, seq, dim) so mean(dim=1) -> (1, dim)
    hidden_dim = 768
    fake_hidden = torch.randn(1, 10, hidden_dim)
    model = MagicMock()
    model.return_value = MagicMock(last_hidden_state=fake_hidden)

    device = torch.device("cpu")
    lang_to_idx = {"en": 0}

    # Mock feature_extractor call to return inputs dict; model receives it
    with patch.object(
        type(feature_extractor),
        "__call__",
        return_value=MagicMock(to=MagicMock(return_value={"input_values": torch.zeros(1, 100)})),
    ):
        with patch("ups_challenge.dataloaders.lid.torch.no_grad"):
            # Act
            result = extract_embedding_from_sample(
                sample, model, feature_extractor, device, lang_to_idx
            )

    # Assert
    assert result is not None
    assert "embedding" in result
    assert "label" in result
    assert isinstance(result["embedding"], torch.Tensor)
    assert isinstance(result["label"], torch.Tensor)
    assert result["label"].dtype == torch.long
    assert result["label"].item() == 0


def test_extract_embedding_from_sample_label_matches_lang_to_idx():
    """label tensor value equals lang_to_idx[lang]."""
    # Arrange
    sample = {"waveform": np.zeros(16000), "lang": "es", "key": "k"}
    fake_hidden = torch.randn(1, 5, 64)
    model = MagicMock()
    model.return_value = MagicMock(last_hidden_state=fake_hidden)

    feature_extractor = MagicMock()
    feature_extractor.sampling_rate = 16000
    device = torch.device("cpu")
    lang_to_idx = {"en": 0, "es": 1, "fr": 2}

    with patch.object(
        type(feature_extractor),
        "__call__",
        return_value=MagicMock(to=MagicMock(return_value={"input_values": torch.zeros(1, 50)})),
    ):
        with patch("ups_challenge.dataloaders.lid.torch.no_grad"):
            result = extract_embedding_from_sample(
                sample, model, feature_extractor, device, lang_to_idx
            )

    assert result["label"].item() == 1


# -----------------------------------------------------------------------------
# collate_embeddings
# -----------------------------------------------------------------------------


def test_collate_embeddings_returns_none_when_batch_empty():
    """collate_embeddings([]) returns None."""
    # Arrange
    batch = []

    # Act
    result = collate_embeddings(batch)

    # Assert
    assert result is None


def test_collate_embeddings_returns_none_when_all_none():
    """collate_embeddings([None, None]) returns None."""
    # Arrange
    batch = [None, None]

    # Act
    result = collate_embeddings(batch)

    # Assert
    assert result is None


def test_collate_embeddings_returns_tuple_embeddings_labels():
    """collate_embeddings returns (embeddings, labels) tuple of stacked tensors."""
    # Arrange
    batch = [
        {"embedding": torch.randn(64), "label": torch.tensor(0, dtype=torch.long)},
        {"embedding": torch.randn(64), "label": torch.tensor(1, dtype=torch.long)},
    ]

    # Act
    result = collate_embeddings(batch)

    # Assert
    assert result is not None
    embeddings, labels = result
    assert isinstance(embeddings, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert embeddings.shape == (2, 64)
    assert labels.shape == (2,)


def test_collate_embeddings_filters_out_none():
    """None entries are removed; result length matches non-None count."""
    # Arrange
    sample = {"embedding": torch.randn(32), "label": torch.tensor(0, dtype=torch.long)}
    batch = [None, sample, None]

    # Act
    result = collate_embeddings(batch)

    # Assert
    embeddings, labels = result
    assert embeddings.shape[0] == 1
    assert labels.shape[0] == 1


# -----------------------------------------------------------------------------
# build_lid_dataset
# -----------------------------------------------------------------------------


def test_build_lid_dataset_returns_iterable(tmp_path):
    """build_lid_dataset returns an object that can be iterated."""
    # Arrange: valid index pickle and mock build_urls
    index_path = tmp_path / "lid_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({("0001", "a.mp3"): "en"}, f, protocol=4)

    with patch("ups_challenge.dataloaders.lid.build_urls") as mock_build_urls:
        mock_build_urls.return_value = ["pipe:curl -s -L https://example.com/1.tar"]

        # Act
        dataset = build_lid_dataset(
            index_path=str(index_path),
            model=MagicMock(),
            feature_extractor=MagicMock(),
            device=torch.device("cpu"),
            lang_to_idx={"en": 0},
            hf_token="dummy",
        )

    # Assert
    assert dataset is not None
    assert hasattr(dataset, "__iter__")


@pytest.mark.parametrize("max_samples", [1, 10, 100])
def test_build_lid_dataset_with_max_samples_returns_limited_dataset(tmp_path, max_samples):
    """When max_samples is set, returned dataset is a LimitedDataset."""
    from ups_challenge.utils import LimitedDataset

    # Arrange
    index_path = tmp_path / "lid_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({("0001", "a.mp3"): "en"}, f, protocol=4)

    with patch("ups_challenge.dataloaders.lid.build_urls") as mock_build_urls:
        mock_build_urls.return_value = ["pipe:curl -s -L https://example.com/1.tar"]

        # Act
        dataset = build_lid_dataset(
            index_path=str(index_path),
            model=MagicMock(),
            feature_extractor=MagicMock(),
            device=torch.device("cpu"),
            lang_to_idx={"en": 0},
            hf_token="dummy",
            max_samples=max_samples,
        )

    # Assert
    assert isinstance(dataset, LimitedDataset)
    assert dataset.max_samples == max_samples


def test_build_lid_dataset_without_max_samples_not_limited_dataset(tmp_path):
    """When max_samples is None, returned object is not LimitedDataset."""
    from ups_challenge.utils import LimitedDataset

    # Arrange
    index_path = tmp_path / "lid_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({("0001", "a.mp3"): "en"}, f, protocol=4)

    with patch("ups_challenge.dataloaders.lid.build_urls") as mock_build_urls:
        mock_build_urls.return_value = ["pipe:curl -s -L https://example.com/1.tar"]

        # Act
        dataset = build_lid_dataset(
            index_path=str(index_path),
            model=MagicMock(),
            feature_extractor=MagicMock(),
            device=torch.device("cpu"),
            lang_to_idx={"en": 0},
            hf_token="dummy",
            max_samples=None,
        )

    # Assert
    assert not isinstance(dataset, LimitedDataset)
