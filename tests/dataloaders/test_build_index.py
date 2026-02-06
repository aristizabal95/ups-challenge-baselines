"""
Tests for ups_challenge.dataloaders.build_index.

Each test is atomic and follows Arrange, Act, Assert.
Tests assert on the public interface of each function, not implementation details.
"""
import pickle
import pytest
from pathlib import Path
from unittest.mock import patch

from ups_challenge.dataloaders.build_index import build_lid_index_splits, build_lid_index


def _make_lid_index(n_per_lang=None):
    """Small index: (tar_number, filename) -> lang. Default: 10 en, 10 es for stratification."""
    if n_per_lang is None:
        n_per_lang = {"en": 10, "es": 10}
    index = {}
    for lang, n in n_per_lang.items():
        for i in range(n):
            index[(f"0001", f"clip_{lang}_{i}.mp3")] = lang
    return index


# -----------------------------------------------------------------------------
# build_lid_index_splits
# -----------------------------------------------------------------------------


def test_build_lid_index_splits_writes_train_and_test_files_in_same_directory(tmp_path):
    """build_lid_index_splits writes lid_index_train.pkl and lid_index_test.pkl next to index_path."""
    # Arrange
    index_path = tmp_path / "lid_index.pkl"
    index = _make_lid_index()
    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=4)

    # Act
    build_lid_index_splits(str(index_path), test_size=0.2, random_state=42)

    # Assert
    train_path = tmp_path / "lid_index_train.pkl"
    test_path = tmp_path / "lid_index_test.pkl"
    assert train_path.exists()
    assert test_path.exists()


def test_build_lid_index_splits_train_and_test_are_disjoint_and_cover_original(tmp_path):
    """Train and test keys are disjoint and their union equals the original index keys."""
    # Arrange
    index_path = tmp_path / "lid_index.pkl"
    index = _make_lid_index()
    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=4)

    # Act
    build_lid_index_splits(str(index_path), test_size=0.2, random_state=42)

    # Assert
    with open(tmp_path / "lid_index_train.pkl", "rb") as f:
        train = pickle.load(f)
    with open(tmp_path / "lid_index_test.pkl", "rb") as f:
        test = pickle.load(f)

    train_keys = set(train.keys())
    test_keys = set(test.keys())
    assert train_keys.isdisjoint(test_keys)
    assert train_keys | test_keys == set(index.keys())


def test_build_lid_index_splits_output_structure_same_as_input(tmp_path):
    """Loaded train and test are dicts mapping (tar_number, filename) -> lang."""
    # Arrange
    index_path = tmp_path / "lid_index.pkl"
    index = _make_lid_index()
    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=4)

    # Act
    build_lid_index_splits(str(index_path), test_size=0.2, random_state=42)

    # Assert
    with open(tmp_path / "lid_index_train.pkl", "rb") as f:
        train = pickle.load(f)
    with open(tmp_path / "lid_index_test.pkl", "rb") as f:
        test = pickle.load(f)

    for d in (train, test):
        for key, val in d.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(val, str)


@pytest.mark.parametrize("test_size", [0.1, 0.2, 0.5])
def test_build_lid_index_splits_respects_test_size(tmp_path, test_size):
    """Test set size is approximately test_size of total (stratified split)."""
    # Arrange: enough samples for stratification (2 langs, 15 each)
    index_path = tmp_path / "lid_index.pkl"
    index = _make_lid_index({"en": 15, "es": 15})
    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=4)

    # Act
    build_lid_index_splits(str(index_path), test_size=test_size, random_state=42)

    # Assert
    with open(tmp_path / "lid_index_test.pkl", "rb") as f:
        test = pickle.load(f)

    n_total = len(index)
    n_test = len(test)
    ratio = n_test / n_total
    assert test_size - 0.05 <= ratio <= test_size + 0.05


def test_build_lid_index_splits_same_random_state_gives_same_split(tmp_path):
    """Same random_state produces identical train/test split when called twice."""
    # Arrange
    index_path = tmp_path / "lid_index.pkl"
    index = _make_lid_index()
    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=4)

    # Act
    build_lid_index_splits(str(index_path), test_size=0.2, random_state=99)
    with open(tmp_path / "lid_index_train.pkl", "rb") as f:
        train_a = set(pickle.load(f).keys())
    with open(tmp_path / "lid_index_test.pkl", "rb") as f:
        test_a = set(pickle.load(f).keys())

    build_lid_index_splits(str(index_path), test_size=0.2, random_state=99)
    with open(tmp_path / "lid_index_train.pkl", "rb") as f:
        train_b = set(pickle.load(f).keys())
    with open(tmp_path / "lid_index_test.pkl", "rb") as f:
        test_b = set(pickle.load(f).keys())

    # Assert
    assert train_a == train_b
    assert test_a == test_b


# -----------------------------------------------------------------------------
# build_lid_index
# -----------------------------------------------------------------------------


def test_build_lid_index_raises_when_hf_token_not_set():
    """build_lid_index raises ValueError when hf_token is None and HF_TOKEN env is not set."""
    # Arrange: patch getenv so it returns None (simulating unset env)
    # Act & Assert
    with patch("ups_challenge.dataloaders.build_index.os.getenv", return_value=None):
        with pytest.raises(ValueError, match="HF_TOKEN is not set"):
            build_lid_index(index_path="/tmp/data/lid_index.pkl", hf_token=None)


def test_build_lid_index_uses_provided_hf_token_when_given(tmp_path):
    """When hf_token is provided, build_lid_index does not require HF_TOKEN env."""
    # Arrange: pre-create jsonl; need enough samples so StratifiedShuffleSplit works (n_test >= n_classes, so ≥10 total with 2 classes)
    lid_folder = tmp_path / "data"
    lid_folder.mkdir()
    index_path = str(lid_folder / "lid_index.pkl")
    jsonl_lines = [
        f'{{"tar_number": "0001", "filepath": "/some/0001/en_{i}.mp3", "prediction": "en"}}\n'
        for i in range(5)
    ] + [
        f'{{"tar_number": "0001", "filepath": "/some/0001/es_{i}.mp3", "prediction": "es"}}\n'
        for i in range(5)
    ]
    (lid_folder / "lang_id_results.jsonl").write_text("".join(jsonl_lines), encoding="utf-8")

    # Act (hf_token can be dummy since we're not downloading)
    with patch.dict("os.environ", {}, clear=True):
        build_lid_index(index_path=index_path, hf_token="dummy_token")

    # Assert
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    assert index[("0001", "en_0.mp3")] == "en"
    assert index[("0001", "es_0.mp3")] == "es"
    assert len(index) == 10


def test_build_lid_index_creates_lid_folder_when_missing(tmp_path):
    """build_lid_index creates the parent directory of index_path if it does not exist."""
    # Arrange: parent of index_path does not exist; mock download so jsonl is written after makedirs
    lid_folder = tmp_path / "new_data"
    index_path = str(lid_folder / "lid_index.pkl")
    assert not lid_folder.exists()

    mock_jsonl = "".join(
        [f'{{"tar_number": "0001", "filepath": "/x/0001/en_{i}.mp3", "prediction": "en"}}\n' for i in range(5)]
        + [f'{{"tar_number": "0001", "filepath": "/x/0001/es_{i}.mp3", "prediction": "es"}}\n' for i in range(5)]
    )
    with patch("requests.get") as mock_get:
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = mock_jsonl.encode("utf-8")
        # Act
        build_lid_index(index_path=index_path, hf_token="dummy")

    # Assert
    assert lid_folder.exists()
    assert (lid_folder / "lid_index.pkl").exists()


def test_build_lid_index_writes_index_with_expected_structure(tmp_path):
    """Built index is a pickle dict mapping (tar_number, filename) -> prediction string."""
    # Arrange: need ≥10 samples (2 classes) so StratifiedShuffleSplit test set has ≥2
    lid_folder = tmp_path / "data"
    lid_folder.mkdir()
    index_path = str(lid_folder / "lid_index.pkl")
    lines = (
        [f'{{"tar_number": "42", "filepath": "/any/path/42/fr_{i}.mp3", "prediction": "fr"}}\n' for i in range(5)]
        + [f'{{"tar_number": "42", "filepath": "/any/path/42/de_{i}.mp3", "prediction": "de"}}\n' for i in range(5)]
    )
    (lid_folder / "lang_id_results.jsonl").write_text("".join(lines), encoding="utf-8")

    # Act
    build_lid_index(index_path=index_path, hf_token="dummy")

    # Assert
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    assert isinstance(index, dict)
    assert index[("42", "fr_0.mp3")] == "fr"
    assert index[("42", "de_0.mp3")] == "de"


def test_build_lid_index_skips_empty_lines_in_jsonl(tmp_path):
    """Empty lines in lang_id_results.jsonl are skipped."""
    # Arrange: ≥10 samples (2 classes) for split; one empty line in the middle
    lid_folder = tmp_path / "data"
    lid_folder.mkdir()
    index_path = str(lid_folder / "lid_index.pkl")
    en_lines = [f'{{"tar_number": "1", "filepath": "/a/1/en_{i}.mp3", "prediction": "en"}}\n' for i in range(5)]
    es_lines = [f'{{"tar_number": "2", "filepath": "/a/2/es_{i}.mp3", "prediction": "es"}}\n' for i in range(5)]
    (lid_folder / "lang_id_results.jsonl").write_text(
        "".join(en_lines) + "\n" + "".join(es_lines),
        encoding="utf-8",
    )

    # Act
    build_lid_index(index_path=index_path, hf_token="dummy")

    # Assert
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    assert len(index) == 10
    assert ("1", "en_0.mp3") in index
    assert ("2", "es_0.mp3") in index


def test_build_lid_index_calls_build_lid_index_splits_and_writes_splits(tmp_path):
    """build_lid_index produces lid_index.pkl and train/test split files."""
    # Arrange: need ≥10 samples (2 classes) so StratifiedShuffleSplit succeeds
    lid_folder = tmp_path / "data"
    lid_folder.mkdir()
    index_path = str(lid_folder / "lid_index.pkl")
    lines = (
        [f'{{"tar_number": "1", "filepath": "/a/1/en_{i}.mp3", "prediction": "en"}}\n' for i in range(5)]
        + [f'{{"tar_number": "1", "filepath": "/a/1/es_{i}.mp3", "prediction": "es"}}\n' for i in range(5)]
    )
    (lid_folder / "lang_id_results.jsonl").write_text("".join(lines), encoding="utf-8")

    # Act
    build_lid_index(index_path=index_path, hf_token="dummy")

    # Assert
    assert (lid_folder / "lid_index.pkl").exists()
    assert (lid_folder / "lid_index_train.pkl").exists()
    assert (lid_folder / "lid_index_test.pkl").exists()
