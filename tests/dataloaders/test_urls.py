"""
Tests for ups_challenge.dataloaders.urls.

Each test is atomic and follows Arrange, Act, Assert.
Tests assert on the public interface of each function, not implementation details.
"""
import io
import pickle

import pytest
from unittest.mock import patch

from ups_challenge.dataloaders.urls import build_urls


# -----------------------------------------------------------------------------
# build_urls
# -----------------------------------------------------------------------------


def test_build_urls_raises_when_hf_token_not_set():
    """build_urls raises ValueError when hf_token is None and HF_TOKEN env is not set."""
    # Arrange
    index_path = "/nonexistent/index.pkl"

    # Act & Assert
    with patch("ups_challenge.dataloaders.urls.os.getenv", return_value=None):
        with pytest.raises(ValueError, match="HF_TOKEN is not set"):
            build_urls(langs=[], index_path=index_path, hf_token=None)


def test_build_urls_returns_list_of_strings(tmp_path):
    """build_urls returns a list of URL strings."""
    # Arrange: minimal index so URLs are built
    index_path = tmp_path / "lid_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({("0001", "a.mp3"): "en"}, f, protocol=4)

    # Act
    result = build_urls(langs=[], index_path=str(index_path), hf_token="dummy_token")

    # Assert
    assert isinstance(result, list)
    assert all(isinstance(u, str) for u in result)


def test_build_urls_each_url_contains_pipe_curl_and_authorization(tmp_path):
    """Each returned URL contains 'pipe:curl' and the Bearer token."""
    # Arrange
    index_path = tmp_path / "lid_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({("0001", "a.mp3"): "en"}, f, protocol=4)

    # Act
    result = build_urls(langs=[], index_path=str(index_path), hf_token="my_token")

    # Assert
    assert len(result) >= 1
    for url in result:
        assert "pipe:curl" in url
        assert "Authorization:Bearer my_token" in url


def test_build_urls_tar_leq_5000_uses_audio_path(tmp_path):
    """Tar numbers <= 5000 produce URLs containing 'audio/' (not audio2/)."""
    # Arrange
    index_path = tmp_path / "lid_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({("1000", "a.mp3"): "en"}, f, protocol=4)

    # Act
    result = build_urls(langs=[], index_path=str(index_path), hf_token="dummy")

    # Assert
    assert len(result) == 1
    assert "audio/1000.tar" in result[0]
    assert "audio2/" not in result[0]


def test_build_urls_tar_gt_5000_uses_audio2_path(tmp_path):
    """Tar numbers > 5000 produce URLs containing 'audio2/'."""
    # Arrange
    index_path = tmp_path / "lid_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({("5001", "a.mp3"): "en"}, f, protocol=4)

    # Act
    result = build_urls(langs=[], index_path=str(index_path), hf_token="dummy")

    # Assert
    assert len(result) == 1
    assert "audio2/5001.tar" in result[0]


@pytest.mark.parametrize("tar_number,expected_subpath", [
    ("1", "audio/1.tar"),
    ("5000", "audio/5000.tar"),
    ("5001", "audio2/5001.tar"),
    ("10000", "audio2/10000.tar"),
])
def test_build_urls_audio_vs_audio2_by_tar_number(tmp_path, tar_number, expected_subpath):
    """URL subpath is audio/ for tar <= 5000, audio2/ for tar > 5000."""
    # Arrange
    index_path = tmp_path / "lid_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({(tar_number, "x.mp3"): "en"}, f, protocol=4)

    # Act
    result = build_urls(langs=[], index_path=str(index_path), hf_token="dummy")

    # Assert
    assert len(result) == 1
    assert expected_subpath in result[0]


def test_build_urls_empty_langs_includes_all_tar_numbers(tmp_path):
    """When langs is empty, all tar numbers in the index appear in the URLs."""
    # Arrange: two tar numbers, two languages
    index_path = tmp_path / "lid_index.pkl"
    lid_index = {
        ("0001", "a.mp3"): "en",
        ("0002", "b.mp3"): "es",
    }
    with open(index_path, "wb") as f:
        pickle.dump(lid_index, f, protocol=4)

    # Act
    result = build_urls(langs=[], index_path=str(index_path), hf_token="dummy")

    # Assert
    assert len(result) == 2
    tar_numbers_in_urls = []
    for u in result:
        if "audio/0001.tar" in u:
            tar_numbers_in_urls.append("0001")
        if "audio/0002.tar" in u:
            tar_numbers_in_urls.append("0002")
    assert set(tar_numbers_in_urls) == {"0001", "0002"}


def test_build_urls_filter_by_langs(tmp_path):
    """When langs is non-empty, only tar numbers with that language are included."""
    # Arrange: 0001=en, 0002=es, 0003=en
    index_path = tmp_path / "lid_index.pkl"
    lid_index = {
        ("0001", "a.mp3"): "en",
        ("0002", "b.mp3"): "es",
        ("0003", "c.mp3"): "en",
    }
    with open(index_path, "wb") as f:
        pickle.dump(lid_index, f, protocol=4)

    # Act
    result = build_urls(langs=["en"], index_path=str(index_path), hf_token="dummy")

    # Assert: only 0001 and 0003 (en), not 0002 (es)
    assert len(result) == 2
    urls_str = " ".join(result)
    assert "0001" in urls_str
    assert "0003" in urls_str
    assert "0002" not in urls_str


def test_build_urls_calls_build_lid_index_when_index_missing(tmp_path):
    """When index_path does not exist, build_lid_index is called."""
    # Arrange: path that does not exist; when code later opens it for read, return minimal index
    index_path_str = str(tmp_path / "missing" / "lid_index.pkl")
    minimal_index = {("0001", "a.mp3"): "en"}
    pickle_bytes = pickle.dumps(minimal_index, protocol=4)

    real_open = open

    def fake_open(path, mode="rb", *args, **kwargs):
        if path == index_path_str and "b" in mode:
            return io.BytesIO(pickle_bytes)
        return real_open(path, mode, *args, **kwargs)

    with patch("ups_challenge.dataloaders.urls.build_lid_index") as mock_build:
        with patch("ups_challenge.dataloaders.urls.open", fake_open):
            # Act
            build_urls(langs=[], index_path=index_path_str, hf_token="dummy")

    # Assert
    mock_build.assert_called_once()
    call_kwargs = mock_build.call_args[1]
    assert call_kwargs["hf_token"] == "dummy"


def test_build_urls_uses_env_hf_token_when_not_passed(tmp_path):
    """When hf_token is not passed, build_urls uses HF_TOKEN from environment."""
    # Arrange
    index_path = tmp_path / "lid_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({("0001", "a.mp3"): "en"}, f, protocol=4)

    # Act
    with patch("ups_challenge.dataloaders.urls.os.getenv", return_value="env_token"):
        result = build_urls(langs=[], index_path=str(index_path))

    # Assert
    assert len(result) >= 1
    assert "Authorization:Bearer env_token" in result[0]
