"""
Language identification (LID) dataset: decode audio from index and extract
embeddings with a frozen wav2vec2 model for linear classifier training.
"""
import pickle

import torch
import numpy as np
import webdataset as wds
from torchcodec.decoders import AudioDecoder

from ups_challenge.utils import LimitedDataset
from ups_challenge.dataloaders.urls import build_urls


def decode_with_lang(sample, lid_index, target_sr=16000, chunk_sec=10.0):
    """
    Decode audio sample and extract language label from index.

    Args:
        sample: (mp3_bytes, key, url) tuple from WebDataset
        lid_index: dict mapping (tar_number, filename) -> lang
        target_sr: Target sample rate
        chunk_sec: Chunk duration in seconds
    Returns:
        dict with 'waveform' and 'lang', or None if sample not in index
    """
    mp3_bytes, key, url = sample

    try:
        tar_number = None
        if "audio/" in url:
            after_audio = url.split("audio/")[1]
            tar_number = after_audio.split(".tar")[0]
        elif "audio2/" in url:
            after_audio = url.split("audio2/")[1]
            tar_number = after_audio.split(".tar")[0]

        if tar_number is None:
            return None

        if "/" in key:
            filename = key.split("/")[-1]
        else:
            filename = key
        filename += ".mp3"

        index_key = (tar_number, filename)
        if index_key not in lid_index:
            return None

        lang = lid_index[index_key]

        decoder = AudioDecoder(
            source=mp3_bytes, sample_rate=target_sr, num_channels=1
        )
        duration = decoder.metadata.duration_seconds_from_header

        if duration < chunk_sec:
            return None

        max_start_sec = max(0.0, duration - chunk_sec)
        start_sec = np.random.uniform(0.0, max_start_sec)
        end_sec = start_sec + chunk_sec

        samples = decoder.get_samples_played_in_range(start_sec, end_sec)
        chunk = samples.data.squeeze(0)
        chunk_np = chunk.cpu().numpy()

        return {
            "waveform": chunk_np,
            "lang": lang,
            "key": key,
        }
    except Exception:  # pylint: disable=broad-except
        return None


def extract_embedding_from_sample(
    sample, model, feature_extractor, device, lang_to_idx
):
    """
    Extract embedding from a single sample and convert lang to index.

    Args:
        sample: dict with 'waveform' (numpy array) and 'lang' (str)
        model: wav2vec2 AutoModel (frozen)
        feature_extractor: AutoFeatureExtractor
        device: torch device
        lang_to_idx: dict mapping language strings to integer indices
    Returns:
        dict with 'embedding' and 'label' tensors, or None if lang not mapped
    """
    if sample is None:
        return None

    waveform = sample["waveform"]
    lang = sample["lang"]

    if lang not in lang_to_idx:
        return None

    inputs = feature_extractor(
        [waveform],
        sampling_rate=feature_extractor.sampling_rate,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state.mean(dim=1)

    embedding = hidden_states.squeeze(0).cpu().float()
    label = torch.tensor(lang_to_idx[lang], dtype=torch.long)

    return {"embedding": embedding, "label": label}


def build_lid_dataset(
    index_path,
    model,
    feature_extractor,
    device,
    lang_to_idx,
    hf_token=None,
    target_sr=16000,
    chunk_sec=10.0,
    max_samples=None,
):
    """
    Build a WebDataset that extracts embeddings from samples in the lid_index.

    Args:
        index_path: Path to lid_index pickle file
        model: Frozen wav2vec2 model
        feature_extractor: AutoFeatureExtractor
        device: torch device
        lang_to_idx: dict mapping language strings to integer indices
        hf_token: HuggingFace token
        target_sr: Target sample rate
        chunk_sec: Chunk duration
        max_samples: Maximum number of valid samples to include (None for all)
    Returns:
        Dataset yielding dicts with 'embedding' and 'label'
    """
    with open(index_path, "rb") as f:
        lid_index = pickle.load(f)

    urls = build_urls(langs=[], index_path=index_path, hf_token=hf_token)

    dataset = (
        wds.WebDataset(
            urls,
            shardshuffle=False,
            handler=wds.handlers.ignore_and_continue,
        )
        .to_tuple(
            "mp3", "__key__", "__url__",
            handler=wds.handlers.ignore_and_continue,
        )
        .map(
            lambda s: decode_with_lang(
                s, lid_index, target_sr=target_sr, chunk_sec=chunk_sec
            ),
        )
        .map(
            lambda s: extract_embedding_from_sample(
                s, model, feature_extractor, device, lang_to_idx
            ),
        )
    )

    if max_samples is not None:
        dataset = LimitedDataset(dataset, max_samples)

    return dataset


def collate_embeddings(batch):
    """Collate embeddings and labels from WebDataset samples."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    embeddings = torch.stack([b["embedding"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return embeddings, labels
