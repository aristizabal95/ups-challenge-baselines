import random
import torch
import webdataset as wds
from torchcodec.decoders import AudioDecoder
from .urls import build_urls


def decode_contrastive(sample, target_sr=16000, chunk_sec=10.0, min_offset_frac=0.1):
    """Decode one mp3 into two random chunks (anchor & positive), with correct attention masks."""
    mp3_bytes, key, url = sample
    try:
        decoder = AudioDecoder(source=mp3_bytes, sample_rate=target_sr, num_channels=1)
        duration = decoder.metadata.duration_seconds_from_header
    except Exception:
        return None

    chunk_samples = int(chunk_sec * target_sr)
    if duration < chunk_sec:
        return None  # too short even for one chunk

    max_start_sec = max(0.0, duration - chunk_sec)
    anchor_start = random.uniform(0.0, max_start_sec)
    offset = random.uniform(min_offset_frac * chunk_sec, max_start_sec)
    positive_start = (anchor_start + offset) % max_start_sec

    def extract(start_sec):
        """
        Return (chunk_tensor, attention_mask) where:
           - chunk_tensor: FloatTensor [chunk_samples]
           - attention_mask: LongTensor [chunk_samples] with 1 for real, 0 for pad
        """
        end_sec = start_sec + chunk_sec
        samples = decoder.get_samples_played_in_range(start_sec, end_sec)
        chunk = samples.data.squeeze(0)

        orig_len = chunk.shape[-1]
        if orig_len >= chunk_samples:
            # If for some reason returned more, trim. Usually it should be <= chunk_samples.
            chunk = chunk[..., :chunk_samples]
            mask = torch.ones(chunk_samples, dtype=torch.long)
        else:
            # Pad to chunk_samples and build mask
            pad = chunk_samples - orig_len
            chunk = torch.nn.functional.pad(chunk, (0, pad))
            mask = torch.cat([torch.ones(orig_len, dtype=torch.long),
                              torch.zeros(pad, dtype=torch.long)], dim=0)

        return chunk, mask

    anchor_chunk, anchor_mask = extract(anchor_start)
    positive_chunk, positive_mask = extract(positive_start)

    def make_entry(chunk, mask):
        return {
            "input_values": chunk,
            "attention_mask": mask,
        }

    return {
        "anchor": make_entry(anchor_chunk, anchor_mask),
        "positive": make_entry(positive_chunk, positive_mask),
        "meta": {"key": key, "url": url, "duration": duration},
    }


def collate_contrastive(batch):
    """Build batch dict with anchor, positive, and negative sets (including masks)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Extract inputs and masks
    anchors = [b["anchor"]["input_values"] for b in batch]
    positives = [b["positive"]["input_values"] for b in batch]
    anchor_masks = [b["anchor"]["attention_mask"] for b in batch]
    positive_masks = [b["positive"]["attention_mask"] for b in batch]

    # negatives: randomly shuffle anchors (and their masks) to ensure mask matches audio
    perm = list(range(len(anchors)))
    random.shuffle(perm)
    negatives = [anchors[i] for i in perm]
    negative_masks = [anchor_masks[i] for i in perm]

    # Stack into tensors: shape (B, T)
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    anchor_masks = torch.stack(anchor_masks)
    positive_masks = torch.stack(positive_masks)
    negative_masks = torch.stack(negative_masks)

    return {
        "anchor": {
            "input_values": anchors,
            "attention_mask": anchor_masks,
        },
        "positive": {
            "input_values": positives,
            "attention_mask": positive_masks,
        },
        "negative": {
            "input_values": negatives,
            "attention_mask": negative_masks,
        },
    }


def build_contrastive_dataset(
    langs=None,
    index_path="./data/lid_index.pkl",
    hf_token=None,
    target_sr=16000,
    chunk_sec=10.0,
):
    """Return a WebDataset yielding dicts with anchor/positive audio chunks (with masks)."""
    if langs is None:
        langs = []

    urls = build_urls(langs, index_path=index_path, hf_token=hf_token)

    dataset = (
        wds.WebDataset(urls, shardshuffle=True)
        .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
        .map(lambda s: decode_contrastive(s, target_sr=target_sr, chunk_sec=chunk_sec))
    )

    return dataset
