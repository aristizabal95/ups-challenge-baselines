import os
import braceexpand
import webdataset as wds
import torch
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm
import random

from io import BytesIO

from datasets import load_dataset
import pickle

target_langs = ['en']
def build_urls(langs):
    with open("lid_index.pkl", "rb") as f:
        lid_index = pickle.load(f)
    
    
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN is not set")
    if len(langs) > 0:
        all_relevant_tar_numbers = set()
        for (tar_number, filename), lang in tqdm(lid_index.items()):
            if lang in langs:
                all_relevant_tar_numbers.add(tar_number)
        all_relevant_tar_numbers = list(all_relevant_tar_numbers)
        urls = []
        for tar_number in all_relevant_tar_numbers:
            if (int(tar_number) <= 5000):
                urls.append(f"https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/audio/{tar_number}.tar?download=True")
            else:
                urls.append(f"https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/audio2/{tar_number}.tar?download=True")
        token = f'Authorization:Bearer {token}'
        urls = [f"pipe:curl -s -L {url} -H {token}" for url in urls]
        return urls
    else:
        # Choose the number of tars to download
        url = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/audio/{000001..000004}.tar?download=True"
        token = f'Authorization:Bearer {token}'
        urls = list(braceexpand.braceexpand(url))
        urls = [f"pipe:curl -s -L {url} -H {token}" for url in urls]
        return urls


def decode_and_normalize_no_lang(sample,
                         target_sr=16000,
                         chunk_sec=10.0,
                         max_chunks_per_example=16,
                         shuffle_chunks=False,
                        ):
    """
    sample comes from .to_tuple('mp3', '__key__', '__url__')
    so it's (mp3_bytes, key, url).

    We:
      - decode mp3 using torchaudio
      - resample to default_sample_rate
      - convert to mono
      - return a dict

    Any samples that fail to decode are logged and skipped.
    """
    mp3_bytes, key, url = sample
    chunk_samples = int(chunk_sec * target_sr)

    # Extract TAR number from key/url for language lookup and logging
    tar_number = url.split("/")[-1].split(".")[0]
    
    output_chunks = []
    
    decoder = AudioDecoder(source=mp3_bytes,
                           sample_rate=target_sr,
                           num_channels=1
                           )
    
    duration = decoder.metadata.duration_seconds_from_header
    total_samples = int(duration * target_sr)
    
     # ---- 2) If short file, stream entire audio ----
    if duration <= chunk_sec:
        samples = decoder.get_samples_played_in_range(0.0, duration)
        chunk = samples.data
        chunk = chunk.squeeze(0)

        # pad to exact chunk length
        if chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        output_chunks.append(chunk)
        batch_wave = torch.stack(output_chunks)
        attention_mask = torch.ones_like(batch_wave, dtype=torch.long)
        return {
            "input_values": batch_wave,      # [N_chunks, chunk_samples]
            "attention_mask": attention_mask # same shape
        }

    # ---- 3) Choose random chunk start times (in seconds) ----
    max_start_sec = duration - chunk_sec

    # Generate random starting times
    start_times = [
        random.uniform(0.0, max_start_sec)
        for _ in range(max_chunks_per_example)
    ]

    # ---- 4) Stream each chunk ----
    for start_sec in start_times:
        end_sec = start_sec + chunk_sec

        samples = decoder.get_samples_played_in_range(start_sec, end_sec)

        chunk = samples.data
        chunk = chunk.squeeze(0)
        # Pad end-of-file short outputs
        if chunk.shape[-1] < chunk_samples:
            pad = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        output_chunks.append(chunk)

    # ---- 5) Shuffle chunks across examples ----
    if shuffle_chunks:
        random.shuffle(output_chunks)

    # ---- 6) Stack into batch tensors ----
    batch_wave = torch.stack(output_chunks)

    attention_mask = torch.ones_like(batch_wave, dtype=torch.long)

    return {
        "input_values": batch_wave,      # [N_chunks, chunk_samples]
        "attention_mask": attention_mask # same shape
    }

def collate_fn(batch):
    """
    Custom collate function to:
      - pad variable-length audio to the max length in the batch
      - stack metadata into convenient structures.
    """
    # Filter out any Nones that might slip through
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    input_values = [b["input_values"] for b in batch]

    attention_masks = [b["attention_mask"] for b in batch]
    return {
        "input_values": torch.cat(input_values, dim=0),              # (sum_N_chunks, T)
        "attention_mask": torch.cat(attention_masks, dim=0),
    }
