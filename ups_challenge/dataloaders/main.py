from base import build_urls, decode_and_normalize_no_lang, collate_fn
import torch
from torch.utils.data import DataLoader
import webdataset as wds
from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from datasets import load_dataset
import datasets
import torch



def build_wds_dataset(langs):
    urls = build_urls(langs)
    return (wds.WebDataset(
        urls,
        shardshuffle=False,
    )
    .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
    .map(decode_and_normalize_no_lang)
)
    
if __name__ == "__main__":
    batch_size = 1
    num_workers = 1
    langs = ['en']
    wds_dataset = build_wds_dataset(langs)
    
    data_loader = torch.utils.data.DataLoader(
        wds_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

    for batch in data_loader:
        if batch is None:
            continue

        waveforms = [x.cpu().numpy() for x in batch["input_values"]]
        inputs = feature_extractor(
            waveforms,
            sampling_rate=feature_extractor.sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(
                input_values=inputs.input_values,
                attention_mask=getattr(inputs, "attention_mask", None),
            ).logits
        pred_ids = torch.argmax(logits, dim=-1)

        outputs = tokenizer.batch_decode(pred_ids, output_word_offsets=True)
        print(outputs)
        break