"""
Shared utilities for ups_challenge datasets and training.
"""
import torch


class LimitedDataset(torch.utils.data.IterableDataset):  # pylint: disable=abstract-method
    """
    Limit the number of valid (non-None) samples from an iterable dataset.
    Use with streaming WebDataset pipelines to cap how many samples are
    consumed (e.g. for quick experiments or balanced splits).
    """

    def __init__(self, dataset, max_samples):
        super().__init__()
        self.dataset = dataset
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        for sample in self.dataset:
            if sample is not None:
                yield sample
                count += 1
                if count >= self.max_samples:
                    break
