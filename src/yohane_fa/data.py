from typing import Any, TypedDict, cast

import lightning as L
import numpy as np
import torch
from datasets import IterableDataset, load_dataset
from numpy.typing import NDArray
from tokenizers import Regex
from tokenizers.normalizers import BertNormalizer, Replace, Sequence
from torch.utils.data import DataLoader
from torchcodec.decoders import AudioDecoder
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)

from yohane_fa.model import ModelInput


class DatasetTiming(TypedDict):
    text: str
    start: int
    end: int


class DatasetRow(TypedDict):
    audio: AudioDecoder
    timings: list[list[DatasetTiming]]


class PreparedExample(TypedDict):
    input_values: NDArray[np.float32]
    labels: list[int]


class TimingsDataset(L.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        n_eval: int,
        max_duration_seconds: int = 300,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.n_eval = n_eval
        self.max_duration_seconds = max_duration_seconds
        self.batch_size = batch_size

        self.normalizer = Sequence([BertNormalizer(), Replace(Regex(r"[^a-z']"), " ")])

        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            "./",
            word_delimiter_token="|",
            unk_token="[UNK]",
            pad_token="[PAD]",
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

        self.train_dataset: IterableDataset | None = None
        self.val_dataset: IterableDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if all((self.train_dataset, self.val_dataset)):
            return
        dataset = load_dataset(
            self.dataset_path,
            split=self.dataset_split,
            streaming=True,
        )
        dataset = dataset.filter(
            lambda x: x.metadata.duration_seconds <= self.max_duration_seconds,
            input_columns=["audio"],
        )
        dataset = dataset.map(self._prepare, remove_columns=dataset.column_names)
        self.val_dataset = dataset.take(self.n_eval)
        self.train_dataset = dataset.skip(self.n_eval)

    def train_dataloader(self) -> Any:
        assert self.train_dataset
        return DataLoader(
            self.train_dataset,  # pyright: ignore[reportArgumentType]
            collate_fn=self._collate,
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> Any:
        assert self.val_dataset
        return DataLoader(
            self.val_dataset,  # pyright: ignore[reportArgumentType]
            batch_size=self.batch_size,
            collate_fn=self._collate,
        )

    def _prepare(self, example: DatasetRow) -> PreparedExample:
        audio = example["audio"]
        inputs = self.processor(
            audio=audio["array"],  # pyright: ignore[reportIndexIssue]
            sampling_rate=audio["sampling_rate"],  # pyright: ignore[reportIndexIssue,reportCallIssue]
        )
        input_values = cast(NDArray[np.float32], inputs.input_values[0])

        output_length = self._get_feat_extract_output_lengths(len(input_values))
        duration_ms = float(audio.metadata.duration_seconds * 1000)  # pyright: ignore[reportOptionalOperand]
        frames_per_ms = output_length / duration_ms
        pad_token_id = cast(int, self.processor.tokenizer.pad_token_id)
        labels = [pad_token_id] * output_length
        for line in example["timings"]:
            for timing in line:
                value = cast(str, self.normalizer.normalize_str(timing["text"]))
                if not value:
                    continue
                start_frame = round(timing["start"] * frames_per_ms)
                assert start_frame < output_length
                end_frame = round(timing["end"] * frames_per_ms)
                assert start_frame <= end_frame <= output_length
                n_frames = end_frame - start_frame
                inputs = self.processor(text=value)
                input_ids = cast(list[int], inputs.input_ids)
                for idx, frame in enumerate(range(start_frame, end_frame)):
                    if labels[frame] == pad_token_id:
                        token_idx = (idx * len(input_ids)) // n_frames
                        labels[frame] = input_ids[token_idx]
                    else:
                        # Mask overlapping alignments with -100 to ignore them in loss calculation
                        labels[frame] = -100

        return {"input_values": input_values, "labels": labels}

    def _collate(self, examples: list[PreparedExample]) -> ModelInput:
        input_values = [torch.Tensor(e["input_values"]) for e in examples]
        labels = [torch.Tensor(e["labels"]) for e in examples]

        max_input_len = max(v.shape[0] for v in input_values)
        padded_input = torch.zeros(len(examples), max_input_len)
        attention_mask = torch.zeros(len(examples), max_input_len)
        for i, v in enumerate(input_values):
            padded_input[i, : v.shape[0]] = v
            attention_mask[i, : v.shape[0]] = 1

        max_label_len = max(lbl.shape[0] for lbl in labels)
        padded_labels = torch.full((len(examples), max_label_len), -100)
        for i, lbl in enumerate(labels):
            padded_labels[i, : lbl.shape[0]] = lbl

        return {
            "input_values": padded_input,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }

    def _get_feat_extract_output_lengths(self, input_length: int) -> int:
        wav2vec2 = cast(Wav2Vec2Model, self.trainer.lightning_module.wav2vec2)  # pyright: ignore[reportOptionalMemberAccess]
        return int(wav2vec2._get_feat_extract_output_lengths(input_length))
