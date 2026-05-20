import re
import unicodedata
from typing import Any, TypedDict, cast

import lightning as L
import torch
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader
from torchcodec.decoders import AudioDecoder
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)


class DatasetTiming(TypedDict):
    text: str
    start: int
    end: int


class DatasetRow(TypedDict):
    audio: AudioDecoder
    timings: list[list[DatasetTiming]]


def _normalize_text(text: str) -> str:
    text = text.casefold()
    text = text.replace("\u2019", "'")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z']", " ", text)
    return re.sub(r" +", " ", text).strip()


class TimingsDataset(L.LightningDataModule):
    def __init__(
        self,
        dataset_path: str = "NextFire/karaoke-mugen-timings",
        max_duration_seconds: int = 300,
        n_eval: int = 1000,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.max_duration_seconds = max_duration_seconds
        self.n_eval = n_eval
        self.batch_size = batch_size
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
        self.test_dataset: IterableDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if all((self.train_dataset, self.val_dataset, self.test_dataset)):
            return
        dataset = load_dataset(self.dataset_path, split="train", streaming=True)
        dataset = dataset.filter(
            lambda x: x.metadata.duration_seconds <= self.max_duration_seconds,
            input_columns=["audio"],
        )
        dataset = dataset.map(self._prepare, remove_columns=dataset.column_names)
        self.test_dataset = dataset.take(self.n_eval // 2)
        self.val_dataset = dataset.skip(self.n_eval // 2).take(self.n_eval // 2)
        self.train_dataset = dataset.skip(self.n_eval)

    def _prepare(self, example: DatasetRow) -> dict[str, Any]:
        audio = example["audio"]
        input_values = self.processor(
            audio=audio["array"],  # type: ignore
            sampling_rate=audio["sampling_rate"],  # type: ignore
        ).input_values[0]

        get_feat_extract_output_lengths = (
            self.trainer.lightning_module.wav2vec2._get_feat_extract_output_lengths  # type: ignore
        )
        output_length = int(get_feat_extract_output_lengths(len(input_values)))  # type: ignore
        duration_ms = float(audio.metadata.duration_seconds * 1000)  # type: ignore
        frames_per_ms = output_length / duration_ms
        labels = [self.processor.tokenizer.pad_token_id] * output_length
        for line in example["timings"]:
            for timing in line:
                value = cast(str, _normalize_text(timing["text"]))
                if not value.strip():
                    continue
                start_frame = round(timing["start"] * frames_per_ms)
                assert start_frame < output_length
                end_frame = round(timing["end"] * frames_per_ms)
                assert start_frame <= end_frame <= output_length
                n_frames = end_frame - start_frame
                input_ids = self.processor(text=value).input_ids
                for idx, frame in enumerate(range(start_frame, end_frame)):
                    if labels[frame] == self.processor.tokenizer.pad_token_id:
                        token_idx = (idx * len(input_ids)) // n_frames
                        labels[frame] = input_ids[token_idx]
                    else:
                        # Mask overlapping alignments with -100 to ignore them in loss calculation
                        labels[frame] = -100

        return {"input_values": input_values, "labels": labels}

    def train_dataloader(self) -> Any:
        assert self.train_dataset
        return DataLoader(
            self.train_dataset.with_format("torch"),  # type: ignore
            collate_fn=self._collate,
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> Any:
        assert self.val_dataset
        return DataLoader(
            self.val_dataset.with_format("torch"),  # type: ignore
            batch_size=self.batch_size,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> Any:
        assert self.test_dataset
        return DataLoader(
            self.test_dataset.with_format("torch"),  # type: ignore
            batch_size=self.batch_size,
            collate_fn=self._collate,
        )

    def _collate(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_values = [e["input_values"] for e in examples]
        labels = [e["labels"] for e in examples]

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
