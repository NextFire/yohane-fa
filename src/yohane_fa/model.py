from typing import TypedDict, Unpack

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers.monkey_patching import (
    apply_patches,
    clear_patch_mapping,
    register_patch_mapping,
)

from yohane_fa.modules import (
    BiaslessWav2Vec2Attention,
    PatchedWav2Vec2Model,
    RMSNormWav2Vec2EncoderLayerStableLayerNorm,
    RMSNormWav2Vec2EncoderStableLayerNorm,
    RMSNormWav2Vec2FeatureProjection,
    SwiGLUWav2Vec2FeedForward,
)


def _build_patched_wav2vec2model(config: Wav2Vec2Config) -> Wav2Vec2Model:
    register_patch_mapping(
        mapping={
            "Wav2Vec2FeatureProjection": RMSNormWav2Vec2FeatureProjection,
            "Wav2Vec2Attention": BiaslessWav2Vec2Attention,
            "Wav2Vec2FeedForward": SwiGLUWav2Vec2FeedForward,
            "Wav2Vec2EncoderLayerStableLayerNorm": RMSNormWav2Vec2EncoderLayerStableLayerNorm,
            "Wav2Vec2EncoderStableLayerNorm": RMSNormWav2Vec2EncoderStableLayerNorm,
        }
    )
    try:
        with apply_patches():
            return PatchedWav2Vec2Model(config)
    finally:
        clear_patch_mapping()


class ModelInput(TypedDict, total=False):
    input_values: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class YohaneForcedAligner(L.LightningModule):
    def __init__(
        self,
        config: Wav2Vec2Config,
        gradient_checkpointing: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        super().__init__()
        self.wav2vec2 = _build_patched_wav2vec2model(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,  # pyright: ignore[reportArgumentType]
            bias=False,
        )
        if gradient_checkpointing:
            self.wav2vec2.gradient_checkpointing_enable()
        if attn_implementation:
            self.wav2vec2.set_attn_implementation(attn_implementation)
        self.save_hyperparameters()

    def forward(
        self,
        **kwargs: Unpack[ModelInput],
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        outputs = self.wav2vec2(**kwargs)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        loss = None
        labels = kwargs.get("labels")
        if labels is not None:
            ce = F.cross_entropy(
                logits.transpose(1, 2),
                labels,
                ignore_index=-100,
                reduction="none",
            )
            ce = ce[labels != -100]
            pt = torch.exp(-ce)
            loss = ((1 - pt) ** 2 * ce).mean()
        return loss, logits

    def training_step(self, batch: ModelInput, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: ModelInput, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="val")

    def _shared_step(
        self,
        batch: ModelInput,
        batch_idx: int,
        *,
        stage: str,
    ) -> torch.Tensor:
        loss, logits = self(**batch)
        labels = batch["labels"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            valid_mask = labels != -100
            frame_accuracy = (preds[valid_mask] == labels[valid_mask]).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_frame_accuracy", frame_accuracy, prog_bar=True)
        return loss
