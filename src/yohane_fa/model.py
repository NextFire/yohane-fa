import math
from typing import TypedDict, Unpack

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers import initialization as init
from transformers.monkey_patching import (
    apply_patches,
    clear_patch_mapping,
    register_patch_mapping,
)

from yohane_fa.modules import (
    BiaslessWav2Vec2Attention,
    RMSNormWav2Vec2EncoderLayerStableLayerNorm,
    RMSNormWav2Vec2EncoderStableLayerNorm,
    RMSNormWav2Vec2FeatureProjection,
    SwiGLUWav2Vec2FeedForward,
)


class CustomWav2Vec2Model(Wav2Vec2Model):
    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, RMSNormWav2Vec2FeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            init.uniform_(module.projection.weight, a=-k, b=k)
        else:
            super()._init_weights(module)


def _build_patched_wav2vec2model(config: Wav2Vec2Config) -> Wav2Vec2Model:
    try:
        register_patch_mapping(
            mapping={
                "Wav2Vec2FeatureProjection": RMSNormWav2Vec2FeatureProjection,
                "Wav2Vec2Attention": BiaslessWav2Vec2Attention,
                "Wav2Vec2FeedForward": SwiGLUWav2Vec2FeedForward,
                "Wav2Vec2EncoderLayerStableLayerNorm": RMSNormWav2Vec2EncoderLayerStableLayerNorm,
                "Wav2Vec2EncoderStableLayerNorm": RMSNormWav2Vec2EncoderStableLayerNorm,
            }
        )
        with apply_patches():
            return CustomWav2Vec2Model(config)
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
        attn_implementation: str | None = None,
        gradient_checkpointing: bool = False,
        z_loss_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.wav2vec2 = _build_patched_wav2vec2model(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,  # pyright: ignore[reportArgumentType]
            bias=False,
        )
        if attn_implementation:
            self.wav2vec2.set_attn_implementation(attn_implementation)
        if gradient_checkpointing:
            self.wav2vec2.gradient_checkpointing_enable()
        self.z_loss_weight = z_loss_weight
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
            focal_loss = ((1 - pt) ** 2 * ce).mean()
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss = self.z_loss_weight * (log_z**2).mean()
            loss = focal_loss + z_loss
        return loss, logits

    def training_step(self, batch: ModelInput, batch_idx: int) -> torch.Tensor:
        loss, _ = self(**batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: ModelInput, batch_idx: int) -> torch.Tensor:
        loss, logits = self(**batch)
        labels = batch["labels"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            valid_mask = labels != -100
            frame_accuracy = (preds[valid_mask] == labels[valid_mask]).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_frame_accuracy", frame_accuracy, prog_bar=True)
        return loss
