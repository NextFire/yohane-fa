import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config

from yohane_fa.modules import build_patched_wav2vec2model


class YohaneForcedAligner(L.LightningModule):
    def __init__(
        self,
        config: Wav2Vec2Config,
        gradient_checkpointing: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        super().__init__()
        self.wav2vec2 = build_patched_wav2vec2model(config)
        if gradient_checkpointing:
            self.wav2vec2.gradient_checkpointing_enable()
        if attn_implementation:
            self.wav2vec2.set_attn_implementation(attn_implementation)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size or 32,
            bias=False,
        )
        self.save_hyperparameters()

    def on_fit_start(self) -> None:
        torch.compile(self, dynamic=True)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        loss = None
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

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, _ = self(**batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, logits = self(**batch)
        labels = batch["labels"]
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            valid_mask = labels != -100
            frame_accuracy = (preds[valid_mask] == labels[valid_mask]).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_frame_accuracy", frame_accuracy, prog_bar=True)
