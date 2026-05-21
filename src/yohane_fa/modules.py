import torch.nn as nn
import torch.nn.functional as F
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Attention,
    Wav2Vec2EncoderLayerStableLayerNorm,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2FeatureProjection,
    Wav2Vec2FeedForward,
)


class RMSNormWav2Vec2FeatureProjection(Wav2Vec2FeatureProjection):
    """Feature projection with RMSNorm and biasless linear."""

    def __init__(self, config):
        super().__init__(config)
        self.layer_norm = nn.RMSNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size, bias=False)


class BiaslessWav2Vec2Attention(Wav2Vec2Attention):
    """Multi-head attention with biasless projections."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        is_decoder=False,
        bias=True,
        is_causal=False,
        config=None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            bias=False,
            is_causal=is_causal,
            config=config,
        )


class SwiGLUWav2Vec2FeedForward(Wav2Vec2FeedForward):
    """SwiGLU feed-forward with biasless linear layers."""

    def __init__(self, config):
        super(Wav2Vec2FeedForward, self).__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.output_dropout = nn.Dropout(config.hidden_dropout)
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dropout(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        hidden_states = self.output_dropout(self.down_proj(hidden_states))
        return hidden_states


class RMSNormWav2Vec2EncoderLayerStableLayerNorm(Wav2Vec2EncoderLayerStableLayerNorm):
    """Pre-norm encoder layer with RMSNorm instead of LayerNorm."""

    def __init__(self, config):
        super().__init__(config)
        self.layer_norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.final_layer_norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )


class RMSNormWav2Vec2EncoderStableLayerNorm(Wav2Vec2EncoderStableLayerNorm):
    """Encoder with RMSNorm instead of the outer LayerNorm."""

    def __init__(self, config):
        super().__init__(config)
        self.layer_norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
