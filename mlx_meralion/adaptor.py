"""MLP adaptor module for MERaLiON.

Bridges the Whisper encoder output (1280-d, 1500 timesteps) to the
SEA-LION/Gemma2 decoder input (hidden_size-d, 100 timesteps).

Two variants:
    - MERaLiONSpeechAudioAdapter: v1 simple MLP (modeling_meralion.py)
    - MERaLiONSpeechAudioAdapterLarge: v2 gated MLP (modeling_meralion2.py)
"""

import mlx.core as mx
import mlx.nn as nn


class MERaLiONSpeechAudioAdapter(nn.Module):
    """Audio-to-text adapter with timestep compression (v1).

    Matches MERaLiONSpeechAudioAdaper from modeling_meralion.py.
    Architecture: reshape -> Linear+SiLU -> Linear+SiLU+Linear -> output
    """

    def __init__(
        self,
        speech_hidden_size: int = 1280,
        text_hidden_size: int = 3584,
        scale_factor: int = 15,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        mlp_input_dim = speech_hidden_size * scale_factor

        self.mlp_adapter = nn.Sequential(
            nn.Linear(mlp_input_dim, speech_hidden_size),
            nn.SiLU(),
        )

        self.speech_llm_proj = nn.Sequential(
            nn.Linear(speech_hidden_size, speech_hidden_size * 4),
            nn.SiLU(),
            nn.Linear(speech_hidden_size * 4, text_hidden_size),
        )

    def __call__(self, speech_embeds: mx.array) -> mx.array:
        B, T, D = speech_embeds.shape
        new_T = (T // self.scale_factor) * self.scale_factor
        speech_embeds = speech_embeds[:, :new_T, :]
        speech_embeds = speech_embeds.reshape(B, new_T // self.scale_factor, D * self.scale_factor)
        speech_embeds = self.mlp_adapter(speech_embeds)
        speech_embeds = self.speech_llm_proj(speech_embeds)
        return speech_embeds


class MERaLiONSpeechAudioAdapterLarge(nn.Module):
    """Audio-to-text adapter with gated linear unit (v2 / MERaLiON-2).

    Matches MERaLiON2SpeechAudioAdaperLarge from modeling_meralion2.py.
    Architecture: reshape -> Linear+SiLU -> GLU(gate_proj, pool_proj) -> out_proj
    """

    def __init__(
        self,
        speech_hidden_size: int = 1280,
        text_hidden_size: int = 3584,
        scale_factor: int = 15,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        mlp_input_dim = speech_hidden_size * scale_factor
        intermediate_dim = speech_hidden_size * 5

        self.mlp_adapter = nn.Sequential(
            nn.Linear(mlp_input_dim, intermediate_dim),
            nn.SiLU(),
        )

        self.gate_proj = nn.Linear(intermediate_dim, intermediate_dim)
        self.pool_proj = nn.Linear(intermediate_dim, intermediate_dim)
        self.act_fn = nn.SiLU()

        self.out_proj = nn.Linear(intermediate_dim, text_hidden_size)

    def __call__(self, speech_embeds: mx.array) -> mx.array:
        B, T, D = speech_embeds.shape
        new_T = (T // self.scale_factor) * self.scale_factor
        speech_embeds = speech_embeds[:, :new_T, :]
        speech_embeds = speech_embeds.reshape(B, new_T // self.scale_factor, D * self.scale_factor)

        speech_embeds = self.mlp_adapter(speech_embeds)
        speech_embeds = self.act_fn(self.gate_proj(speech_embeds)) * self.pool_proj(speech_embeds)
        speech_embeds = self.out_proj(speech_embeds)
        return speech_embeds


def create_adapter(
    variant: str,
    speech_hidden_size: int = 1280,
    text_hidden_size: int = 3584,
    scale_factor: int = 15,
) -> nn.Module:
    """Create the appropriate adapter variant.

    Args:
        variant: "v1" for simple MLP, "v2" or "large" for gated MLP
    """
    if variant in ("v2", "large", "meralion2"):
        return MERaLiONSpeechAudioAdapterLarge(speech_hidden_size, text_hidden_size, scale_factor)
    return MERaLiONSpeechAudioAdapter(speech_hidden_size, text_hidden_size, scale_factor)
