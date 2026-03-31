"""MLX-native inference for MERaLiON AudioLLM on Apple Silicon.

Quick start:
    from mlx_meralion import load_model, transcribe

    model = load_model("MERaLiON/MERaLiON-2-10B-MLX")
    text = transcribe(model, "audio.wav")
"""

__version__ = "0.1.0"

from .inference import (
    load_model as load_model,
    transcribe as transcribe,
    transcribe_batch as transcribe_batch,
    LoadedModel as LoadedModel,
)
from .processor import get_task_prompt as get_task_prompt
