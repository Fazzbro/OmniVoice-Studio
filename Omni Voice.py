#!/usr/bin/env python3
"""
HuggingFace Space entry point for OmniVoice demo.

"""

import logging
import os
from typing import Any, Dict

# Configure cache directories on D drive before any model/hub libraries are imported.
# This ensures large model checkpoints (~3.36 GB) are downloaded to D drive instead of C drive.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_BASE_DIR, "models_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = os.path.join(_CACHE_DIR, "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(_CACHE_DIR, "huggingface", "hub")
os.environ["TORCH_HOME"] = os.path.join(_CACHE_DIR, "torch")

# Since all models are already downloaded to your D drive, enable offline mode.
# This prevents Hugging Face from hanging/waiting on slow internet checks to huggingface.co on every startup.
# (If you ever need to download a new model checkpoint in the future, comment out the two lines below.)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.DEBUG)

import sys
sys.path.insert(0, _BASE_DIR)

import numpy as np
try:
    import spaces
except ImportError:
    class _SpacesDummy:
        @staticmethod
        def GPU(fn=None, *args, **kwargs):
            if fn is not None and callable(fn):
                return fn
            def decorator(f):
                return f
            return decorator
    spaces = _SpacesDummy()
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.cli.demo import build_demo

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
CHECKPOINT = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")

print(f"Loading model from {CHECKPOINT} to cuda ...")
model = OmniVoice.from_pretrained(
    CHECKPOINT,
    device_map="cuda",
    dtype=torch.float16,
    load_asr=True,
)
sampling_rate = model.sampling_rate
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------


def _gen_core(
    text,
    language,
    ref_audio,
    instruct,
    num_step,
    guidance_scale,
    denoise,
    speed,
    duration,
    preprocess_prompt,
    postprocess_output,
    mode,
    ref_text=None,
    prompt_file=None,
):
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None

    kw: Dict[str, Any] = dict(
        text=text.strip(), language=lang, generation_config=gen_config
    )

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if prompt_file:
            try:
                from omnivoice import VoiceClonePrompt
                kw["voice_clone_prompt"] = VoiceClonePrompt.load(prompt_file)
            except Exception as e:
                return None, f"Error loading prompt file: {e}"
        elif not ref_audio:
            return None, "Please upload a reference audio or select a saved .pt prompt file."
        else:
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

    if instruct and instruct.strip():
        kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    waveform = (audio[0] * 32767).astype(np.int16)
    return (sampling_rate, waveform), "Done."


# ---------------------------------------------------------------------------
# ZeroGPU wrapper
# ---------------------------------------------------------------------------


@spaces.GPU(duration=60)
def generate_fn(*args, **kwargs):
    return _gen_core(*args, **kwargs)


# ---------------------------------------------------------------------------
# Build and launch demo
# ---------------------------------------------------------------------------
demo = build_demo(model, CHECKPOINT, generate_fn=generate_fn)

if __name__ == "__main__":
    demo.queue().launch()
