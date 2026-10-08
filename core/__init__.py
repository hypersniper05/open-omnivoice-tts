"""Shared OmniVoice TTS core.

The single source of truth for loading and running the OmniVoice model:
model loading, GPU selection and memory cap, the voice-prompt cache, ffmpeg
resolution, idle offload and out-of-memory recovery. ``app/`` (the
OpenAI-compatible API server) is built on it.

Usage (after putting the project root on ``sys.path``)::

    from core import ENGINE, ensure_ffmpeg_on_path
"""

from .omnivoice_engine import (  # noqa: F401
    ENGINE,
    GPUMemoryError,
    OmniVoiceEngine,
    ensure_ffmpeg_on_path,
    ensure_decodable,
    content_digest,
    split_paragraphs,
    AUDIO_EXTS,
    MODEL_DIR,
    VOICES_DIR,
    SAMPLES_DIR,
    PROMPT_CACHE_DIR,
    VOICE_CACHE_DIR,
    PROJECT_ROOT,
)

__all__ = [
    "ENGINE",
    "GPUMemoryError",
    "OmniVoiceEngine",
    "ensure_ffmpeg_on_path",
    "ensure_decodable",
    "content_digest",
    "split_paragraphs",
    "AUDIO_EXTS",
    "MODEL_DIR",
    "VOICES_DIR",
    "SAMPLES_DIR",
    "PROMPT_CACHE_DIR",
    "VOICE_CACHE_DIR",
    "PROJECT_ROOT",
]
