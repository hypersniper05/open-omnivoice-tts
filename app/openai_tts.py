#!/usr/bin/env python3
"""OpenAI-compatible Text-to-Speech layer for OmniVoice.

Exposes the project's OmniVoice engine through the same request/response shape
as OpenAI's audio API, so any LLM front end that speaks the OpenAI TTS protocol
(Open WebUI, LibreChat, SillyTavern, AnythingLLM, the ``openai`` Python/JS SDKs,
...) can point its ``base_url`` at this machine.

Wire protocol (subset of https://platform.openai.com/docs/api-reference/audio):

    POST /v1/audio/speech
        {
          "model": "tts-1",            # accepted; used only as a label
          "input": "Text to speak.",   # required
          "voice": "alloy",            # alias, a Voices/ file, or "design:<instruct>"
          "response_format": "mp3",    # mp3|opus|aac|flac|wav|pcm  (default mp3)
          "speed": 1.0,                # 0.25..4.0  (default from config)
          "instructions": "..."        # optional voice-steering hint
        }
      -> binary audio body with the matching Content-Type

    GET  /v1/models          -> OpenAI-style model list
    GET  /v1/audio/voices    -> alias map + available reference voices (helper)

Voice resolution order for the ``voice`` field:
    1. an alias in ``openai_voices.json``   ("alloy" -> "narr_new.wav")
    2. a reference file in ``../Voices``     ("Erik.wav" or just "Erik")
    3. a ``design:<instruct>`` string         ("design:female, young adult, british accent")
    4. otherwise -> the configured default voice

The alias map and the default generation parameters live in
``app/openai_voices.json`` (written on first run) so the voice line-up and the
quality/latency trade-off can be tuned without editing code.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root for `core`
from core import ENGINE, VOICES_DIR, AUDIO_EXTS, ensure_ffmpeg_on_path
from core.omnivoice_engine import VOICE_DESIGN_CATEGORIES

import custom_voices

logger = logging.getLogger("omnivoice.openai")

CONFIG_PATH = Path(__file__).resolve().parent / "openai_voices.json"

# Output formats we can emit and the Content-Type each maps to.
FORMAT_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",   # raw 16-bit signed LE @ engine sample rate (24 kHz)
}

# Model ids advertised on /v1/models so a front end's model dropdown is happy.
ADVERTISED_MODELS = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts", "gpt-4o-mini-tts-2025-12-15", "omnivoice"]

# Generation kwargs OmniVoiceEngine.generate() accepts (config is filtered to
# this set so a hand-edited config can't crash a request with a stray key).
_GEN_KEYS = {
    "num_step", "guidance_scale", "t_shift", "duration", "denoise",
    "preprocess_prompt", "postprocess_output", "position_temperature",
    "class_temperature", "layer_penalty_factor", "chunk_paragraphs",
    "para_gap", "language",
    # Added upstream in 0.2.x; without them here a hand-edited config would be
    # silently dropped by this filter rather than reaching the engine.
    "pad_duration", "fade_duration", "normalize_text",
}

# Defaults: the tracked app/openai_voices.example.json. The start scripts copy it
# to openai_voices.json (local, not in git), the file this module reads; a server
# started without that copy uses the example as it is.
EXAMPLE_PATH = CONFIG_PATH.with_name("openai_voices.example.json")
try:
    DEFAULT_CONFIG = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
except (OSError, ValueError):
    DEFAULT_CONFIG = {"api_key": None, "allow_custom_voices": True, "default_voice": "alloy",
                      "default_format": "mp3", "default_speed": 1.0, "normalize": True,
                      "loudness_target_dbfs": -20.0, "voices": {}, "generation": {"num_step": 16}}


class TTSRequestError(Exception):
    """Client/server error carrying an HTTP status + OpenAI-style code."""

    def __init__(self, message: str, status: int = 400, code: str = "invalid_request_error"):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


# -- config -----------------------------------------------------------------
_cfg_cache: dict = {"mtime": None, "data": None}


def _load_config() -> dict:
    """Load (and on first run create) ``openai_voices.json``, cached by mtime."""
    try:
        mtime = CONFIG_PATH.stat().st_mtime_ns if CONFIG_PATH.is_file() else None
    except Exception:
        mtime = None
    if _cfg_cache["data"] is not None and _cfg_cache["mtime"] == mtime:
        return _cfg_cache["data"]

    if CONFIG_PATH.is_file():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for k, v in DEFAULT_CONFIG.items():       # forward-compat defaults
                cfg.setdefault(k, v)
        except Exception:
            logger.warning("Could not parse %s; using built-in defaults", CONFIG_PATH)
            cfg = dict(DEFAULT_CONFIG)
    else:
        cfg = dict(DEFAULT_CONFIG)
        try:
            CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
            logger.info("Created %s from %s", CONFIG_PATH.name, EXAMPLE_PATH.name)
        except Exception:
            pass

    try:
        mtime = CONFIG_PATH.stat().st_mtime_ns if CONFIG_PATH.is_file() else None
    except Exception:
        pass
    _cfg_cache.update(mtime=mtime, data=cfg)
    return cfg


def error_response(message: str, status: int = 400, code: str = "invalid_request_error") -> dict:
    """OpenAI-style error envelope."""
    return {"error": {"message": message, "type": code, "code": code, "param": None}}


def check_auth(auth_header: Optional[str]) -> bool:
    """True if the request may proceed. Open unless ``api_key`` is configured."""
    key = _load_config().get("api_key")
    if not key:
        return True
    if not auth_header:
        return False
    token = auth_header.split(" ", 1)[1] if auth_header.lower().startswith("bearer ") else auth_header
    return token.strip() == str(key)


# -- voice resolution -------------------------------------------------------
def _reference_lookup() -> dict[str, str]:
    """Map lowercased filename and stem -> the real filename in Voices/."""
    lut: dict[str, str] = {}
    try:
        for p in sorted(VOICES_DIR.iterdir(), key=lambda x: x.name.lower()):
            if p.suffix.lower() in AUDIO_EXTS:
                lut[p.name.lower()] = p.name
                stem = p.stem.lower()
                if stem not in lut or p.suffix.lower() == ".wav":   # prefer .wav for a bare stem
                    lut[stem] = p.name
    except Exception:
        pass
    return lut


def resolve_voice(name, cfg: dict) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve a requested voice to ``(mode, voice_file, instruct)``.

    ``name`` is a string or, as in OpenAI's API since custom voices, an object
    ``{"id": "..."}``; both are accepted for every kind of voice."""
    if isinstance(name, dict):
        name = name.get("id")
    if name is not None and not isinstance(name, str):
        raise TTSRequestError("'voice' must be a string or an object with an 'id'.")
    voices = {k.lower(): v for k, v in (cfg.get("voices") or {}).items()}
    default = cfg.get("default_voice") or "alloy"
    name = (name or default).strip()
    lut = _reference_lookup()

    # 0. a custom voice created through POST /v1/audio/voices
    if name.lower().startswith("voice_"):
        custom = custom_voices.voice_file(name)
        if custom:
            return "clone", custom, None
        raise TTSRequestError(f"No custom voice with id {name!r}.", status=404, code="voice_not_found")

    def as_design(spec: str):
        instruct = spec.split(":", 1)[1].strip()
        if not instruct:
            raise TTSRequestError("A 'design:' voice needs an instruct string after the colon.")
        return "design", None, instruct

    # 3. explicit voice-design request
    if name.lower().startswith("design:"):
        return as_design(name)

    # 1. alias -> a reference file or a design: string
    mapped = voices.get(name.lower())
    if mapped is not None:
        if str(mapped).lower().startswith("design:"):
            return as_design(mapped)
        key, stem = mapped.lower(), Path(mapped).stem.lower()
        if key in lut:
            return "clone", lut[key], None
        if stem in lut:
            return "clone", lut[stem], None
        raise TTSRequestError(
            f"Voice alias {name!r} maps to {mapped!r}, which is not present in Voices/.",
            status=500, code="server_error")

    # 2. a raw reference file (full name or stem)
    if name.lower() in lut:
        return "clone", lut[name.lower()], None

    # 4. fall back to the default alias
    fb = voices.get(default.lower())
    if fb and not str(fb).lower().startswith("design:") and fb.lower() in lut:
        logger.warning("Unknown voice %r; falling back to default %r (%s)", name, default, fb)
        return "clone", lut[fb.lower()], None

    raise TTSRequestError(
        f"Unknown voice {name!r}. Use an alias ({', '.join(sorted(voices)) or 'none configured'}), "
        f"a file in Voices/, or 'design:<instruct>'."
    )


# -- audio encoding ---------------------------------------------------------
def encode_audio(audio: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    """Encode a float32 mono waveform to the requested format -> (bytes, ctype)."""
    fmt = (fmt or "mp3").lower()
    if fmt not in FORMAT_CONTENT_TYPES:
        raise TTSRequestError(
            f"Unsupported response_format {fmt!r}. Choose one of: {', '.join(FORMAT_CONTENT_TYPES)}."
        )
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    if fmt in ("wav", "flac"):
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, int(sr), format=fmt.upper())   # libsndfile defaults to PCM_16
        return buf.getvalue(), FORMAT_CONTENT_TYPES[fmt]

    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    if fmt == "pcm":
        return pcm16.tobytes(), FORMAT_CONTENT_TYPES[fmt]

    # mp3 / opus / aac via ffmpeg (pydub). Make sure ffmpeg is reachable even if
    # the engine hasn't been loaded yet (it prepends ffmpeg to PATH on load()).
    ensure_ffmpeg_on_path()
    from pydub import AudioSegment
    seg = AudioSegment(data=pcm16.tobytes(), sample_width=2, frame_rate=int(sr), channels=1)
    export_args = {
        "mp3": dict(format="mp3", bitrate="128k"),
        "opus": dict(format="opus", codec="libopus", parameters=["-b:a", "64k"]),
        "aac": dict(format="adts", codec="aac", parameters=["-b:a", "160k"]),
    }[fmt]
    buf = io.BytesIO()
    try:
        seg.export(buf, **export_args)
    except Exception as e:
        raise TTSRequestError(
            f"Failed to encode {fmt!r} (is ffmpeg available?): {e}", status=500, code="server_error"
        )
    return buf.getvalue(), FORMAT_CONTENT_TYPES[fmt]


# -- loudness normalization -------------------------------------------------
def normalize_loudness(audio: np.ndarray, sr: int, target_dbfs: float = -20.0,
                       peak_ceiling: float = 0.97, max_drift_db: float = 4.0,
                       frame_s: float = 0.4, smooth_s: float = 2.0) -> np.ndarray:
    """Even out loudness so every voice plays at a consistent, clip-safe level.

    OmniVoice output inherits the loudness of the reference clip, so voices range
    from very quiet to full-scale clipping. This: (1) normalizes overall level to
    ``target_dbfs`` RMS, (2) gently flattens slow drift / a hot onset by allowing
    only +/- ``max_drift_db`` of smoothed local correction (so it stays natural,
    no pumping), and (3) hard-limits the peak to ``peak_ceiling`` to prevent
    clipping.
    """
    a = np.ascontiguousarray(audio, dtype=np.float32)
    if a.size == 0:
        return a
    g_rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    if g_rms < 1e-6:
        return a                                  # silence — leave as-is
    target = 10.0 ** (target_dbfs / 20.0)
    fw = max(1, int(sr * frame_s))
    nf = a.size // fw
    if nf < 3:                                     # too short for envelope work
        out = a * (target / g_rms)
    else:
        frames = a[: nf * fw].reshape(nf, fw).astype(np.float64)
        frms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9
        g_global = target / g_rms
        lo, hi = g_global * 10 ** (-max_drift_db / 20), g_global * 10 ** (max_drift_db / 20)
        gf = np.clip(target / frms, lo, hi)
        # Clamp the smoothing window to the frame count. np.convolve(..., "same")
        # returns max(len(gf), sm), so a window wider than the signal makes gf
        # longer than `centers` and np.interp raises "fp and xp are not of the
        # same length". With the defaults (frame_s=0.4, smooth_s=2.0 -> sm=5) that
        # hit every clip of roughly 1.2-2.0 s (nf of 3 or 4) with a 500; the
        # nf < 3 guard above is not enough on its own.
        sm = max(1, min(nf, int(round(smooth_s / frame_s))))
        gf = np.convolve(gf, np.ones(sm) / sm, mode="same")
        centers = (np.arange(nf) + 0.5) * fw
        gain = np.interp(np.arange(a.size), centers, gf, left=gf[0], right=gf[-1]).astype(np.float32)
        out = a * gain
    peak = float(np.max(np.abs(out)))
    if peak > peak_ceiling:
        out = out * (peak_ceiling / peak)
    return out.astype(np.float32)


# -- main entry points ------------------------------------------------------
def synthesize_speech(payload: dict) -> tuple[bytes, str]:
    """Handle a POST /v1/audio/speech body -> (audio_bytes, content_type)."""
    cfg = _load_config()

    if not ENGINE.ready:
        raise TTSRequestError(
            ENGINE.error or "Model is still loading; please retry shortly.",
            status=503, code="model_not_ready",
        )

    text = payload.get("input", payload.get("text", ""))
    if not isinstance(text, str) or not text.strip():
        raise TTSRequestError("Missing required field 'input'.")

    fmt = payload.get("response_format") or cfg.get("default_format") or "mp3"
    mode, voice_file, instruct = resolve_voice(payload.get("voice"), cfg)

    # OpenAI's free-text 'instructions' -> an OmniVoice design hint, but only in
    # clone/auto mode. In design mode the voice string IS the instruct, and
    # OmniVoice validates it against a constrained vocabulary (gender/age/pitch/
    # style/accent), so appending arbitrary prose would make it reject the whole
    # request. Free-text hints that aren't valid vocab are dropped below.
    instructions = payload.get("instructions")
    if mode != "design" and isinstance(instructions, str) and instructions.strip():
        instruct = instructions.strip()

    speed = payload.get("speed", cfg.get("default_speed", 1.0))
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        speed = 1.0
    speed = max(0.25, min(4.0, speed))

    gen = {k: v for k, v in (cfg.get("generation") or {}).items() if k in _GEN_KEYS}

    def _gen(instr):
        return ENGINE.generate(
            text=text, mode=mode, voice=voice_file, instruct=instr,
            speed=speed, save=False, **gen,
        )

    try:
        result = _gen(instruct)
    except ValueError as e:
        # OmniVoice raises ValueError for unsupported instruct vocabulary.
        if "instruct" in str(e).lower():
            if mode != "design" and instruct:
                logger.warning("Ignoring unsupported instruct hint %r (%s)", instruct, e)
                result = _gen(None)            # graceful: clone/auto still works
            else:
                raise TTSRequestError(f"Invalid voice-design attributes: {e}", status=400)
        else:
            raise

    audio = result["audio"]
    if cfg.get("normalize", True):
        audio = normalize_loudness(audio, result["sampling_rate"],
                                   target_dbfs=float(cfg.get("loudness_target_dbfs", -20.0)))
    return encode_audio(audio, result["sampling_rate"], fmt)


def list_models() -> dict:
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": created, "owned_by": "omnivoice"}
            for m in ADVERTISED_MODELS
        ],
    }


def model_object(model_id: str) -> dict:
    return {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "omnivoice"}


def list_voices() -> dict:
    cfg = _load_config()
    lut = _reference_lookup()
    aliases = [{"name": k, "maps_to": v} for k, v in (cfg.get("voices") or {}).items()]
    return {
        "aliases": aliases,
        "reference_files": sorted(set(lut.values())),
        "formats": list(FORMAT_CONTENT_TYPES),
        "default_voice": cfg.get("default_voice"),
        "default_format": cfg.get("default_format"),
        "custom_voices": custom_voices.list_voices(),
    }


def alias_names() -> list[str]:
    """The voice names configured in openai_voices.json (for the API docs)."""
    return list((_load_config().get("voices") or {}).keys())


# -- custom voices (POST /v1/audio/voices) -------------------------------------
def custom_voices_allowed() -> bool:
    return bool(_load_config().get("allow_custom_voices"))


def design_synth(text: str, instruct: str) -> tuple[np.ndarray, int]:
    """Speak ``text`` with an OmniVoice voice-design ``instruct`` (a prompt voice's
    reference clip). Uses the API's generation profile and loudness target."""
    cfg = _load_config()
    if not ENGINE.ready:
        raise TTSRequestError(ENGINE.error or "Model is still loading; please retry shortly.",
                              status=503, code="model_not_ready")
    gen = {k: v for k, v in (cfg.get("generation") or {}).items() if k in _GEN_KEYS}
    try:
        result = ENGINE.generate(text=text, mode="design", instruct=instruct, save=False, **gen)
    except ValueError as e:
        if "instruct" not in str(e).lower():
            raise
        vocab = "; ".join(f"{k}: {', '.join(v)}" for k, v in VOICE_DESIGN_CATEGORIES.items())
        raise TTSRequestError(
            f"OmniVoice builds voices from a comma-separated list of attributes, one per category "
            f"({vocab}). For example: 'female, middle-aged, low pitch, british accent'. ({e})")
    audio = result["audio"]
    if cfg.get("normalize", True):
        audio = normalize_loudness(audio, result["sampling_rate"],
                                   target_dbfs=float(cfg.get("loudness_target_dbfs", -20.0)))
    return audio, result["sampling_rate"]
