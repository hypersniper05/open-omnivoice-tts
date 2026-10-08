#!/usr/bin/env python3
"""OpenAI custom voices on top of OmniVoice cloning.

Implements OpenAI's voice-creation endpoints (openai-openapi, October 2026):

    POST   /v1/audio/voice_consents          multipart: name, recording, language
    GET    /v1/audio/voice_consents          ?after=<id>&limit=<n>  -> list
    GET    /v1/audio/voice_consents/{id}
    POST   /v1/audio/voice_consents/{id}     JSON: {"name": ...}
    DELETE /v1/audio/voice_consents/{id}
    POST   /v1/audio/voices                  multipart: name, audio_sample, consent    (type "audio_sample")
                                             JSON or multipart: type "prompt", name, prompt, script_hint?
    DELETE /v1/audio/voices/{id}             extension: OpenAI has no delete for voices

A created voice is used as in OpenAI: ``"voice": {"id": "voice_..."}``, or the id
as a plain string.

Storage, in Voices/custom/ (writable; keep it out of git):
    index.json                 voices + consents metadata
    voice_<id>.wav             reference clip: the sample, converted to 24 kHz mono
                               and padded 0.5 s / 0.3 s with silence (README:
                               "Breath or blip at the start of the output")
    voice_<id>.txt             transcript of a prompt voice's clip (the script it speaks);
                               sample voices have none, so Whisper transcribes them once
    consents/cons_<id>.<ext>   consent recordings, stored as uploaded
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import tempfile
import threading
import time
from email.parser import BytesParser
from email.policy import HTTP
from pathlib import Path
from typing import Callable, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root for `core`
from core import VOICES_DIR, ensure_ffmpeg_on_path

CUSTOM_DIR = VOICES_DIR / "custom"
CONSENT_DIR = CUSTOM_DIR / "consents"
INDEX_PATH = CUSTOM_DIR / "index.json"

MAX_FILE_BYTES = 10 * 1024 * 1024        # OpenAI's limit for audio_sample and recording
LEAD_S, TAIL_S = 0.5, 0.3                # silence added around every reference clip
MIN_SAMPLE_S = 1.0
MIN_SCRIPT_CHARS = 20
DEFAULT_SCRIPT = ("Hello! This is my new voice. I can read articles, stories and messages aloud, "
                  "and I will sound the same every time you use me.")
AUDIO_MIME = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/ogg", "audio/aac",
              "audio/flac", "audio/webm", "audio/mp4"}
LANG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})*$")   # BCP 47, loosely

_lock = threading.Lock()


class CustomVoiceError(Exception):
    """Client/server error carrying an HTTP status + OpenAI-style code."""

    def __init__(self, message: str, status: int = 400, code: str = "invalid_request_error"):
        super().__init__(message)
        self.message, self.status, self.code = message, status, code


# -- index ------------------------------------------------------------------
def _load() -> dict:
    try:
        idx = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        idx = {}
    idx.setdefault("voices", {})
    idx.setdefault("consents", {})
    return idx


def _save(idx: dict) -> None:
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, INDEX_PATH)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


def voice_obj(vid: str, v: dict) -> dict:
    return {"object": "audio.voice", "id": vid, "type": v["type"], "name": v["name"],
            "created_at": v["created_at"]}


def consent_obj(cid: str, c: dict) -> dict:
    return {"object": "audio.voice_consent", "id": cid, "name": c["name"],
            "language": c["language"], "created_at": c["created_at"]}


# -- request parsing --------------------------------------------------------
def parse_multipart(content_type: str, body: bytes) -> tuple[dict, dict]:
    """multipart/form-data -> ({field: str}, {field: (filename, content_type, bytes)})."""
    msg = BytesParser(policy=HTTP).parsebytes(
        b"MIME-Version: 1.0\r\nContent-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise CustomVoiceError("Expected a multipart/form-data body.")
    fields, files = {}, {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        data = part.get_payload(decode=True) or b""
        if part.get_filename() is not None:
            files[name] = (part.get_filename(), part.get_content_type(), data)
        else:
            fields[name] = data.decode("utf-8", "replace").strip()
    return fields, files


def _required(fields: dict, *names: str) -> None:
    for n in names:
        if not (fields.get(n) or "").strip():
            raise CustomVoiceError(f"Missing required parameter: '{n}'.")


def _check_file(files: dict, field: str) -> tuple[str, bytes]:
    if field not in files:
        raise CustomVoiceError(f"Missing required file: '{field}'.")
    filename, ctype, data = files[field]
    if not data:
        raise CustomVoiceError(f"'{field}' is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise CustomVoiceError(f"'{field}' is {len(data) / 2**20:.1f} MiB; the maximum is 10 MiB.", 413)
    ext = Path(filename or "").suffix.lower()
    if ctype not in AUDIO_MIME and ext not in {".mp3", ".wav", ".ogg", ".oga", ".opus", ".aac", ".m4a",
                                               ".mp4", ".flac", ".webm"}:
        raise CustomVoiceError(
            f"Unsupported audio type {ctype!r} for '{field}'. Supported: {', '.join(sorted(AUDIO_MIME))}.")
    return ext or ".bin", data


# -- audio ------------------------------------------------------------------
def _ffmpeg() -> str:
    exe = ensure_ffmpeg_on_path()
    if not exe:
        raise CustomVoiceError("ffmpeg is not available on the server.", 500, "server_error")
    return exe


def _decode_to_padded_wav(data: bytes, ext: str, out: Path) -> float:
    """Decode any supported upload to 24 kHz mono 16-bit WAV with the standard padding.
    Returns the duration of the audio before padding."""
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"in{ext}"
        src.write_bytes(data)
        raw = Path(td) / "raw.wav"
        r = subprocess.run([_ffmpeg(), "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "24000",
                            "-sample_fmt", "s16", str(raw)], capture_output=True, text=True)
        if r.returncode != 0 or not raw.is_file():
            last = ([ln for ln in (r.stderr or "").splitlines() if ln.strip()] or ["unknown error"])[-1]
            raise CustomVoiceError(f"Could not decode the audio file ({last.strip()[:160]}).")
        audio, sr = sf.read(raw, dtype="int16")
    seconds = len(audio) / sr
    if seconds < MIN_SAMPLE_S:
        raise CustomVoiceError(f"The audio sample is {seconds:.1f} s; it must be at least {MIN_SAMPLE_S:.0f} s.")
    _write_padded(audio, sr, out)
    return seconds


def _write_padded(audio, sr: int, out: Path) -> None:
    import numpy as np
    import soundfile as sf
    a = np.asarray(audio)
    if a.dtype != np.int16:
        a = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)
    pad = lambda s: np.zeros(int(s * sr), dtype=np.int16)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, np.concatenate([pad(LEAD_S), a.reshape(-1), pad(TAIL_S)]), sr, subtype="PCM_16")


# -- consents ---------------------------------------------------------------
def create_consent(fields: dict, files: dict) -> dict:
    _required(fields, "name", "language")
    if not LANG_RE.match(fields["language"]):
        raise CustomVoiceError(f"'language' must be a BCP 47 tag such as en-US, not {fields['language']!r}.")
    ext, data = _check_file(files, "recording")
    with tempfile.TemporaryDirectory() as td:          # prove it is audio before keeping it
        _decode_to_padded_wav(data, ext, Path(td) / "probe.wav")
    cid = _new_id("cons")
    CONSENT_DIR.mkdir(parents=True, exist_ok=True)
    (CONSENT_DIR / f"{cid}{ext}").write_bytes(data)
    with _lock:
        idx = _load()
        idx["consents"][cid] = {"name": fields["name"], "language": fields["language"],
                                "created_at": int(time.time()), "file": f"consents/{cid}{ext}"}
        _save(idx)
        return consent_obj(cid, idx["consents"][cid])


def list_consents(after: Optional[str] = None, limit: Optional[str] = None) -> dict:
    try:
        n = max(1, min(100, int(limit or 20)))
    except ValueError:
        raise CustomVoiceError("'limit' must be an integer.")
    items = sorted(_load()["consents"].items(), key=lambda kv: (kv[1]["created_at"], kv[0]), reverse=True)
    if after:
        ids = [k for k, _ in items]
        if after not in ids:
            raise CustomVoiceError(f"Unknown 'after' cursor {after!r}.")
        items = items[ids.index(after) + 1:]
    page = [consent_obj(k, v) for k, v in items[:n]]
    return {"object": "list", "data": page, "first_id": page[0]["id"] if page else None,
            "last_id": page[-1]["id"] if page else None, "has_more": len(items) > n}


def get_consent(cid: str) -> dict:
    c = _load()["consents"].get(cid)
    if not c:
        raise CustomVoiceError(f"No voice consent with id {cid!r}.", 404, "not_found")
    return consent_obj(cid, c)


def update_consent(cid: str, body: dict) -> dict:
    name = (body.get("name") or "").strip() if isinstance(body.get("name"), str) else ""
    if not name:
        raise CustomVoiceError("Missing required parameter: 'name'.")
    with _lock:
        idx = _load()
        if cid not in idx["consents"]:
            raise CustomVoiceError(f"No voice consent with id {cid!r}.", 404, "not_found")
        idx["consents"][cid]["name"] = name
        _save(idx)
        return consent_obj(cid, idx["consents"][cid])


def delete_consent(cid: str) -> dict:
    with _lock:
        idx = _load()
        c = idx["consents"].pop(cid, None)
        if not c:
            raise CustomVoiceError(f"No voice consent with id {cid!r}.", 404, "not_found")
        (CUSTOM_DIR / c["file"]).unlink(missing_ok=True)
        _save(idx)
    return {"id": cid, "object": "audio.voice_consent", "deleted": True}


# -- voices -----------------------------------------------------------------
def create_voice_from_sample(fields: dict, files: dict) -> dict:
    _required(fields, "name", "consent")
    if fields["consent"] not in _load()["consents"]:
        raise CustomVoiceError(f"No voice consent with id {fields['consent']!r}. "
                               "Create one with POST /v1/audio/voice_consents first.", 400, "invalid_consent")
    ext, data = _check_file(files, "audio_sample")
    vid = _new_id("voice")
    seconds = _decode_to_padded_wav(data, ext, CUSTOM_DIR / f"{vid}.wav")
    with _lock:
        idx = _load()
        idx["voices"][vid] = {"type": "audio_sample", "name": fields["name"], "created_at": int(time.time()),
                              "consent": fields["consent"], "sample_seconds": round(seconds, 2)}
        _save(idx)
        return voice_obj(vid, idx["voices"][vid])


def create_voice_from_prompt(fields: dict, synth: Callable[[str, str], tuple]) -> dict:
    """``synth(text, instruct) -> (float32 audio, sample_rate)`` runs OmniVoice voice design."""
    _required(fields, "name", "prompt")
    script = (fields.get("script_hint") or "").strip() or DEFAULT_SCRIPT
    if len(script) < MIN_SCRIPT_CHARS:
        raise CustomVoiceError(f"'script_hint' is too short; use at least {MIN_SCRIPT_CHARS} characters.")
    audio, sr = synth(script, fields["prompt"].strip())
    vid = _new_id("voice")
    _write_padded(audio, sr, CUSTOM_DIR / f"{vid}.wav")
    (CUSTOM_DIR / f"{vid}.txt").write_text(script, encoding="utf-8")   # exact transcript, no Whisper
    with _lock:
        idx = _load()
        idx["voices"][vid] = {"type": "prompt", "name": fields["name"], "created_at": int(time.time()),
                              "prompt": fields["prompt"].strip(), "script": script}
        _save(idx)
        return voice_obj(vid, idx["voices"][vid])


def delete_voice(vid: str) -> dict:
    with _lock:
        idx = _load()
        if not idx["voices"].pop(vid, None):
            raise CustomVoiceError(f"No custom voice with id {vid!r}.", 404, "not_found")
        for suffix in (".wav", ".txt"):
            (CUSTOM_DIR / f"{vid}{suffix}").unlink(missing_ok=True)
        _save(idx)
    return {"id": vid, "object": "audio.voice", "deleted": True}


def voice_file(vid: str) -> Optional[str]:
    """Voices/-relative path of a custom voice's reference clip, or None."""
    if vid in _load()["voices"] and (CUSTOM_DIR / f"{vid}.wav").is_file():
        return f"custom/{vid}.wav"
    return None


def list_voices() -> list[dict]:
    return [voice_obj(k, v) for k, v in sorted(_load()["voices"].items(), key=lambda kv: kv[1]["created_at"])]
