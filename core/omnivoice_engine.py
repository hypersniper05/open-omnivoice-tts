#!/usr/bin/env python3
"""OmniVoice inference engine wrapper.

A thin, dependency-light layer over the upstream ``omnivoice.OmniVoice`` model
that the API server (``app/server.py``) uses.

Responsibilities
----------------
* Load the model once from the locally-cloned weights in ``../model``.
* Expose UI metadata (reference voices, languages, voice-design vocabulary,
  non-verbal tags, parameter defaults).
* Cache ``VoiceClonePrompt`` objects so that iterating on text / parameters
  with the same reference voice is fast (the expensive audio tokenisation +
  optional ASR only happens once per voice).
* Run generation under a lock (the model is not safe for concurrent calls).

The engine is intentionally framework-free: no Flask, no Gradio. ``server.py``
talks to it directly and serves a plain HTML page.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("omnivoice.engine")

# --------------------------------------------------------------------------
# GPU placement
# --------------------------------------------------------------------------
# OMNIVOICE_GPU picks the card: an nvidia-smi index ("1") or a UUID. It is applied
# before CUDA initialises -- every entry point imports this module first -- and
# with PCI bus order, so the index means the same card as in nvidia-smi (torch
# would otherwise number cards fastest-first). Unset = CUDA's default, GPU 0.
# A caller that already set CUDA_VISIBLE_DEVICES (the container, or "-1" for
# CPU) is left alone. OMNIVOICE_GPU_UUID is the older name and still works.
_GPU = (os.environ.get("OMNIVOICE_GPU") or os.environ.get("OMNIVOICE_GPU_UUID") or "").strip()
if _GPU:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", _GPU)

# Whisper (auto-transcription of a reference voice with no ref_text) runs on the
# CPU: on the GPU it would add ~3.7 GB next to the TTS model (measured on an RTX
# 3080). It only runs once per new voice (prompts are cached).
ASR_DEVICE = os.environ.get("OMNIVOICE_ASR_DEVICE") or "cpu"
# On the CPU, faster-whisper (CTranslate2 int8) is used when installed (the
# container has it): measured on the i9-14900K, 6.5-7.9 s per reference vs
# 9.5-12 s for the transformers pipeline in fp32, with the same transcript and
# ~1 GB RAM instead of ~3.2 GB. Falls back to transformers (e.g. the host .venv).
ASR_MODEL = os.environ.get("OMNIVOICE_ASR_MODEL") or "large-v3-turbo"
# torch.compile of the Qwen3 backbone: "" = off, "1"/"default", or a torch
# compile mode such as "reduce-overhead".
COMPILE = os.environ.get("OMNIVOICE_COMPILE", "").strip()
# Upstream's FlashInfer path (omnivoice_flashinfer.apply_flashinfer, needs the
# flashinfer package and the image's AOT kernels): "" = off, "1" = packed ragged attention,
# "graphs" = plus a CUDA graph: each generation captures one graph for its exact shape
# and replays it for every step, then drops it (one generation = one graph in VRAM).
# Measured on a 5090: sentence 0.20 s -> 0.13-0.16 s with the capture, paragraph 0.64 s
# -> 0.43 s. OMNIVOICE_FI_BUCKETS (e.g. "4,6,8,10,15,20,30") switches to upstream's
# bucket mode instead: fixed padded shapes, graphs kept (OMNIVOICE_FI_OVERHEAD tokens of
# room for text and reference); with large buckets it was slower than no graphs.
FLASHINFER = os.environ.get("OMNIVOICE_FLASHINFER", "").strip()
FI_BUCKETS = [int(x) for x in (os.environ.get("OMNIVOICE_FI_BUCKETS") or "").split(",") if x.strip()]
FI_OVERHEAD = int(os.environ.get("OMNIVOICE_FI_OVERHEAD") or 512)
# Hard cap on torch's allocator on the GPU, for a card shared with other work:
# a fraction of the card (OMNIVOICE_MEM_FRACTION, e.g. 0.55 = 5,632 MiB of a
# 10 GB card) or an absolute OMNIVOICE_VRAM_CAP_MIB. Unset = no cap. Past the cap a
# request fails here with an OOM, which generate() recovers (503), instead of
# pushing the card over. The CUDA context (~0.3-0.5 GB) is outside the cap.
MEM_FRACTION = float(os.environ.get("OMNIVOICE_MEM_FRACTION") or 0)
VRAM_CAP_MIB = float(os.environ.get("OMNIVOICE_VRAM_CAP_MIB") or 0)
ASR_THREADS = int(os.environ.get("OMNIVOICE_ASR_THREADS") or 8)
# Model precision: "" = automatic (float16 on GPUs with tensor cores, float32 on
# older GPUs, the GTX 16-series and the CPU), or float16 / float32 / bfloat16.
# float32 takes twice the VRAM. FlashInfer needs float16 and is skipped with any
# other precision.
DTYPE = (os.environ.get("OMNIVOICE_DTYPE") or "").strip().lower()


from core.prompt_worker import FasterWhisperASR as _FasterWhisperASR  # noqa: E402

# Build new voices' prompts (trim + ASR + CPU encode) in a separate 'spawn'
# process: "1" always, "0" never, "auto" (default) only when FlashInfer is on --
# with it loaded, doing that work in-process segfaulted (see core/prompt_worker.py).
PROMPT_WORKER = (os.environ.get("OMNIVOICE_PROMPT_WORKER") or "auto").strip().lower()


def _trim_host_heap() -> None:
    """Return freed heap pages to the OS (glibc only; a no-op elsewhere). The
    CPU reference encode allocates GBs of fp32 activations for a few seconds;
    glibc keeps them in its arenas afterwards, so without this the container's
    RAM stays near the 10.8 GB peak measured while pre-building prompts."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class GPUMemoryError(RuntimeError):
    """A generation hit CUDA OOM (normally the memory cap); recovered."""


def _is_oom(e: BaseException) -> bool:
    """torch.OutOfMemoryError, or a RuntimeError that is an OOM in disguise:
    under WSL2 the driver can surface one as "device not ready", and on a full
    4 GB card expandable segments failed with "!handles_.at(i) INTERNAL ASSERT
    FAILED ... CUDACachingAllocator.cpp" (the next request worked again)."""
    if type(e).__name__ == "OutOfMemoryError":
        return True
    msg = str(e).lower()
    return isinstance(e, RuntimeError) and (
        "out of memory" in msg or "device not ready" in msg
        or ("internal assert failed" in msg and "cudacachingallocator" in msg))


def _no_tensor_cores(name: str, capability: tuple[int, int]) -> bool:
    """GPUs without tensor cores: everything before Volta (7.0), and the Turing
    parts that report 7.5 like the RTX 20-series but lack them (GTX 16-series,
    MX 450/550, T400-T2000). Libraries take tensor-core paths for float16 there,
    which run slowly: on a GTX 1650 a sentence took 44.6 s in float16 and 16.3 s
    in float32, with the card at 20 of 75 W."""
    if capability < (7, 0):
        return True
    return capability == (7, 5) and bool(re.search(r"GTX 16|\bMX ?[45]\d0\b|\bT(400|500|550|600|1000|1200|2000)\b", name))


def _drop_graphs_after_each_generation(model) -> None:
    """Exact-shape CUDA graphs (upstream apply_flashinfer(enable_cuda_graph=True)) are
    cached per input shape and never freed: each kept its own memory pool and a 64 MB
    attention workspace, +1.1 GB after a dozen texts on a 5090. Nearly all of the gain
    is inside one generation (one capture, then a replay per step), so drop the cache
    when each _generate_iterative call returns: one graph at a time, however long the
    text."""
    run = model._generate_iterative

    def generate_then_drop(*args, **kwargs):
        try:
            return run(*args, **kwargs)
        finally:
            model._fi_graph_cache.clear()

    model._generate_iterative = generate_then_drop


def _sentence_units(text: str) -> Optional[str]:
    """The text re-split one sentence per paragraph (each paragraph is its own
    generation unit), for the OOM retry. None if it would not split further."""
    sents = [s for s in re.split(r"(?<=[.!?;:])\s+", " ".join((text or "").split())) if s]
    return "\n\n".join(sents) if len(sents) > 1 else None


def _refuse_second_instance() -> None:
    """One OmniVoice instance per GPU. A host-side run (run_tts_api.ps1) must
    not load a second copy of the model while the omnivoice-tts container runs
    or another OmniVoice server answers on :8008. OMNIVOICE_ALLOW_SECOND=1
    overrides (only when you know the card has room)."""
    if os.path.exists("/.dockerenv") or os.environ.get("OMNIVOICE_ALLOW_SECOND") == "1":
        return  # this IS the container, or explicitly allowed
    why = None
    try:
        running = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", "omnivoice-tts"],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
        if running == "true":
            why = "The omnivoice-tts container is running"
    except Exception:
        pass  # no docker CLI: fall through to the port check
    if why is None and os.environ.get("OMNIVOICE_SELF_PORT") != "8008":  # not ourselves
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:8008/api/meta", timeout=3):
                why = "An OmniVoice server already answers on http://127.0.0.1:8008"
        except Exception:
            pass
    if why:
        raise RuntimeError(
            f"{why}, and only one OmniVoice instance should use a GPU. "
            "Use its API at http://127.0.0.1:8008, stop it first (docker stop omnivoice-tts), "
            "run with --device cpu, or set OMNIVOICE_ALLOW_SECOND=1 if you are sure the card has room.")


def _make_cpu_asr():
    """faster-whisper on the CPU if available, else None (caller falls back to
    OmniVoice's own transformers pipeline on ASR_DEVICE)."""
    if ASR_DEVICE != "cpu":
        return None
    try:
        t0 = time.time()
        asr = _FasterWhisperASR(ASR_MODEL, ASR_THREADS)
        logger.info("ASR: faster-whisper %s int8 on CPU (%d threads) loaded in %.1fs.",
                    ASR_MODEL, ASR_THREADS, time.time() - t0)
        return asr
    except ImportError:
        return None

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
HUB_MODEL = "k2-fsa/OmniVoice"


def _local_model() -> bool:
    """True when MODEL_DIR holds the model (Docker creates an empty folder for the mount)."""
    return (MODEL_DIR / "config.json").is_file()


VOICES_DIR = PROJECT_ROOT / "Voices"
VOICES_META = VOICES_DIR / "voices.json"          # optional sidecar: {file: {ref_text, display, language}}
# A transcript beside a voice file is used as its ref_text (no Whisper, exact
# words): <stem>.txt, or LibriTTS(-R)'s <stem>.normalized.txt / <stem>.original.txt.
TRANSCRIPT_SUFFIXES = (".txt", ".normalized.txt", ".original.txt")
SAMPLES_DIR = PROJECT_ROOT / "output" / "samples"
# Persisted VoiceClonePrompt objects (upstream 0.2.0, db1039f). See _clone_prompt.
PROMPT_CACHE_DIR = PROJECT_ROOT / "output" / "prompt_cache"
# Transcoded copies of reference voices libsndfile cannot open. See ensure_decodable.
VOICE_CACHE_DIR = PROJECT_ROOT / "output" / "voice_cache"

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}

# Non-verbal symbols understood by the model (must match
# ``omnivoice.models.omnivoice._NONVERBAL_PATTERN``).
NONVERBAL_TAGS = [
    "laughter", "sigh", "confirmation-en",
    "question-en", "question-ah", "question-oh", "question-ei", "question-yi",
    "surprise-ah", "surprise-oh", "surprise-wa", "surprise-yo",
    "dissatisfaction-hnn",
]

# Voice-design attributes, grouped by mutually-exclusive category. English
# only; the model also accepts Chinese dialect attributes, which are not listed.
VOICE_DESIGN_CATEGORIES = {
    "gender": ["male", "female"],
    "age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "style": ["whisper"],
    "accent": [
        "american accent", "british accent", "australian accent", "canadian accent",
        "indian accent", "chinese accent", "korean accent", "japanese accent",
        "portuguese accent", "russian accent",
    ],
}

# Defaults mirror ``OmniVoiceGenerationConfig``, except num_step: 16, chosen by
# listening test (twice as fast as 32, same word accuracy).
PARAM_DEFAULTS = {
    "num_step": 16,
    "guidance_scale": 2.0,
    "t_shift": 0.1,
    "speed": 1.0,
    "duration": None,
    "denoise": True,
    "preprocess_prompt": True,
    "postprocess_output": True,
    "position_temperature": 5.0,
    "class_temperature": 0.0,
    "layer_penalty_factor": 5.0,
    # Added upstream in 0.2.0 (398b611). Both were hard-coded at 0.1s before;
    # exposing them lets a caller tighten paragraph joins (0.0 disables).
    "pad_duration": 0.1,
    "fade_duration": 0.1,
    # Added upstream in 0.2.1 (52aa404). Expands numbers/dates/currency before
    # synthesis. Requires WeTextProcessing, which depends on pynini -- no Windows
    # wheel exists, so this only works inside the Linux container. Default off:
    # the English ruleset mis-renders decade forms ("1980s" -> "eighty seconds"),
    # which would be wrong constantly in a management textbook.
    "normalize_text": False,
}


def ensure_ffmpeg_on_path() -> Optional[str]:
    """Make sure an ffmpeg binary is reachable.

    Order: existing PATH -> a Gyan winget build -> the imageio-ffmpeg bundle.
    Whichever is found has its directory prepended to ``PATH`` so that both
    ``subprocess`` (m4b muxing) and librosa/audioread (mp3/m4a reference
    decoding) can use it. Returns the ffmpeg path or ``None``.
    """
    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found

    import glob

    candidates = glob.glob(os.path.expanduser(
        r"~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\**\bin\ffmpeg.exe"
    ), recursive=True)
    if candidates:
        found = candidates[0]

    if not found:
        try:
            import imageio_ffmpeg
            found = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            found = None

    if found:
        bindir = str(Path(found).parent)
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    return found


def content_digest(path: Path) -> str:
    """SHA-1 of a file's size + bytes.

    Both on-disk caches key on this rather than on ``(path, st_mtime_ns)`` so an
    entry stays valid when the file is moved or re-saved unchanged, and so the
    host and the container -- which see the same voice at different absolute
    paths and different mtimes across the bind mount -- resolve to one entry
    instead of duplicating the work.

    Memoized per process on (path, size, mtime): otherwise every request
    re-hashed its reference (up to ~34 MB) over the bind mount.
    """
    st = path.stat()
    memo_key = (str(path), st.st_size, st.st_mtime_ns)
    digest = _DIGEST_MEMO.get(memo_key)
    if digest is None:
        h = hashlib.sha1(str(st.st_size).encode("utf-8"))
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = _DIGEST_MEMO[memo_key] = h.hexdigest()
    return digest


_DIGEST_MEMO: dict[tuple, str] = {}


def ensure_decodable(path: Path) -> Path:
    """Return a path OmniVoice can actually load, transcoding if it cannot.

    libsndfile 1.2.2 (bundled with soundfile) reads wav/flac/ogg and, since 1.1,
    mp3 — but it has no AAC decoder, so ``.m4a``/``.aac`` (and ``.wma``) fail with
    "Format not recognised". Upstream's fallback for that is librosa, which this
    project deliberately does not install (its numba dependency will not build on
    Python 3.12), so the failure surfaces to the user as a bare
    ``ModuleNotFoundError: No module named 'librosa'``.

    ffmpeg is already a hard dependency here (m4b muxing) and is already placed on
    PATH by :func:`ensure_ffmpeg_on_path`, so unreadable references are converted
    to a cached 16-bit WAV instead. The source rate and channel count are kept —
    OmniVoice does its own mono-mixdown and resample — so this adds a container
    change only, not a resampling step.

    Detection is by probing rather than by extension: a mislabelled file is
    common with hand-collected voice samples, and ``sf.info`` settles it cheaply
    without decoding the audio.
    """
    try:
        import soundfile as sf
        sf.info(str(path))
        return path                      # libsndfile can open it directly
    except Exception:
        pass

    out = VOICE_CACHE_DIR / f"{path.stem}_{content_digest(path)[:16]}.wav"
    if out.is_file():
        return out

    ff = ensure_ffmpeg_on_path()
    if not ff:
        raise RuntimeError(
            f"{path.name} needs transcoding (libsndfile cannot decode it) but ffmpeg "
            f"was not found. Install ffmpeg, or use a .wav/.mp3/.flac reference."
        )
    VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    proc = subprocess.run(
        [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-c:a", "pcm_s16le", "-f", "wav", str(tmp)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0 or not tmp.is_file():
        try:
            tmp.unlink()
        except Exception:
            pass
        detail = (proc.stdout or b"").decode(errors="replace").strip()[-300:]
        raise RuntimeError(f"ffmpeg could not decode {path.name}: {detail or 'unknown error'}")
    tmp.replace(out)                     # atomic: never leave a half-written cache entry
    logger.info("Transcoded %s -> %s (libsndfile cannot decode it).", path.name, out.name)
    return out


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines, collapsing wrapped lines
    within a paragraph to a single line of prose."""
    paras = re.split(r"\n\s*\n", (text or "").strip())
    paras = [re.sub(r"\s+", " ", p).strip() for p in paras]
    return [p for p in paras if p]


MAX_RUN_WORDS = 30  # ~11 s of speech
MAX_REF_SECONDS = 20.0  # upstream's trim threshold for references (trim_long_audio)


def bound_runs(text: str, max_words: int = MAX_RUN_WORDS) -> str:
    """Insert a comma after every ``max_words`` words that pass without
    punctuation OmniVoice can split at.

    Upstream chunks long text only at ``.,;:!?`` (``chunk_text_punctuation``),
    so a run without any is generated -- and decoded -- as one piece whose VRAM
    grows with its length (~13 MB per second of audio). Measured on the 3080:
    333 unpunctuated words peaked at 3.5 GB allocated and 666 words went past
    4 GB, vs 2.2 GB for the same 333 words punctuated. Normal prose rarely runs
    30 words without a comma, so it is left as is.
    """
    from omnivoice.utils.text import CLOSING_MARKS, SPLIT_PUNCTUATION

    closing = "".join(CLOSING_MARKS)
    words, run = (text or "").split(" "), 0
    for i, w in enumerate(words):
        if not w:
            continue
        if w.rstrip(closing)[-1:] in SPLIT_PUNCTUATION:
            run = 0
            continue
        run += 1
        if run >= max_words:
            words[i] = w + ","
            run = 0
    return " ".join(words)


def _pick_device(requested: Optional[str]) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OmniVoiceEngine:
    """Singleton-style engine. Call :meth:`load` once, then :meth:`generate`."""

    def __init__(self, device: Optional[str] = None, load_asr: bool = False):
        self.device = device or os.environ.get("OMNIVOICE_DEVICE") or None
        self.want_asr = load_asr or os.environ.get("OMNIVOICE_LOAD_ASR") == "1"
        self.model = None
        self.sampling_rate = 24000
        self.gpu_name = ""
        self.ready = False
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        self._prompt_cache: dict[str, Any] = {}
        self._cpu_tokenizer = None  # see _build_prompt
        self._prompt_pool = None    # out-of-process prompt builder, see _build_prompt_in_worker
        self.flashinfer_on = False  # set by _build_model
        self._voice_meta, self._voice_meta_mtime = {}, -1.0   # loaded by _fresh_voice_meta
        # Idle offload: free the model from VRAM after this many seconds with no
        # generation, and reload it on demand. 0 / None disables it. The server
        # sets this (default 600s); direct engine users leave it at 0 so the
        # model stays resident.
        self.idle_ttl = float(os.environ.get("OMNIVOICE_IDLE_TTL") or 0)
        self._last_active = 0.0
        self._idle_thread_started = False
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    def _build_model(self) -> None:
        """Import omnivoice and load the weights into VRAM. Caller holds ``_lock``."""
        ensure_ffmpeg_on_path()  # for mp3/m4a reference voices & m4b muxing

        import torch
        from omnivoice import OmniVoice

        self.device = _pick_device(self.device)
        dtype = torch.float16 if str(self.device).startswith(("cuda", "xpu")) else torch.float32
        if DTYPE:
            named = {"float16": torch.float16, "fp16": torch.float16, "float32": torch.float32,
                     "fp32": torch.float32, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
            if DTYPE in named:
                dtype = named[DTYPE]
            else:
                logger.warning("OMNIVOICE_DTYPE=%r is not float16, float32 or bfloat16; using %s.", DTYPE, dtype)
        elif str(self.device).startswith("cuda"):
            props = torch.cuda.get_device_properties(self.device)
            if _no_tensor_cores(props.name, (props.major, props.minor)):
                dtype = torch.float32
                logger.info("%s has no tensor cores: running in float32, which is faster there "
                            "than float16 (OMNIVOICE_DTYPE overrides this).", props.name)
        if str(self.device).startswith("cuda"):
            _refuse_second_instance()
        if str(self.device).startswith("cuda") and (MEM_FRACTION > 0 or VRAM_CAP_MIB > 0):
            # After CUDA init (get_device_properties), before the weights load.
            total = torch.cuda.get_device_properties(self.device).total_memory
            frac = min(1.0, MEM_FRACTION if MEM_FRACTION > 0 else VRAM_CAP_MIB * 2**20 / total)
            torch.cuda.set_per_process_memory_fraction(frac, self.device)
            logger.info("VRAM cap: torch allocator limited to %.0f MiB (fraction %.3f of %s).",
                        frac * total / 2**20, frac, self.device)

        # ./model (mounted) when it holds the model, else the Hugging Face Hub: the first
        # start downloads it (about 3 GB) into HF_HOME, a persistent volume in Docker.
        src = str(MODEL_DIR) if _local_model() else HUB_MODEL
        logger.info("Loading OmniVoice from %s on %s (%s) ...", src, self.device, dtype)
        t0 = time.time()
        self.model = OmniVoice.from_pretrained(
            src,
            device_map=self.device,
            dtype=dtype,
            asr_device=ASR_DEVICE,  # for the transformers fallback, see _ensure_asr
        )
        self.flashinfer_on = False
        if FLASHINFER and str(self.device).startswith("cuda") and dtype != torch.float16:
            logger.info("FlashInfer skipped: it needs float16, the model runs in %s.", dtype)
        elif FLASHINFER and str(self.device).startswith("cuda"):
            from omnivoice.models.omnivoice_flashinfer import apply_flashinfer
            fi_kw = {}
            if FLASHINFER == "graphs":
                fi_kw = (dict(cuda_graph_buckets=FI_BUCKETS, overhead_budget=FI_OVERHEAD)
                         if FI_BUCKETS else dict(enable_cuda_graph=True))
            apply_flashinfer(self.model, **fi_kw)
            if fi_kw.get("enable_cuda_graph"):
                _drop_graphs_after_each_generation(self.model)
            self.flashinfer_on = True
            logger.info("FlashInfer enabled (%s).", "no CUDA graphs" if not fi_kw else
                        "CUDA graph per generation" if fi_kw.get("enable_cuda_graph") else fi_kw)
        if COMPILE and str(self.device).startswith("cuda"):
            # dynamic=True: every text chunk has its own sequence length.
            self.model.llm.compile(dynamic=True, mode=None if COMPILE in ("1", "default") else COMPILE)
            logger.info("torch.compile enabled on the LLM backbone (mode=%s).", COMPILE)
        if self.want_asr:
            self._ensure_asr()
        self.sampling_rate = int(self.model.sampling_rate or 24000)
        if str(self.device).startswith("cuda"):
            try:
                idx = int(str(self.device).split(":")[1]) if ":" in str(self.device) else 0
                self.gpu_name = torch.cuda.get_device_name(idx)
            except Exception:
                self.gpu_name = "CUDA device"
        self._last_active = time.monotonic()
        logger.info("Model ready in %.1fs (sr=%d, gpu=%s)", time.time() - t0, self.sampling_rate, self.gpu_name)

    def load(self) -> None:
        """Load the model. Safe to run in a background thread."""
        try:
            with self._lock:
                if self.model is None:
                    self._build_model()
                    if COMPILE:
                        self._warmup_locked()
                self.ready = True
            self._start_idle_watch()
        except Exception as e:  # surfaced to the UI
            self.error = f"{type(e).__name__}: {e}"
            logger.exception("Model load failed")

    def _warmup_locked(self) -> None:
        """Trigger torch.compile at startup, before ``ready`` flips, so no request
        pays for it: ~70 s with an empty inductor cache, ~30 s with the persisted
        one. A reload after idle offload recompiles from the cache in ~2 s."""
        from omnivoice import OmniVoiceGenerationConfig

        t0 = time.time()
        with self._vram_trimmed():
            self.model.generate(text="Warming up the voice engine.",
                                generation_config=OmniVoiceGenerationConfig(num_step=2))
        logger.info("torch.compile warm-up done in %.1fs.", time.time() - t0)

    def _ensure_loaded_locked(self) -> None:
        """Reload the model if it was offloaded for being idle. Caller holds ``_lock``."""
        if self.model is None:
            logger.info("Reloading model on demand (was offloaded after %.0fs idle TTL).", self.idle_ttl)
            self._build_model()

    # ----------------------------------------------------------- idle offload
    def unload(self, *, force: bool = False) -> bool:
        """Free the model from VRAM. Returns True if it actually unloaded.

        Re-checks the idle window under the lock (unless ``force``) so it never
        offloads a model that a request just used.
        """
        with self._lock:
            if self.model is None:
                return False
            idle = time.monotonic() - self._last_active
            if not force and self.idle_ttl and idle < self.idle_ttl:
                return False
            logger.info("Offloading model from %s after %.0fs idle (TTL %.0fs).",
                        self.device, idle, self.idle_ttl)
            # Collect the classes of every submodule first. Some carry
            # functools.lru_cache-decorated *methods* keyed on ``self`` (the
            # Higgs audio tokenizer does), which would otherwise keep ~0.8 GB of
            # weights resident after the model is dropped. The set comprehension
            # has its own scope, so it leaves no lingering model reference.
            try:
                classes = {k for sub in self.model.modules() for k in type(sub).__mro__}
            except Exception:
                classes = set()
            self.model = None
            self._cpu_tokenizer = None
            self._stop_prompt_worker()
            self._prompt_cache.clear()  # prompts are bound to the old model instance
        # Outside the lock: drop the per-instance caches that pin GPU tensors,
        # then reclaim VRAM.
        try:
            import functools
            for klass in classes:
                for val in list(vars(klass).values()):
                    if isinstance(val, functools._lru_cache_wrapper):
                        try:
                            val.cache_clear()
                        except Exception:
                            pass
        except Exception:
            pass
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return True

    def _start_idle_watch(self) -> None:
        """Start the background idle-offload watcher once (if a TTL is set)."""
        if self._idle_thread_started or not self.idle_ttl or self.idle_ttl <= 0:
            return
        self._idle_thread_started = True
        threading.Thread(target=self._idle_watch_loop, name="omnivoice-idle", daemon=True).start()
        logger.info("Idle offload enabled: model unloads after %.0fs without a request.", self.idle_ttl)

    def _idle_watch_loop(self) -> None:
        interval = max(5.0, min(self.idle_ttl / 4.0, 60.0))
        while self.idle_ttl and self.idle_ttl > 0:
            time.sleep(interval)
            try:
                self.unload()
            except Exception:
                logger.exception("Idle offload check failed")

    # --------------------------------------------------------------- metadata
    def _read_voice_meta(self) -> dict:
        if VOICES_META.is_file():
            try:
                return json.loads(VOICES_META.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Could not parse %s", VOICES_META)
        return {}

    def _fresh_voice_meta(self) -> dict:
        """voices.json, re-read whenever the file changes (no restart needed)."""
        try:
            mtime = VOICES_META.stat().st_mtime
        except OSError:
            mtime = None
        if mtime != self._voice_meta_mtime:
            self._voice_meta = self._read_voice_meta() if mtime is not None else {}
            self._voice_meta_mtime = mtime
        return self._voice_meta

    def voice_ref_text(self, voice_file: str) -> Optional[str]:
        """The known transcript of a Voices/ file: voices.json ``ref_text``, else a
        transcript file beside it (TRANSCRIPT_SUFFIXES). None = Whisper decides."""
        try:
            path = self.voice_path(voice_file)
        except (ValueError, FileNotFoundError):
            return None
        text = (self._fresh_voice_meta().get(path.name) or {}).get("ref_text")
        if not text:
            for suffix in TRANSCRIPT_SUFFIXES:
                side = path.with_name(path.stem + suffix)
                if side.is_file():
                    text = side.read_text(encoding="utf-8", errors="replace")
                    if text.strip():
                        break
        return " ".join(text.split()) if text and text.strip() else None

    def list_voices(self) -> list[dict]:
        voices = []
        if VOICES_DIR.is_dir():
            for p in sorted(VOICES_DIR.iterdir(), key=lambda x: x.name.lower()):
                if p.suffix.lower() in AUDIO_EXTS:
                    meta = self._fresh_voice_meta().get(p.name, {})
                    voices.append({
                        "file": p.name,
                        "display": meta.get("display", p.stem.replace("_", " ")),
                        "ext": p.suffix.lower().lstrip("."),
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                        "has_ref_text": bool(self.voice_ref_text(p.name)),
                        "language": meta.get("language", ""),
                        # libsndfile reads wav/flac/ogg/mp3 directly; anything else
                        # (.m4a/.aac/.wma) is transcoded via ffmpeg on first use,
                        # so every listed voice is usable either way.
                        "native_decode": p.suffix.lower() in {".wav", ".flac", ".ogg", ".opus", ".mp3"},
                    })
        return voices

    def languages(self) -> list[str]:
        # Don't trigger the first import of the omnivoice package from a request
        # thread while the model is still loading in a background thread: two
        # threads doing the first import concurrently can trip Python's import
        # lock and raise _DeadlockError (which would abort the model load). Until
        # the package is imported / the model is ready, return a minimal list.
        import sys
        if not self.ready and "omnivoice.utils.lang_map" not in sys.modules:
            return ["Auto", "English"]
        try:
            from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
            return ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
        except Exception:
            return ["Auto", "English"]

    @contextlib.contextmanager
    def _vram_trimmed(self):
        """Hand torch's cached-but-free VRAM back to the driver when a request
        ends. Without this the reserved pool only ever grows: measured on the
        3080, one long generation left 3.0 GB reserved and each new-voice prompt
        build (audio-tokenizer encode of the <=20 s reference, in fp32) added
        ~2.3 GB more, reaching 7.7 GB with 1.9 GB actually in use -- far over a
        3 GB budget on a card shared with other work. Costs a few ms per request.
        (An OOM is recovered in generate(), not here.)"""
        try:
            yield
        finally:
            if str(self.device).startswith("cuda"):
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def _vram(self) -> Optional[dict]:
        """torch allocator stats for this process, in MiB (the CUDA context, a
        few hundred MiB, is not included). ``card_used`` is the whole card, so
        it also counts other programs on it. None when not on a GPU or CUDA is not up."""
        if not str(self.device).startswith("cuda"):
            return None
        try:
            import torch
            if not torch.cuda.is_initialized():
                return None
            mib = 1024 * 1024
            free, total = torch.cuda.mem_get_info()
            return {
                "allocated": round(torch.cuda.memory_allocated() / mib),
                "peak_allocated": round(torch.cuda.max_memory_allocated() / mib),
                "reserved": round(torch.cuda.memory_reserved() / mib),
                "peak_reserved": round(torch.cuda.max_memory_reserved() / mib),
                "card_used": round((total - free) / mib),
                "card_total": round(total / mib),
            }
        except Exception:
            return None

    def meta(self) -> dict:
        loaded = self.model is not None
        return {
            "ready": self.ready,
            "error": self.error,
            "loaded": loaded,                       # False while offloaded for idleness
            "idle_ttl": self.idle_ttl,
            "seconds_idle": round(time.monotonic() - self._last_active, 1) if (loaded and self._last_active) else None,
            "device": self.device or "(detecting)",
            "gpu_name": self.gpu_name,
            "sampling_rate": self.sampling_rate,
            "model_path": str(MODEL_DIR) if _local_model() else f"{HUB_MODEL} (Hugging Face Hub)",
            "asr_loaded": self.want_asr,
            "asr_device": ASR_DEVICE,
            "asr_backend": self._asr_backend(),
            "vram": self._vram(),
            "voices": self.list_voices(),
            "languages": self.languages(),
            "voice_design": VOICE_DESIGN_CATEGORIES,
            "nonverbal_tags": NONVERBAL_TAGS,
            "param_defaults": PARAM_DEFAULTS,
        }

    # ------------------------------------------------------------- voice path
    def voice_path(self, file_name: str) -> Path:
        """Resolve a voice file safely inside the Voices directory."""
        p = (VOICES_DIR / file_name).resolve()
        if VOICES_DIR.resolve() not in p.parents:
            raise ValueError("Invalid voice path")
        if not p.is_file():
            raise FileNotFoundError(file_name)
        return p

    # ----------------------------------------------------------- clone prompt
    def _clone_prompt(self, voice_file: str, ref_text: Optional[str], preprocess_prompt: bool):
        path = self.voice_path(voice_file)
        # Keyed on the reference's CONTENT (see content_digest), not its path or
        # mtime, so a prompt built once is reused by the container, survives the
        # file being moved, and is not invalidated by a no-op re-save.
        key_src = f"{content_digest(path)}|{ref_text or ''}|{int(preprocess_prompt)}"
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
        if key in self._prompt_cache:
            return self._prompt_cache[key]

        # On-disk cache (upstream 0.2.0, db1039f). This matters more than it
        # looks: with ``ref_text=None`` -- which is every voice in
        # ``openai_voices.json`` -- building a prompt runs Whisper to
        # auto-transcribe the reference. That loads Whisper (on the CPU, see
        # ASR_DEVICE) which then stays resident, and it repeats on every fresh
        # process and after every idle offload. A saved prompt already contains
        # the transcript, so the ASR model is never loaded at all.
        cached = PROMPT_CACHE_DIR / f"{key}.pt"
        if cached.is_file():
            try:
                from omnivoice import VoiceClonePrompt
                prompt = VoiceClonePrompt.load(
                    str(cached), map_location=str(self.device or "cpu")
                )
                self._prompt_cache[key] = prompt
                logger.info("Voice prompt for %s loaded from cache (no ASR).", voice_file)
                return prompt
            except Exception:
                logger.warning("Prompt cache %s unreadable; rebuilding.", cached.name)

        prompt = self._build_prompt(ensure_decodable(path), ref_text, preprocess_prompt)
        try:
            PROMPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            prompt.save(str(cached))
            logger.info("Voice prompt for %s cached to %s.", voice_file, cached.name)
        except Exception:
            logger.warning("Could not persist prompt cache for %s.", voice_file)
        self._prompt_cache[key] = prompt
        return prompt

    def _asr_backend(self) -> Optional[str]:
        """Which ASR is loaded right now (None until a new voice needs one)."""
        if self._prompt_pool is not None:
            return "faster-whisper (worker process)"
        pipe = getattr(self.model, "_asr_pipe", None)
        if pipe is None:
            return None
        return "faster-whisper" if isinstance(pipe, _FasterWhisperASR) else "transformers"

    def _ensure_asr(self) -> None:
        """Load the ASR used to auto-transcribe references, once per model load:
        faster-whisper on the CPU if installed, else OmniVoice's transformers
        pipeline on ASR_DEVICE. Caller holds ``_lock`` (or is loading)."""
        if self.model._asr_pipe is None:
            asr = _make_cpu_asr()
            if asr is not None:
                self.model._asr_pipe = asr
            else:
                self.model.load_asr_model()

    def _build_prompt_in_worker(self, kwargs: dict):
        """Build a voice prompt in the out-of-process worker (core/prompt_worker.py).
        A crash there fails only this request; the next one starts a new worker.
        Caller holds ``_lock``."""
        import torch
        from concurrent.futures.process import BrokenProcessPool
        from omnivoice import VoiceClonePrompt
        from core import prompt_worker

        if self._prompt_pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            self._prompt_pool = ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context("spawn"), initializer=prompt_worker.init,
                initargs=(self.model.audio_tokenizer.name_or_path, ASR_MODEL, ASR_THREADS))
        t0 = time.time()
        try:
            tokens, text, rms = self._prompt_pool.submit(
                prompt_worker.build, kwargs["ref_audio"], kwargs["ref_text"], kwargs["preprocess_prompt"],
                self.sampling_rate).result(timeout=900)
        except BrokenProcessPool as e:
            self._prompt_pool = None
            logger.error("Voice prompt worker died (%s); a new one starts on the next request.", e)
            raise RuntimeError("The voice-prompt worker crashed while building this voice; the server is "
                               "fine -- retry the request.") from None
        logger.info("Voice prompt built in the worker process (%s) in %.1fs.",
                    "encode, transcript given" if kwargs["ref_text"] else "Whisper + encode", time.time() - t0)
        return VoiceClonePrompt(ref_audio_tokens=torch.from_numpy(tokens), ref_text=text, ref_rms=rms)

    def _stop_prompt_worker(self) -> None:
        pool, self._prompt_pool = self._prompt_pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _build_prompt(self, ref_audio: Path, ref_text: Optional[str], preprocess_prompt: bool):
        """``create_voice_clone_prompt`` with the reference encode on the CPU.

        The audio tokenizer encodes the whole (<=20 s) reference in one pass in
        fp32, and on the 3080 that transient measured 7.8 GB allocated for a
        20 s clip (~5.9 GB on top of the resident model) -- it alone breaks the
        budget of a card shared with other work. So a CPU copy of the tokenizer is
        swapped in for the encode; the resulting codes are tiny and generate()
        moves them to the model's device. Whisper is already on the CPU (see
        ASR_DEVICE). Caller holds ``_lock``.
        """
        if ref_text or not preprocess_prompt:
            # Every reference is cut to <= MAX_REF_SECONDS before encoding,
            # whatever the request flags. Upstream trims only when
            # preprocess_prompt is on AND no ref_text is given (a transcript must
            # match its audio), so otherwise the whole reference goes into every
            # generation: a 110 s one went past 4 GB on the 3080. For a longer
            # reference we drop ref_text and force preprocessing, so it is
            # trimmed and re-transcribed on the CPU -- chosen over rejecting
            # with a 400 so existing transcripts in voices.json keep working.
            try:
                import soundfile as sf
                ref_seconds = sf.info(str(ref_audio)).duration
            except Exception:
                ref_seconds = 0.0
            if ref_seconds > MAX_REF_SECONDS:
                logger.warning("Reference %s is %.0fs (> %.0fs; ref_text=%s, preprocess_prompt=%s): "
                               "trimming it and auto-transcribing instead (VRAM budget).",
                               Path(ref_audio).name, ref_seconds, MAX_REF_SECONDS,
                               "given" if ref_text else "none", preprocess_prompt)
                ref_text, preprocess_prompt = None, True
        kwargs = dict(ref_audio=str(ref_audio), ref_text=(ref_text or None),
                      preprocess_prompt=preprocess_prompt)
        on_gpu = str(self.device).startswith("cuda")
        if on_gpu and (PROMPT_WORKER == "1" or (PROMPT_WORKER == "auto" and self.flashinfer_on)):
            return self._build_prompt_in_worker(kwargs)
        if not ref_text:
            self._ensure_asr()
        if not on_gpu:
            return self.model.create_voice_clone_prompt(**kwargs)
        if self._cpu_tokenizer is None:
            import torch
            from transformers import HiggsAudioV2TokenizerModel
            src = self.model.audio_tokenizer.name_or_path
            t0 = time.time()
            self._cpu_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
                src, device_map="cpu", dtype=torch.float32
            ).eval()
            logger.info("CPU audio tokenizer for prompt encoding loaded in %.1fs.", time.time() - t0)
        gpu_tokenizer = self.model.audio_tokenizer
        self.model.audio_tokenizer = self._cpu_tokenizer
        try:
            t0 = time.time()
            prompt = self.model.create_voice_clone_prompt(**kwargs)
            logger.info("Voice prompt built on CPU (%s) in %.1fs.",
                        "encode, transcript given" if kwargs.get("ref_text") else "Whisper + encode", time.time() - t0)
            return prompt
        finally:
            self.model.audio_tokenizer = gpu_tokenizer
            _trim_host_heap()

    def clear_prompt_cache(self) -> None:
        self._prompt_cache.clear()

    # -------------------------------------------------------------- generate
    def generate(self, **kwargs) -> dict:
        """Run one request (arguments: see ``_generate``).

        A CUDA OOM -- normally the memory cap -- is recovered: VRAM is freed and
        the request is retried once with the text split one sentence per
        generation unit; if that also fails, GPUMemoryError (-> 503). The model
        stays loaded either way. The collect happens outside the except block on
        purpose: until the OOM error and its traceback are gone, their reference
        cycles keep the failed request's tensors alive, and the next attempt
        then OOMs too (measured: ~300 MiB left pinned)."""
        try:
            return self._generate(**kwargs)
        except Exception as e:
            if not _is_oom(e):
                raise
            oom = str(e).splitlines()[0]
        self._free_after_oom(oom)
        smaller = None if kwargs.get("duration") else _sentence_units(kwargs.get("text", ""))
        if smaller:
            logger.warning("Retrying once with %d sentence-sized generation units.", smaller.count("\n\n") + 1)
            try:
                return self._generate(**{**kwargs, "text": smaller, "chunk_paragraphs": True})
            except Exception as e:
                if not _is_oom(e):
                    raise
                oom = str(e).splitlines()[0]
            self._free_after_oom(oom)
        raise GPUMemoryError(f"GPU memory limit reached for this request ({oom}). "
                             "The server recovered; retry, or send shorter text.")

    def _free_after_oom(self, why: str) -> None:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        logger.error("CUDA out of memory; VRAM freed: %s", why)

    def _generate(
        self,
        *,
        text: str,
        mode: str = "clone",
        language: Optional[str] = None,
        voice: Optional[str] = None,
        ref_text: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        instruct: Optional[str] = None,
        num_step: int = 32,
        guidance_scale: float = 2.0,
        t_shift: float = 0.1,
        speed: Optional[float] = None,
        duration: Optional[float] = None,
        denoise: bool = True,
        preprocess_prompt: bool = True,
        postprocess_output: bool = True,
        position_temperature: float = 5.0,
        class_temperature: float = 0.0,
        layer_penalty_factor: float = 5.0,
        pad_duration: float = 0.1,
        fade_duration: float = 0.1,
        normalize_text: bool = False,
        chunk_paragraphs: bool = True,
        para_gap: float = 0.35,
        save: bool = True,
    ) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "Model is not loaded yet.")
        if not text or not text.strip():
            raise ValueError("Text is empty.")

        from omnivoice import OmniVoiceGenerationConfig

        lang = None if (not language or language.lower() == "auto") else language

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step),
            guidance_scale=float(guidance_scale),
            t_shift=float(t_shift),
            denoise=bool(denoise),
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
            position_temperature=float(position_temperature),
            class_temperature=float(class_temperature),
            layer_penalty_factor=float(layer_penalty_factor),
            pad_duration=float(pad_duration),
            fade_duration=float(fade_duration),
        )

        # Shared args for every model.generate call (text added per chunk below).
        kw: dict[str, Any] = dict(language=lang, generation_config=gen_config)
        if normalize_text:
            kw["normalize_text"] = True
        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        with self._lock, self._vram_trimmed():
            self._ensure_loaded_locked()       # reload if offloaded for idleness
            self._last_active = time.monotonic()
            if mode == "clone":
                if ref_audio_path:
                    kw["voice_clone_prompt"] = self._build_prompt(
                        ensure_decodable(Path(ref_audio_path)), ref_text, preprocess_prompt
                    )
                elif voice:
                    kw["voice_clone_prompt"] = self._clone_prompt(
                        voice, ref_text or self.voice_ref_text(voice), preprocess_prompt)
                else:
                    raise ValueError("Clone mode requires a reference voice.")
            elif mode == "design":
                if not instruct or not instruct.strip():
                    raise ValueError("Design mode requires an instruct string.")
                kw["instruct"] = instruct.strip()
            elif mode == "auto":
                pass
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Voice design / auto can still be combined with an instruct hint.
            if mode == "clone" and instruct and instruct.strip():
                kw["instruct"] = instruct.strip()

            # Chunk at paragraph boundaries (each paragraph = one generation
            # unit, joined by a short pause) instead of letting the model split
            # long text at every sentence. Skipped when a fixed duration is set
            # (that targets a single clip). Each paragraph's output is already
            # faded/padded by post-processing, so a plain silence join is clean.
            paras = split_paragraphs(text) if (chunk_paragraphs and "duration" not in kw) else [text.strip()]
            paras = [bound_runs(p) for p in paras]  # keep every chunk short (VRAM budget)

            t0 = time.time()
            if len(paras) > 1:
                gap = np.zeros(int(max(0.0, para_gap) * self.sampling_rate), dtype=np.float32)
                pieces: list[np.ndarray] = []
                for k, para in enumerate(paras):
                    pieces.append(np.asarray(self.model.generate(text=para, **kw)[0], dtype=np.float32))
                    if k < len(paras) - 1:
                        pieces.append(gap)
                audio = np.concatenate(pieces)
            else:
                audio = np.asarray(self.model.generate(text=paras[0], **kw)[0], dtype=np.float32)
            gen_seconds = time.time() - t0
            self._last_active = time.monotonic()   # reset the idle clock at completion
        audio_seconds = len(audio) / self.sampling_rate if self.sampling_rate else 0.0
        rtf = (gen_seconds / audio_seconds) if audio_seconds > 0 else 0.0

        out_path = None
        if save:
            out_path = self._save(audio, mode=mode, voice=voice, instruct=instruct, kw=kw,
                                  gen_seconds=gen_seconds, audio_seconds=audio_seconds, rtf=rtf,
                                  text=text, language=language)

        return {
            "audio": audio,
            "sampling_rate": self.sampling_rate,
            "gen_seconds": round(gen_seconds, 3),
            "audio_seconds": round(audio_seconds, 3),
            "rtf": round(rtf, 4),
            "out_file": out_path.name if out_path else None,
        }

    # ------------------------------------------------------------------ save
    def _save(self, audio: np.ndarray, *, mode, voice, instruct, kw, gen_seconds,
              audio_seconds, rtf, text, language) -> Path:
        import soundfile as sf

        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = (voice or instruct or "auto") if mode != "auto" else "auto"
        tag = "".join(c for c in str(tag) if c.isalnum() or c in "-_")[:24] or mode
        short = hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
        base = f"{stamp}_{mode}_{tag}_{short}"
        wav_path = SAMPLES_DIR / f"{base}.wav"
        sf.write(str(wav_path), audio, self.sampling_rate)

        meta = {
            "created": stamp,
            "mode": mode,
            "text": text,
            "language": language,
            "voice": voice,
            "instruct": instruct,
            "gen_seconds": round(gen_seconds, 3),
            "audio_seconds": round(audio_seconds, 3),
            "rtf": round(rtf, 4),
            "params": {k: v for k, v in asdict(kw["generation_config"]).items()},
            "speed": kw.get("speed"),
            "duration": kw.get("duration"),
        }
        (SAMPLES_DIR / f"{base}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return wav_path


# Module-level singleton used by the server.
ENGINE = OmniVoiceEngine()
