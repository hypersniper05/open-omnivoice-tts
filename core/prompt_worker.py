"""Out-of-process voice-prompt builder: reference trim + ASR + CPU encode.

Why a separate process: with FlashInfer loaded (OMNIVOICE_FLASHINFER), the
server segfaulted (exit 139, 2026-10-03) while building a new voice's prompt on
the CPU -- FlashInfer's native stack and CTranslate2 (faster-whisper) shared one
address space. Here that work runs in a 'spawn' child that never initializes
CUDA and never imports FlashInfer; only the finished prompt (int codes, text,
rms) comes back. A crash in the child fails that one request, and the engine
starts a fresh child for the next one.

The prompt is built by upstream's own ``OmniVoice.create_voice_clone_prompt``
(trim/silence removal, ASR, encode, punctuation), called on a small stand-in
that carries only the CPU audio tokenizer and the ASR, so the result is the
same as the in-process path.
"""
from __future__ import annotations

import os

import numpy as np

_STATE: dict = {}


class FasterWhisperASR:
    """Stand-in for OmniVoice's ``_asr_pipe``: ``OmniVoice.transcribe`` calls it
    with ``{"array", "sampling_rate"}`` (or a file path) and reads ``["text"]``,
    so upstream's trimming / silence removal before ASR stays unchanged."""

    def __init__(self, model_name: str, threads: int):
        import torch
        from faster_whisper import WhisperModel
        # CTranslate2 shares OpenMP with torch and resets its thread count
        # (16 -> cpu_threads). In-process, torch.compile guards on that, so the
        # next generation recompiled the backbone (~30 s). Restore it after every use.
        self._torch_threads = torch.get_num_threads()
        try:
            self._model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)
        finally:
            torch.set_num_threads(self._torch_threads)

    def __call__(self, audio):
        import torch
        try:
            return self._transcribe(audio)
        finally:
            torch.set_num_threads(self._torch_threads)

    def _transcribe(self, audio):
        import torch
        import torchaudio
        if isinstance(audio, str):
            # Decode here rather than via faster-whisper's decode_audio: 1.2.1
            # passes av.open(metadata_errors=...), which PyAV 19 removed.
            from omnivoice.utils.audio import load_audio
            audio = {"array": load_audio(audio, 16000), "sampling_rate": 16000}
        wav = torch.as_tensor(np.asarray(audio["array"], dtype=np.float32)).reshape(-1)
        if audio["sampling_rate"] != 16000:  # faster-whisper takes 16 kHz arrays
            wav = torchaudio.functional.resample(wav, audio["sampling_rate"], 16000)
        segments, _ = self._model.transcribe(wav.numpy(), beam_size=5, condition_on_previous_text=False)
        return {"text": " ".join(s.text.strip() for s in segments)}


def init(tokenizer_src: str, asr_model: str, asr_threads: int) -> None:
    """Pool initializer (runs in the child before any job)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # this process never touches a GPU
    _STATE.update(tokenizer_src=tokenizer_src, asr_model=asr_model, asr_threads=asr_threads)


def build(ref_audio: str, ref_text: str | None, preprocess_prompt: bool, sampling_rate: int):
    """-> (ref_audio_tokens as int64 ndarray (C, T), ref_text, ref_rms)."""
    import types

    import torch
    from omnivoice.models.omnivoice import OmniVoice

    if "tokenizer" not in _STATE:
        from transformers import HiggsAudioV2TokenizerModel
        _STATE["tokenizer"] = HiggsAudioV2TokenizerModel.from_pretrained(
            _STATE["tokenizer_src"], device_map="cpu", dtype=torch.float32).eval()
    if not ref_text and "asr" not in _STATE:
        _STATE["asr"] = FasterWhisperASR(_STATE["asr_model"], _STATE["asr_threads"])

    stub = types.SimpleNamespace(sampling_rate=sampling_rate, audio_tokenizer=_STATE["tokenizer"],
                                 _asr_pipe=_STATE.get("asr"))
    stub.transcribe = types.MethodType(OmniVoice.transcribe, stub)
    stub.load_asr_model = lambda *a, **k: None  # the ASR is set above whenever it is needed
    try:
        prompt = OmniVoice.create_voice_clone_prompt(
            stub, ref_audio=ref_audio, ref_text=(ref_text or None), preprocess_prompt=preprocess_prompt)
        return (prompt.ref_audio_tokens.detach().cpu().numpy(), prompt.ref_text, float(prompt.ref_rms))
    finally:
        try:  # hand the encode's GBs of fp32 activations back to the OS
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
