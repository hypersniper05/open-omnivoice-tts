# OmniVoice OpenAI-compatible TTS image: torch 2.14.1 + FlashInfer attention.
# Multi-stage so no CUDA toolkit reaches the runtime image:
#   stage "aot"     -devel base (nvcc): compiles the FlashInfer kernels OmniVoice
#                   uses, ahead of time for one GPU architecture (docker/flashinfer_aot.py).
#                   FLASHINFER_CUDA_ARCH_LIST picks it: 8.6 = RTX 30, 8.9 = RTX 40,
#                   12.0 = RTX 50. The start scripts read it from the GPU and pass it
#                   as a build argument.
#   final stage     -runtime base: app + those compiled .so files only
# FlashInfer is switched on by OMNIVOICE_FLASHINFER=1 (docker-compose.yml): about
# 25% faster than torch.compile (RTF 0.032 vs 0.043 on a 115 s text). New-voice
# prompts are then built in a separate process (core/prompt_worker.py), because
# FlashInfer and CTranslate2 in one process segfaulted. FlashInfer + compile
# needs g++, which this image does not have: use one or the other.

# ---------------------------------------------------------------- stage 1: AOT kernels
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel AS aot
ARG FLASHINFER_CUDA_ARCH_LIST=8.6
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    MAX_JOBS=2 \
    FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
RUN pip install --no-cache-dir "flashinfer-python==0.7.0.post1" ninja
COPY docker/flashinfer_aot.py /tmp/flashinfer_aot.py
RUN python /tmp/flashinfer_aot.py

# ---------------------------------------------------------------- stage 2: runtime
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# System Python (Ubuntu 24.04, PEP 668) -- single-purpose container.
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    MAX_JOBS=2

# torchaudio 2.11.0+cu130 already ships in this base and works on torch 2.14.1
# (stable ABI; resample verified), so it is not reinstalled.
# Versions pinned to the set a clean install was tested with (2026-10-08).
RUN pip install --no-cache-dir \
        "transformers==5.18.0" \
        "tokenizers==0.23.2" \
        "accelerate==1.15.0" \
        "safetensors==0.8.0" \
        "soundfile==0.14.0" \
        "pydub==0.25.1" \
        "huggingface_hub==1.33.0" \
        "faster-whisper==1.2.1" \
        "flashinfer-python==0.7.0.post1"

# The AOT kernels from stage 1. With JIT disabled, a kernel that is not among
# them fails at once with its name (MissingJITCacheError) instead of looking for
# an nvcc this image does not have.
COPY --from=aot /usr/local/lib/python3.12/dist-packages/flashinfer/data/aot/ \
                /usr/local/lib/python3.12/dist-packages/flashinfer/data/aot/
ENV FLASHINFER_DISABLE_JIT=1

COPY vendor/omnivoice-src/ /app/vendor/omnivoice-src/
RUN pip install --no-cache-dir --no-deps -e /app/vendor/omnivoice-src

COPY core/ /app/core/
COPY app/ /app/app/

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8008

# Recorded on the image so the start scripts can tell when the GPU needs a rebuild.
# Last, so a different architecture rebuilds only the kernels, not the packages.
ARG FLASHINFER_CUDA_ARCH_LIST=8.6
LABEL omnivoice.flashinfer_arch=${FLASHINFER_CUDA_ARCH_LIST}

CMD ["python", "app/server.py", "--host", "0.0.0.0", "--port", "8008"]
