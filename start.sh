#!/usr/bin/env bash
# Start open-omnivoice-tts. Creates the local settings on the first run, builds the image if it does not
# exist yet (or was built for another GPU architecture), starts the container and waits until the
# model is loaded. The first start downloads the model, about 3 GB.
# --build rebuilds the image even if it exists.
set -euo pipefail
BUILD=0
[ "${1:-}" = "--build" ] && BUILD=1
cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. See README.md -> Requirements." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running (or this user may not use it). Start Docker and try again." >&2
  exit 1
fi

env_get() {  # env_get NAME DEFAULT
  local v
  v=$(sed -nE "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*([^#]*).*/\1/p" .env | head -n1 | tr -d '[:space:]')
  echo "${v:-$2}"
}
env_set() {  # env_set NAME VALUE (replaces the line, or appends it)
  awk -v k="$1" -v v="$2" 'BEGIN{done=0} $0 ~ "^[[:space:]]*"k"[[:space:]]*=" {print k"="v; done=1; next} {print} END{if(!done) print k"="v}' .env > .env.tmp
  mv .env.tmp .env
}

# OmniVoice's source is a git submodule: a clone without --recursive leaves its folder empty.
if [ ! -f vendor/omnivoice-src/pyproject.toml ]; then
  if command -v git >/dev/null 2>&1 && [ -e .git ]; then
    echo "==> fetching the OmniVoice source (git submodule)"
    git submodule update --init --recursive
  else
    echo "vendor/omnivoice-src is empty. Clone the repository with: git clone --recursive https://github.com/hypersniper05/open-omnivoice-tts.git" >&2
    exit 1
  fi
fi

# Local settings (not tracked by git): create them from the tracked examples on the first run.
[ -f .env ] || { cp .env.example .env; echo "==> created .env from .env.example (GPU, port, ...)"; }
[ -f app/openai_voices.json ] || { cp app/openai_voices.example.json app/openai_voices.json; echo "==> created app/openai_voices.json from the example (voice names, API key, ...)"; }
mkdir -p model Voices/custom output/prompt_cache output/voice_cache output/inductor_cache

PORT=$(env_get OMNIVOICE_PORT 8008)
GPU=$(env_get OMNIVOICE_GPU 0)
DEVICE=$(env_get OMNIVOICE_DEVICE cuda:0)
ARCH=$(env_get FLASHINFER_CUDA_ARCH_LIST "")

# The GPU and its architecture: the FlashInfer kernels are built for exactly one.
if [ "$DEVICE" != "cpu" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    if INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader -i "$GPU" 2>/dev/null | head -n1) && [ -n "$INFO" ]; then
      NAME=$(echo "$INFO" | cut -d, -f1 | sed 's/^ *//;s/ *$//')
      CAP=$(echo "$INFO" | cut -d, -f2 | tr -d '[:space:]')
      echo "==> GPU $GPU: $NAME (compute capability $CAP)"
      if [ -z "$ARCH" ]; then ARCH=$CAP; env_set FLASHINFER_CUDA_ARCH_LIST "$ARCH"; echo "    set FLASHINFER_CUDA_ARCH_LIST=$ARCH in .env"; fi
    else
      echo "    warning: nvidia-smi does not know GPU '$GPU' (OMNIVOICE_GPU in .env)" >&2
    fi
  else
    echo "    warning: nvidia-smi was not found. Is the NVIDIA driver installed?" >&2
  fi
else
  echo "==> CPU mode (OMNIVOICE_DEVICE=cpu): more than 100 times slower than a GPU"
fi
ARCH=${ARCH:-8.6}

# Build first, while any running instance keeps serving.
IMAGE=omnivoice-tts:latest
BUILT_ARCH=""
HAVE_IMAGE=0
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  HAVE_IMAGE=1
  BUILT_ARCH=$(docker image inspect -f '{{index .Config.Labels "omnivoice.flashinfer_arch"}}' "$IMAGE" 2>/dev/null || true)
fi
if [ "$BUILD" = 1 ] || [ "$HAVE_IMAGE" = 0 ] || [ "$BUILT_ARCH" != "$ARCH" ]; then
  [ "$HAVE_IMAGE" = 1 ] && [ "$BUILD" = 0 ] && echo "==> the image has kernels for '$BUILT_ARCH', this GPU needs $ARCH"
  echo "==> building the image (the first build downloads about 15 GB and takes 10 to 30 minutes)"
  docker compose build
else
  echo "==> using the existing image $IMAGE (./start.sh --build rebuilds it)"
fi

# Replace the running container with the new one.
docker compose down --remove-orphans >/dev/null 2>&1 || true
docker rm -f omnivoice-tts >/dev/null 2>&1 || true
docker compose up -d --no-build

FROM_HUB=0; [ -f model/config.json ] || FROM_HUB=1   # no local copy: the model comes from Hugging Face
if [ "$FROM_HUB" = 1 ]; then echo "==> waiting for the model (the first start downloads about 3 GB)"; else echo "==> waiting for the model (from ./model)"; fi
START=$(date +%s)
LAST=0
while true; do
  body=$(curl -fsS "http://localhost:${PORT}/api/meta" 2>/dev/null || true)
  if [ -n "$body" ]; then
    printf '%s' "$body" | grep -q '"ready": true' && break
    err=$(printf '%s' "$body" | sed -nE 's/.*"error": "([^"]*)".*/\1/p')
    if [ -n "$err" ]; then
      echo "Startup failed: $err" >&2
      echo "Details: docker compose logs --tail 80" >&2
      exit 1
    fi
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^omnivoice-tts$'; then
    echo "The container stopped. Details: docker compose logs --tail 80" >&2
    exit 1
  fi
  ELAPSED=$(( $(date +%s) - START ))
  if [ $(( ELAPSED - LAST )) -ge 30 ]; then
    NOTE=""
    if [ "$FROM_HUB" = 1 ]; then
      SIZE=$(docker exec omnivoice-tts du -sh /app/.hf_cache/hub 2>/dev/null | cut -f1 || true)
      [ -n "$SIZE" ] && NOTE=", downloaded $SIZE of about 3.1G"
    fi
    echo "    still loading (${ELAPSED} s${NOTE})"
    LAST=$ELAPSED
  fi
  sleep 3
done

SECS=$(( $(date +%s) - START ))
if grep -qE '"api_key": *"[^"]+"' app/openai_voices.json; then AUTH="API key required"; else AUTH="no API key"; fi
echo
echo "Ready after ${SECS} s. OpenAI-compatible API (${AUTH}):"
echo "    base URL   http://localhost:${PORT}/v1"
echo "    API docs   http://localhost:${PORT}/docs   (try POST /v1/audio/speech there)"
echo "From other machines use this computer's IP address instead of localhost."
echo "Logs: docker compose logs -f     Stop: ./stop.sh"
