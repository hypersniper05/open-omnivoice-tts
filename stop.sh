#!/usr/bin/env bash
# Stop open-omnivoice-tts.
cd "$(dirname "$0")"
docker compose down
