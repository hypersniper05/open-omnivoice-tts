#!/usr/bin/env python3
"""OmniVoice OpenAI-compatible TTS server (stdlib HTTP, no Gradio).

Endpoints
---------
POST /v1/audio/speech        -> OpenAI speech request; returns audio bytes
GET  /v1/models              -> OpenAI-style model list
GET  /v1/models/<id>         -> one model object
GET  /v1/audio/voices        -> alias map, reference files in Voices/, custom voices
POST /v1/audio/voices        -> create a custom voice (OpenAI custom voices; see custom_voices.py)
DELETE /v1/audio/voices/<id> -> delete a custom voice (extension)
     /v1/audio/voice_consents[/<id>]  -> consent recordings: POST, GET, POST <id>, DELETE <id>
GET  /api/meta               -> JSON: readiness, loaded/offloaded, idle time, VRAM
                                (used by the container healthcheck)
GET  /docs                   -> interactive API docs (Swagger UI; loads its assets from jsDelivr)
GET  /openapi.json           -> OpenAPI 3.1 description of these endpoints (openapi.py)
GET  /                       -> JSON: service name and these endpoints

Run:
    .venv\\Scripts\\python.exe app\\server.py --port 8008
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import sys

# Put the project root on sys.path so the shared `core` package resolves whether
# this is launched as `python app/server.py` or from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import custom_voices
import openai_tts
import openapi
from core import ENGINE, GPUMemoryError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("omnivoice.server")

# Request limits. The engine runs one generation at a time and the rest wait on
# its lock; past MAX_QUEUE waiting requests the server answers 429 at once
# instead of stacking up minutes of work. Text longer than MAX_INPUT_CHARS
# (~25 min of audio at 20k) is refused with 413.
MAX_INPUT_CHARS = int(os.environ.get("OMNIVOICE_MAX_INPUT_CHARS") or 20000)
MAX_QUEUE = int(os.environ.get("OMNIVOICE_MAX_QUEUE") or 4)
_inflight = 0
_inflight_lock = threading.Lock()

MAX_BODY_BYTES = 25 * 1024 * 1024   # multipart uploads: two 10 MiB files + form overhead

ENDPOINTS = {
    "service": "open-omnivoice-tts: OmniVoice OpenAI-compatible TTS",
    "endpoints": ["POST /v1/audio/speech", "GET /v1/models", "GET /v1/audio/voices",
                  "POST /v1/audio/voices", "DELETE /v1/audio/voices/{id}",
                  "POST /v1/audio/voice_consents", "GET /v1/audio/voice_consents",
                  "GET|POST|DELETE /v1/audio/voice_consents/{id}", "GET /api/meta"],
    "docs": "/docs",
    "openapi": "/openapi.json",
}
CONSENTS = "/v1/audio/voice_consents"
VOICES = "/v1/audio/voices"


def _try_admit() -> bool:
    """Count a generation request in; False if 1 running + MAX_QUEUE waiting."""
    global _inflight
    with _inflight_lock:
        if _inflight > MAX_QUEUE:
            return False
        _inflight += 1
        return True


def _release() -> None:
    global _inflight
    with _inflight_lock:
        _inflight -= 1


def _too_long(text) -> str | None:
    if isinstance(text, str) and len(text) > MAX_INPUT_CHARS:
        return (f"Input is {len(text):,} characters; the limit is {MAX_INPUT_CHARS:,} "
                "(OMNIVOICE_MAX_INPUT_CHARS). Split it into several requests.")
    return None


BUSY = f"Server busy: 1 generation running and {MAX_QUEUE} waiting (OMNIVOICE_MAX_QUEUE). Retry shortly."
RETRY_AFTER_S = 5  # Retry-After on 429 / 503


class Handler(BaseHTTPRequestHandler):
    server_version = "OmniVoiceTTS/1.0"

    # -- helpers ----------------------------------------------------------
    def _send_json(self, obj, status=200, extra_headers=None):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, status=200, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(data)

    # -- CORS preflight (browser-based front ends) ------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, fmt, *args):  # quieter logging
        logger.info("%s - %s", self.address_string(), fmt % args)

    # -- GET --------------------------------------------------------------
    def do_GET(self):
        route = urlparse(self.path).path

        if route == "/":
            return self._send_json(ENDPOINTS)
        if route == "/openapi.json":
            return self._send_json(openapi.build_spec(
                aliases=openai_tts.alias_names(), models=openai_tts.ADVERTISED_MODELS,
                formats=list(openai_tts.FORMAT_CONTENT_TYPES), max_input_chars=MAX_INPUT_CHARS,
                custom_voices_enabled=openai_tts.custom_voices_allowed(),
                design_vocabulary=openai_tts.VOICE_DESIGN_CATEGORIES))
        if route in ("/docs", "/docs/"):
            return self._send_bytes(openapi.DOCS_HTML.encode("utf-8"), "text/html; charset=utf-8")
        if route == "/api/meta":
            return self._send_json(ENGINE.meta())
        if route == "/v1/models":
            return self._send_json(openai_tts.list_models())
        if route.startswith("/v1/models/"):
            return self._send_json(openai_tts.model_object(route[len("/v1/models/"):]))
        if route in (VOICES, "/v1/voices"):
            return self._send_json(openai_tts.list_voices())
        if route == CONSENTS or route.startswith(CONSENTS + "/"):
            qs = parse_qs(urlparse(self.path).query)
            return self._custom(lambda: custom_voices.list_consents(
                (qs.get("after") or [None])[0], (qs.get("limit") or [None])[0])
                if route == CONSENTS else custom_voices.get_consent(route[len(CONSENTS) + 1:]))

        return self._send_json(openai_tts.error_response("Not found.", 404, "not_found"), 404)

    # -- POST -------------------------------------------------------------
    def do_POST(self):
        route = urlparse(self.path).path
        if route == "/v1/audio/speech":
            return self._handle_openai_speech()
        if route == VOICES:
            return self._custom(self._create_voice)
        if route == CONSENTS:
            return self._custom(lambda: custom_voices.create_consent(*self._form()))
        if route.startswith(CONSENTS + "/"):
            return self._custom(lambda: custom_voices.update_consent(route[len(CONSENTS) + 1:], self._json()))
        return self._send_json(openai_tts.error_response("Not found.", 404, "not_found"), 404)

    # -- DELETE -----------------------------------------------------------
    def do_DELETE(self):
        route = urlparse(self.path).path
        if route.startswith(CONSENTS + "/"):
            return self._custom(lambda: custom_voices.delete_consent(route[len(CONSENTS) + 1:]))
        if route.startswith(VOICES + "/"):
            return self._custom(lambda: custom_voices.delete_voice(route[len(VOICES) + 1:]))
        return self._send_json(openai_tts.error_response("Not found.", 404, "not_found"), 404)

    # -- custom voices (OpenAI POST /v1/audio/voices + voice consents) -----
    def _custom(self, action):
        """Auth + feature switch + error mapping around one custom-voice call."""
        if not openai_tts.check_auth(self.headers.get("Authorization")):
            return self._send_json(
                openai_tts.error_response("Incorrect API key provided.", 401, "invalid_api_key"), 401)
        if not openai_tts.custom_voices_allowed():
            return self._send_json(openai_tts.error_response(
                "Custom voices are disabled on this server (allow_custom_voices in app/openai_voices.json).",
                403, "custom_voices_disabled"), 403)
        try:
            return self._send_json(action())
        except (custom_voices.CustomVoiceError, openai_tts.TTSRequestError) as e:
            retry = {"Retry-After": RETRY_AFTER_S} if e.status in (429, 503) else None
            return self._send_json(openai_tts.error_response(e.message, e.status, e.code), e.status, retry)
        except GPUMemoryError as e:
            return self._send_json(openai_tts.error_response(str(e), 503, "gpu_memory_limit"), 503,
                                   {"Retry-After": RETRY_AFTER_S})
        except Exception as e:
            logger.exception("custom voice request failed")
            return self._send_json(
                openai_tts.error_response(f"{type(e).__name__}: {e}", 500, "server_error"), 500)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise custom_voices.CustomVoiceError(
                f"Request body is {length / 2**20:.1f} MiB; the limit is {MAX_BODY_BYTES // 2**20} MiB.", 413)
        return self.rfile.read(length) if length else b""

    def _json(self) -> dict:
        try:
            body = json.loads(self._body() or b"{}")
        except ValueError as e:
            raise custom_voices.CustomVoiceError(f"Invalid JSON body: {e}")
        if not isinstance(body, dict):
            raise custom_voices.CustomVoiceError("The JSON body must be an object.")
        return body

    def _form(self) -> tuple[dict, dict]:
        ctype = self.headers.get("Content-Type") or ""
        if not ctype.lower().startswith("multipart/form-data"):
            raise custom_voices.CustomVoiceError("This endpoint expects multipart/form-data.")
        return custom_voices.parse_multipart(ctype, self._body())

    def _create_voice(self) -> dict:
        ctype = (self.headers.get("Content-Type") or "").lower()
        if ctype.startswith("multipart/form-data"):
            fields, files = self._form()
        else:
            fields, files = {k: v if isinstance(v, str) else str(v) for k, v in self._json().items()}, {}
        kind = (fields.get("type") or "audio_sample").strip()
        if kind == "audio_sample":
            if not ctype.startswith("multipart/form-data"):
                raise custom_voices.CustomVoiceError(
                    "Creating a voice from an audio sample needs multipart/form-data "
                    "(name, audio_sample, consent).")
            return custom_voices.create_voice_from_sample(fields, files)
        if kind != "prompt":
            raise custom_voices.CustomVoiceError(f"Unknown voice type {kind!r}; use 'audio_sample' or 'prompt'.")
        # A prompt voice runs one OmniVoice generation: same admission as speech.
        if not _try_admit():
            raise custom_voices.CustomVoiceError(BUSY, 429, "rate_limit_exceeded")
        try:
            return custom_voices.create_voice_from_prompt(fields, openai_tts.design_synth)
        finally:
            _release()

    # -- OpenAI /v1/audio/speech -----------------------------------------
    def _handle_openai_speech(self):
        if not openai_tts.check_auth(self.headers.get("Authorization")):
            return self._send_json(
                openai_tts.error_response("Incorrect API key provided.", 401, "invalid_api_key"), 401)

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            return self._send_json(openai_tts.error_response(f"Invalid JSON body: {e}"), 400)

        too_long = _too_long(payload.get("input", payload.get("text", "")))
        if too_long:
            return self._send_json(openai_tts.error_response(too_long, 413, "input_too_long"), 413)
        if not _try_admit():
            return self._send_json(openai_tts.error_response(BUSY, 429, "rate_limit_exceeded"), 429,
                                   {"Retry-After": RETRY_AFTER_S})

        try:
            data, ctype = openai_tts.synthesize_speech(payload)
        except openai_tts.TTSRequestError as e:
            return self._send_json(openai_tts.error_response(e.message, e.status, e.code), e.status)
        except GPUMemoryError as e:
            return self._send_json(openai_tts.error_response(str(e), 503, "gpu_memory_limit"), 503,
                                   {"Retry-After": RETRY_AFTER_S})
        except Exception as e:
            logger.exception("speech synthesis failed")
            return self._send_json(
                openai_tts.error_response(f"{type(e).__name__}: {e}", 500, "server_error"), 500)
        finally:
            _release()

        return self._send_bytes(data, ctype, extra_headers={"Cache-Control": "no-store"})


def _reachable_urls(host: str, port: int) -> list[tuple[str, str]]:
    """Return [(label, base_url)] the server can be reached at, incl. Tailscale."""
    urls: list[tuple[str, str]] = [("local", f"http://127.0.0.1:{port}/v1")]
    ips: set[str] = set()

    # Tailscale IP, if the CLI is installed.
    for cmd in (["tailscale", "ip", "-4"], ["tailscale.exe", "ip", "-4"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
            for line in out.stdout.splitlines():
                ip = line.strip()
                if ip:
                    urls.append(("tailscale", f"http://{ip}:{port}/v1"))
            if out.returncode == 0:
                break
        except Exception:
            pass

    # All local IPv4 addresses (LAN + tailnet 100.64.0.0/10 fallback).
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    for ip in sorted(ips):
        if ip.startswith("127."):
            continue
        label = "tailscale" if ip.startswith("100.") else "lan"
        u = f"http://{ip}:{port}/v1"
        if u not in [x[1] for x in urls]:
            urls.append((label, u))
    return urls


def main(argv=None):
    parser = argparse.ArgumentParser(description="OmniVoice OpenAI-compatible TTS server.")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address. 0.0.0.0 (default) = reachable over LAN / Tailscale; "
                             "use 127.0.0.1 to restrict to this machine.")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto).")
    parser.add_argument("--load-asr", action="store_true",
                        help="Preload Whisper ASR for auto-transcribing reference audio.")
    parser.add_argument("--idle-ttl", type=float, default=float(os.environ.get("OMNIVOICE_IDLE_TTL") or 600),
                        help="Offload the model from VRAM after this many seconds with no "
                             "generation (it reloads on the next request). 0 disables. Default: 600.")
    args = parser.parse_args(argv)

    if args.device:
        ENGINE.device = args.device
    if args.load_asr:
        ENGINE.want_asr = True
    ENGINE.idle_ttl = max(0.0, args.idle_ttl)

    # Tell the engine's second-instance guard which port is ours, so it does not
    # mistake this server (answering on that port) for another OmniVoice.
    os.environ["OMNIVOICE_SELF_PORT"] = str(args.port)
    # Load the model in the background; /api/meta reports `ready` once it is up
    # and /v1/audio/speech answers 503 until then.
    threading.Thread(target=ENGINE.load, daemon=True).start()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    print("\n" + "=" * 64)
    print("  open-omnivoice-tts: OmniVoice OpenAI-compatible TTS API (model loads in background)")
    print("  base_url:")
    for label, u in _reachable_urls(args.host, args.port):
        print(f"    [{label:>9}]  {u}")
    print("  example: curl <base_url>/audio/speech -H \"Content-Type: application/json\" \\")
    print("             -d '{\"model\":\"tts-1\",\"voice\":\"alloy\",\"input\":\"Hello.\"}' --output hello.mp3")
    if ENGINE.idle_ttl and ENGINE.idle_ttl > 0:
        print(f"  Idle offload: model unloads from VRAM after {ENGINE.idle_ttl:.0f}s idle, reloads on next request.")
    print("=" * 64 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
