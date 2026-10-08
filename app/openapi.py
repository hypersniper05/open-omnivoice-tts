#!/usr/bin/env python3
"""OpenAPI 3.1 description of this server, served at GET /openapi.json.

Built per request from the live configuration, so the voice list, input limit
and custom-voice switch it documents are the ones in effect. GET /docs renders
it with Swagger UI.
"""

from __future__ import annotations

SWAGGER_UI_VERSION = "5"   # swagger-ui-dist major version, loaded from jsDelivr

DOCS_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>open-omnivoice-tts API</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@{v}/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@{v}/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({{ url: "/openapi.json", dom_id: "#swagger-ui",
                                   deepLinking: true, tryItOutEnabled: false }});
  </script>
</body>
</html>
""".format(v=SWAGGER_UI_VERSION)


def _ref(name: str) -> dict:
    return {"$ref": f"#/components/schemas/{name}"}


def _json(schema: dict, description: str = "OK") -> dict:
    return {"description": description, "content": {"application/json": {"schema": schema}}}


def _errors(*codes: int) -> dict:
    text = {
        400: "Bad request: unknown voice, invalid parameter, or unknown consent id.",
        401: "Missing or wrong API key (only when `api_key` is set).",
        403: "Custom voices are disabled (`allow_custom_voices` is false).",
        404: "Unknown custom voice or consent id.",
        413: "Input or upload too large.",
        429: "Busy: one generation running and the queue is full. Honour `Retry-After`.",
        503: "Model still loading, or out of GPU memory. Honour `Retry-After` when present.",
    }
    return {str(c): _json(_ref("Error"), text[c]) for c in codes}


def build_spec(*, aliases: list[str], models: list[str], formats: list[str], max_input_chars: int,
               custom_voices_enabled: bool, design_vocabulary: dict) -> dict:
    vocab = "; ".join(f"{k}: {', '.join(v)}" for k, v in design_vocabulary.items())
    custom_note = ("Enabled on this server." if custom_voices_enabled else
                   "**Disabled on this server**: set `allow_custom_voices` to true in "
                   "`app/openai_voices.json` (and an `api_key`) to use it.")
    voice_names = ", ".join(f"`{a}`" for a in aliases)
    custom_tag = ["Custom voices"]

    paths = {
        "/v1/audio/speech": {"post": {
            "tags": ["Speech"], "operationId": "createSpeech", "summary": "Generate speech",
            "description": "OpenAI-compatible text-to-speech. Returns the audio file in the requested format.",
            "requestBody": {"required": True, "content": {"application/json": {
                "schema": _ref("CreateSpeechRequest"),
                "example": {"model": "tts-1", "input": "Hello from OmniVoice.", "voice": aliases[0] if aliases else "alloy",
                            "response_format": "mp3"}}}},
            "responses": {
                "200": {"description": "The audio file.", "content": {
                    ct: {"schema": {"type": "string", "format": "binary"}}
                    for ct in ("audio/mpeg", "audio/ogg", "audio/aac", "audio/flac", "audio/wav", "audio/pcm")}},
                **_errors(400, 401, 404, 413, 429, 503)}}},
        "/v1/models": {"get": {
            "tags": ["Models"], "operationId": "listModels", "summary": "List models",
            "description": "OpenAI-style model list. Any model id is accepted by the speech endpoint.",
            "responses": {"200": _json(_ref("ModelList"))}}},
        "/v1/models/{model}": {"get": {
            "tags": ["Models"], "operationId": "retrieveModel", "summary": "Retrieve a model",
            "parameters": [{"name": "model", "in": "path", "required": True, "schema": {"type": "string"}}],
            "responses": {"200": _json(_ref("Model"))}}},
        "/v1/audio/voices": {
            "get": {
                "tags": ["Voices"], "operationId": "listVoices", "summary": "List voices",
                "description": "Extension: the alias map, the reference files in `Voices/` and the custom voices.",
                "responses": {"200": _json(_ref("VoiceList"))}},
            "post": {
                "tags": custom_tag, "operationId": "createVoice", "summary": "Create a custom voice",
                "description": ("OpenAI custom voices. From an audio sample (multipart: `name`, `audio_sample`, "
                                "`consent`), or from a text description (`type: prompt`, JSON or multipart). "
                                f"Descriptions use OmniVoice's voice-design attributes, one per category ({vocab}). "
                                + custom_note),
                "requestBody": {"required": True, "content": {
                    "multipart/form-data": {"schema": _ref("CreateVoiceMultipart")},
                    "application/json": {"schema": _ref("CreateVoicePrompt"),
                                         "example": {"type": "prompt", "name": "Narrator",
                                                     "prompt": "female, middle-aged, low pitch, british accent"}}}},
                "responses": {"200": _json(_ref("Voice")), **_errors(400, 401, 403, 413, 429, 503)}}},
        "/v1/audio/voices/{voice_id}": {"delete": {
            "tags": custom_tag, "operationId": "deleteVoice", "summary": "Delete a custom voice",
            "description": "Extension: OpenAI has no delete for voices.",
            "parameters": [{"name": "voice_id", "in": "path", "required": True, "schema": {"type": "string"}}],
            "responses": {"200": _json(_ref("Deleted")), **_errors(401, 403, 404)}}},
        "/v1/audio/voice_consents": {
            "post": {
                "tags": custom_tag, "operationId": "createVoiceConsent", "summary": "Upload a consent recording",
                "description": "The speaker's recorded consent, required before a voice is created from their sample. "
                               + custom_note,
                "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": _ref("CreateConsent")}}},
                "responses": {"200": _json(_ref("VoiceConsent")), **_errors(400, 401, 403, 413)}},
            "get": {
                "tags": custom_tag, "operationId": "listVoiceConsents", "summary": "List consent recordings",
                "parameters": [
                    {"name": "after", "in": "query", "schema": {"type": "string"}, "description": "Cursor: a consent id."},
                    {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20, "maximum": 100}}],
                "responses": {"200": _json(_ref("VoiceConsentList")), **_errors(401, 403)}}},
        "/v1/audio/voice_consents/{consent_id}": {
            "parameters": [{"name": "consent_id", "in": "path", "required": True, "schema": {"type": "string"}}],
            "get": {"tags": custom_tag, "operationId": "retrieveVoiceConsent", "summary": "Retrieve a consent",
                    "responses": {"200": _json(_ref("VoiceConsent")), **_errors(401, 403, 404)}},
            "post": {"tags": custom_tag, "operationId": "updateVoiceConsent", "summary": "Rename a consent",
                     "requestBody": {"required": True, "content": {"application/json": {"schema": {
                         "type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}}}},
                     "responses": {"200": _json(_ref("VoiceConsent")), **_errors(400, 401, 403, 404)}},
            "delete": {"tags": custom_tag, "operationId": "deleteVoiceConsent", "summary": "Delete a consent",
                       "responses": {"200": _json(_ref("Deleted")), **_errors(401, 403, 404)}}},
        "/api/meta": {"get": {
            "tags": ["Server"], "operationId": "serverMeta", "summary": "Server state",
            "description": "Readiness (`ready`), whether the model is in VRAM (`loaded`), idle time and VRAM figures. "
                           "Used by the container healthcheck.",
            "security": [], "responses": {"200": _json({"type": "object", "additionalProperties": True})}}},
    }

    schemas = {
        "CreateSpeechRequest": {
            "type": "object", "required": ["input"],
            "properties": {
                "model": {"type": "string", "default": "tts-1", "examples": models,
                          "description": "Accepted for compatibility; used only as a label."},
                "input": {"type": "string", "maxLength": max_input_chars,
                          "description": "Text to speak. Long text is generated paragraph by paragraph. "
                                         "Non-verbal tags such as `[laughter]` and `[sigh]` are supported."},
                "voice": {
                    "description": (f"A voice name: {voice_names}; a file in `Voices/` (with or without its "
                                    "extension); a custom voice id (`voice_...`); or `design:<attributes>`. "
                                    "May also be an object `{\"id\": ...}` as in OpenAI's custom voices."),
                    "oneOf": [{"type": "string"},
                              {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}]},
                "response_format": {"type": "string", "enum": formats, "default": "mp3"},
                "speed": {"type": "number", "minimum": 0.25, "maximum": 4.0, "default": 1.0},
                "instructions": {"type": "string",
                                 "description": "Accepted. Only OmniVoice voice-design attributes take effect "
                                                f"({vocab}); other text is ignored."}}},
        "Model": {"type": "object", "properties": {
            "id": {"type": "string"}, "object": {"type": "string", "const": "model"},
            "created": {"type": "integer"}, "owned_by": {"type": "string"}}},
        "ModelList": {"type": "object", "properties": {
            "object": {"type": "string", "const": "list"}, "data": {"type": "array", "items": _ref("Model")}}},
        "VoiceList": {"type": "object", "properties": {
            "aliases": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "maps_to": {"type": "string"}}}},
            "reference_files": {"type": "array", "items": {"type": "string"}},
            "formats": {"type": "array", "items": {"type": "string"}},
            "default_voice": {"type": "string"}, "default_format": {"type": "string"},
            "custom_voices": {"type": "array", "items": _ref("Voice")}}},
        "Voice": {"type": "object", "required": ["object", "id", "name", "type", "created_at"], "properties": {
            "object": {"type": "string", "const": "audio.voice"}, "id": {"type": "string"},
            "type": {"type": "string", "enum": ["audio_sample", "prompt"]}, "name": {"type": "string"},
            "created_at": {"type": "integer"}}},
        "CreateVoiceMultipart": {"type": "object", "required": ["name"], "properties": {
            "type": {"type": "string", "enum": ["audio_sample", "prompt"], "default": "audio_sample"},
            "name": {"type": "string"},
            "audio_sample": {"type": "string", "format": "binary",
                             "description": "Up to 10 MiB: mp3, wav, ogg, aac, flac, webm or mp4. For `audio_sample`."},
            "consent": {"type": "string", "description": "A consent id (`cons_...`). For `audio_sample`."},
            "prompt": {"type": "string", "description": "Voice-design attributes. For `prompt`."},
            "script_hint": {"type": "string", "description": "Text the new voice speaks in its reference clip. For `prompt`."}}},
        "CreateVoicePrompt": {"type": "object", "required": ["type", "name", "prompt"], "properties": {
            "type": {"type": "string", "const": "prompt"}, "name": {"type": "string"},
            "prompt": {"type": "string"}, "script_hint": {"type": "string"}}},
        "CreateConsent": {"type": "object", "required": ["name", "recording", "language"], "properties": {
            "name": {"type": "string"},
            "recording": {"type": "string", "format": "binary", "description": "Up to 10 MiB."},
            "language": {"type": "string", "description": "BCP 47 tag, e.g. `en-US`."}}},
        "VoiceConsent": {"type": "object", "required": ["object", "id", "name", "language", "created_at"],
                         "properties": {"object": {"type": "string", "const": "audio.voice_consent"},
                                        "id": {"type": "string"}, "name": {"type": "string"},
                                        "language": {"type": "string"}, "created_at": {"type": "integer"}}},
        "VoiceConsentList": {"type": "object", "required": ["object", "data", "has_more"], "properties": {
            "object": {"type": "string", "const": "list"}, "data": {"type": "array", "items": _ref("VoiceConsent")},
            "first_id": {"type": ["string", "null"]}, "last_id": {"type": ["string", "null"]},
            "has_more": {"type": "boolean"}}},
        "Deleted": {"type": "object", "required": ["id", "object", "deleted"], "properties": {
            "id": {"type": "string"}, "object": {"type": "string"}, "deleted": {"type": "boolean"}}},
        "Error": {"type": "object", "properties": {"error": {"type": "object", "properties": {
            "message": {"type": "string"}, "type": {"type": "string"}, "code": {"type": ["string", "null"]},
            "param": {"type": ["string", "null"]}}}}},
    }

    return {
        "openapi": "3.1.0",
        "info": {"title": "open-omnivoice-tts: OmniVoice OpenAI-compatible TTS API", "version": "1.0.0",
                 "description": ("Self-hosted [OmniVoice](https://github.com/k2-fsa/OmniVoice) speech server with "
                                 "OpenAI's `/v1/audio/speech` interface. Point any OpenAI client's `base_url` at "
                                 "`http://<host>:8008/v1`.")},
        "servers": [{"url": "/"}],
        "security": [{}, {"bearerAuth": []}],
        "tags": [{"name": "Speech"}, {"name": "Voices"}, {"name": "Custom voices"}, {"name": "Models"},
                 {"name": "Server"}],
        "paths": paths,
        "components": {
            "schemas": schemas,
            "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer",
                                               "description": "Required only when `api_key` is set in "
                                                              "`app/openai_voices.json`."}}},
    }
