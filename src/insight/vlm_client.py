"""Provider-agnostic VLM client (Vertex AI / OpenAI / Gemini API).

Single shared implementation used by both the InSight sim flywheel
(``sim/libero_flywheel/vlm_flywheel``) and the real-world
xArm pipeline (``real/entry``). Replaces the duplicated clients in
``vlm_flywheel/vlm.py`` and ``real/entry/vlm_check.py``.

Public surface:
    set_provider(provider)        — switch active provider/model
    set_research_context(text)    — env-specific system-prompt preamble
    get_model()                   — name of the currently active model
    chat(messages, ...)           — bare chat completion call
    with_images(prompt, imgs, ...)— chat with attached images + system prompt
    parse_json(text)              — robust JSON extractor for VLM responses
    encode_image_b64(np_image)    — PNG -> base64 helper

State is kept at module level. Callers should call ``set_provider`` once at
startup; ``set_research_context`` is optional but recommended (defaults to
empty, which can trigger refusals on real-world images that look ambiguous).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time

import numpy as np
from openai import OpenAI
from PIL import Image


# Models that need ``max_completion_tokens`` instead of ``max_tokens``.
_NEW_TOKEN_PARAM_MODELS = {
    "gpt-5", "gpt-5.2", "gpt-5-mini",
    "o1", "o3", "o3-mini", "o4-mini",
    "google/gemini-3-flash-preview", "google/gemini-3-pro-preview", "google/gemini-3.1-pro-preview",
    "gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-3.1-pro-preview",
    "google/gemini-robotics-er-1.6-preview", "gemini-robotics-er-1.6-preview",
}

_PROVIDERS = {
    "gpt": {
        "kind": "openai",
        "model": "gpt-5.2",
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
    },
    "gemini": {
        "kind": "vertex",
        "model": "google/gemini-3-flash-preview",
        "base_url": "https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/global/endpoints/openapi",
        "project": os.environ.get("VERTEX_PROJECT", "gcp-maggie"),
    },
    "gemini-robotics": {
        "kind": "openai",  # Gemini API is OpenAI-compatible
        "model": "gemini-robotics-er-1.6-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GEMINI_API_KEY",
    },
}

# Mutable state — modified only via set_provider / set_research_context.
_active: dict | None = None
_model: str | None = None
_client: OpenAI | None = None
_research_context: str = ""

# Vertex access-token cache — avoid rebuilding the OpenAI client on every chat
# call. Tokens have ~1h TTL; we refresh ~5min before expiry to be safe.
_vertex_token_cache: tuple[str, float] | None = None  # (token, expires_at_epoch)
_VERTEX_TOKEN_REFRESH_BEFORE = 300  # seconds

# Image observers — invoked with the list of images each time `with_images` is
# called, before the VLM request goes out. Used to add side effects (e.g. an
# on-screen display, frame logging) without subclassing or monkey-patching.
_image_observers: list = []


def set_provider(provider: str) -> None:
    """Configure the active provider. Syntax: ``"gemini"``, ``"gpt"``,
    ``"gemini-robotics"``, or ``"<name>:<model_override>"``."""
    global _active, _model, _client
    parts = provider.split(":", 1)
    name = parts[0].lower()
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown VLM provider '{name}'. Choose from: {list(_PROVIDERS.keys())}")
    _active = _PROVIDERS[name]
    _model = parts[1] if len(parts) > 1 else _active["model"]
    _client = None
    logging.info("VLM provider: %s, model: %s", name, _model)


def set_research_context(text: str) -> None:
    """Set the environment-specific safety preamble prepended to every system
    prompt. Sim and real have different versions (sim mentions MuJoCo renders;
    real mentions RealSense + xArm). Empty by default."""
    global _research_context
    _research_context = text


def get_model() -> str:
    if _model is None:
        raise RuntimeError("Call insight.vlm_client.set_provider(...) first.")
    return _model


def get_active() -> dict:
    if _active is None:
        raise RuntimeError("Call insight.vlm_client.set_provider(...) first.")
    return _active


def _vertex_token() -> str:
    """Return a cached Vertex access token, refreshing only when near-expiry.

    Tokens have ~1h TTL; refreshing on every chat call adds ~100-300ms of
    google-auth overhead per request. With this cache, the typical inference
    loop refreshes once at startup and again ~55 minutes later.
    """
    global _vertex_token_cache
    now = time.time()
    if _vertex_token_cache is not None:
        token, expiry = _vertex_token_cache
        if now < expiry - _VERTEX_TOKEN_REFRESH_BEFORE:
            return token
    import google.auth
    import google.auth.transport.requests
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    if creds.expiry is not None:
        from datetime import timezone
        expiry_epoch = creds.expiry.replace(tzinfo=timezone.utc).timestamp()
    else:
        expiry_epoch = now + 3300  # default 55 min if expiry unknown
    _vertex_token_cache = (creds.token, expiry_epoch)
    return creds.token


def _get_client() -> OpenAI:
    """Return the active OpenAI client. For Vertex, reuses the cached client
    until the underlying access token rotates (every ~55 min)."""
    global _client
    active = get_active()
    if active["kind"] == "vertex":
        token = _vertex_token()
        # Rebuild the OpenAI client only when the token actually changed.
        if _client is None or getattr(_client, "_insight_token", None) != token:
            base_url = active["base_url"].format(project=active.get("project", ""))
            _client = OpenAI(base_url=base_url, api_key=token)
            # Stash the token on the client so we can detect rotations cheaply.
            _client._insight_token = token  # type: ignore[attr-defined]
        return _client
    if _client is not None:
        return _client
    if active["kind"] == "openai":
        env_key = active.get("env_key")
        api_key = os.environ.get(env_key, "") if env_key else ""
        if env_key and not api_key:
            raise ValueError(f"Set {env_key} for provider model {_model}")
        if active.get("base_url"):
            _client = OpenAI(base_url=active["base_url"], api_key=api_key or None)
        elif api_key:
            _client = OpenAI(api_key=api_key)
        else:
            _client = OpenAI()
        return _client
    raise ValueError(f"Unknown provider kind: {active['kind']}")


def chat(messages: list, max_tokens: int = 500, temperature: float = 0.3, retries: int = 5) -> str:
    """Bare chat completion call. Retries on refusals."""
    client = _get_client()
    model = get_model()
    kwargs: dict = {}
    if model in _NEW_TOKEN_PARAM_MODELS:
        kwargs["max_completion_tokens"] = max(max_tokens * 8, 4096)
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    for attempt in range(retries + 1):
        t0 = time.time()
        logging.info("  [VLM] Calling %s (max_tokens=%s)...", model,
                     kwargs.get("max_completion_tokens", kwargs.get("max_tokens")))
        try:
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as e:
            logging.error("VLM API error: %s", e)
            raise
        elapsed = time.time() - t0
        choice = response.choices[0]
        # Gemini occasionally returns choice.message=None on HTTP 200
        # (safety filter, empty response struct). Treat as transient and retry.
        message = choice.message
        if message is None:
            logging.warning("  [VLM] Response in %.1fs but message is None (finish_reason=%s)",
                            elapsed, choice.finish_reason)
            if attempt < retries:
                time.sleep(1)
                continue
            raise ValueError(f"VLM returned null message (finish_reason={choice.finish_reason})")
        content = message.content
        logging.info("  [VLM] Response in %.1fs (%d chars)", elapsed, len(content) if content else 0)
        if content:
            return content
        refusal = getattr(message, "refusal", None)
        if refusal and attempt < retries:
            logging.warning("  VLM refused (attempt %d/%d): %s. Retrying...",
                            attempt + 1, retries + 1, refusal)
            time.sleep(1)
            continue
        raise ValueError(f"VLM empty response (finish_reason={choice.finish_reason}, refusal={refusal})")
    raise ValueError("VLM refused after all retries")


def add_image_observer(fn) -> None:
    """Register a callback invoked on every ``with_images`` call.

    The callback receives the list of ``np.ndarray`` images and may produce any
    side effect (display, logging, recording). Exceptions raised inside the
    callback are caught and logged so they cannot break the VLM call.
    """
    _image_observers.append(fn)


def remove_image_observer(fn) -> None:
    """Unregister a previously-added image observer. No-op if not registered."""
    try:
        _image_observers.remove(fn)
    except ValueError:
        pass


def clear_image_observers() -> None:
    """Drop every registered image observer. Useful for tests."""
    _image_observers.clear()


def _notify_image_observers(images: list) -> None:
    for fn in _image_observers:
        try:
            fn(images)
        except Exception as e:
            logging.warning("image observer %r raised: %s", fn, e)


def encode_image_b64(img: np.ndarray) -> str:
    """Encode an HxWx3 uint8 RGB array as a base64 PNG string."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def with_images(prompt: str, images: list[np.ndarray], max_tokens: int = 500, system: str = "") -> str:
    """Send a chat with attached images. The current research_context (set via
    ``set_research_context``) is prepended to ``system``.

    Image observers (registered via ``add_image_observer``) are notified
    synchronously before the request is built; they receive the original
    ``images`` list and run their side effects (display, logging, etc.).

    Parameter order ``(prompt, images, max_tokens, system)`` matches the
    historical ``vlm_flywheel.vlm_with_images`` signature so positional
    callers keep working unchanged. Internal callers should prefer keyword args.
    """
    _notify_image_observers(images)
    if _research_context and system:
        full_system = f"{_research_context}\n\n{system}"
    else:
        full_system = _research_context or system
    content: list = [{"type": "text", "text": prompt}]
    for img in images:
        b64 = encode_image_b64(img)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": content},
    ]
    return chat(messages, max_tokens=max_tokens)


def parse_json(text: str) -> dict:
    """Extract JSON from a VLM response, handling code fences and common
    artifacts (``true/false`` placeholders, leading ``+`` signs, embedded
    newlines)."""
    if not text:
        raise ValueError("VLM returned empty response")
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    text = re.sub(r"\btrue\s*/\s*false\b", "true", text)
    text = re.sub(r"\btrue\s+or\s+false\b", "true", text)
    text = re.sub(r"(?<=[\[,:\s])\+(?=\d)", "", text)
    text = text.replace("\n", " ").replace("\r", " ")
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            result = json.loads(text[start:end + 1])
        else:
            raise
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object, got {type(result).__name__}: {result!r:.100}")
    return result
