"""Unified OpenAI Responses API client (SSE streaming + configurable reasoning effort) for Paper2SVG-ultra."""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any, Optional, Union

import httpx
from PIL import Image

from .config import Settings

ImageInput = Union[Path, str, Image.Image]
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 520, 522, 524}
HARD_QUOTA_MARKERS = ("DAILY_LIMIT_EXCEEDED", "USAGE_LIMIT_EXCEEDED", "insufficient_quota", "quota_exceeded")


def encode_image_to_data_url(image: ImageInput, max_side: int = 1800) -> str:
    """Load an image (path or PIL.Image), resize safely if oversized, and encode as PNG data URL."""
    if isinstance(image, (str, Path)):
        img = Image.open(image)
        img.load()
    elif isinstance(image, Image.Image):
        img = image.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class ResponsesAPIClient:
    """Resilient SSE-streaming client targeting OpenAI Responses API (/v1/responses).

    - Streams continuously via `stream=True` to eliminate Cloudflare 100s HTTP 524 timeouts.
    - Supports `reasoning: {"effort": "xhigh"}` with automatic graceful fallback if a gateway rejects it.
    - Fails fast on hard daily quota errors instead of wasting retry cycles.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.api_key:
            raise ValueError("OPENAI_API_KEY is missing. Please set it in .env or environment variables.")

    def _apply_reasoning(self, payload: dict[str, Any], *, temperature: Optional[float] = None) -> dict[str, Any]:
        effort = (self.settings.reasoning_effort or "").strip().lower()
        if effort and effort != "none":
            payload["reasoning"] = {"effort": effort}
            # OpenAI reasoning models generally disallow explicit temperature when reasoning effort is active
            payload.pop("temperature", None)
        elif temperature is not None:
            payload["temperature"] = temperature
        return payload

    def _post_responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.base_url}/responses"
        active_payload = {**payload, "stream": True}
        last_err: Optional[Exception] = None

        timeout = httpx.Timeout(
            connect=30.0,
            read=max(self.settings.timeout_seconds, 600.0),
            write=120.0,
            pool=30.0,
        )

        for attempt in range(1, self.settings.max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    with client.stream(
                        "POST", url, headers=self.settings.default_headers, json=active_payload
                    ) as resp:
                        if resp.status_code != 200:
                            body = resp.read().decode("utf-8", errors="replace")

                            # Fail fast on hard daily quota exhaustion
                            if any(marker in body for marker in HARD_QUOTA_MARKERS):
                                raise RuntimeError(
                                    f"API 额度已达每日上限 (HTTP {resp.status_code}): {body[:400]} "
                                    "— 请等待额度刷新或在 .env 中更换新的 OPENAI_API_KEY。"
                                )

                            # Graceful fallback if a model/proxy rejects reasoning.effort='xhigh'
                            if resp.status_code == 400 and "reasoning" in active_payload and (
                                "reasoning" in body.lower() or "effort" in body.lower()
                            ):
                                cur_eff = active_payload["reasoning"].get("effort")
                                if cur_eff == "xhigh":
                                    active_payload["reasoning"] = {"effort": "high"}
                                    continue
                                active_payload.pop("reasoning", None)
                                continue

                            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < self.settings.max_retries:
                                time.sleep(min(5.0 * attempt, 20.0))
                                continue
                            raise RuntimeError(f"Responses API HTTP {resp.status_code}: {body[:800]}")

                        ctype = (resp.headers.get("content-type") or "").lower()
                        if "text/event-stream" not in ctype:
                            raw_body = resp.read().decode("utf-8", errors="replace")
                            return json.loads(raw_body)

                        text_deltas: list[str] = []
                        done_items: list[dict[str, Any]] = []
                        final_response: Optional[dict[str, Any]] = None
                        stream_error: Optional[str] = None

                        for line in resp.iter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if not data_str or data_str == "[DONE]":
                                continue
                            try:
                                evt = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            evt_type = evt.get("type", "")
                            if evt_type == "response.output_text.delta":
                                delta = evt.get("delta")
                                if isinstance(delta, str):
                                    text_deltas.append(delta)
                            elif evt_type == "response.output_item.done":
                                item = evt.get("item")
                                if isinstance(item, dict):
                                    done_items.append(item)
                            elif evt_type == "response.completed":
                                r_obj = evt.get("response")
                                if isinstance(r_obj, dict):
                                    final_response = r_obj
                            elif evt_type in ("response.failed", "error"):
                                err_obj = evt.get("error") or evt.get("response", {}).get("error") or evt
                                stream_error = str(err_obj)

                        if stream_error and not final_response:
                            raise RuntimeError(f"Responses stream failed: {stream_error[:800]}")

                        if final_response is not None:
                            if text_deltas and not self._extract_output_text(final_response):
                                final_response["output_text"] = "".join(text_deltas)
                            return final_response

                        assembled: dict[str, Any] = {"output": done_items}
                        if text_deltas:
                            assembled["output_text"] = "".join(text_deltas)
                        if assembled["output"] or assembled.get("output_text"):
                            return assembled
                        raise RuntimeError("Responses SSE stream ended without response.completed or output items.")

            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_err = exc
                if attempt < self.settings.max_retries:
                    time.sleep(min(5.0 * attempt, 20.0))
                    continue
                raise RuntimeError(f"Responses API network failure after {attempt} attempts: {exc}") from exc

        raise RuntimeError(f"Responses API call failed: {last_err}")

    @staticmethod
    def _extract_output_text(data: dict[str, Any]) -> str:
        """Extract assistant text cleanly from a Responses API JSON payload."""
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"].strip()

        parts: list[str] = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") in ("output_text", "text") and isinstance(block.get("text"), str):
                        parts.append(block["text"])
        return "\n".join(parts).strip()

    def generate_text(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_output_tokens: int = 20000,
        temperature: float = 0.25,
    ) -> str:
        """Run a text-only request via /v1/responses (SSE streaming + reasoning effort)."""
        payload: dict[str, Any] = {
            "model": model or self.settings.main_model,
            "store": not self.settings.disable_response_storage,
            "max_output_tokens": max_output_tokens,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        }
        if system:
            payload["instructions"] = system
        self._apply_reasoning(payload, temperature=temperature)

        data = self._post_responses(payload)
        text = self._extract_output_text(data)
        if not text:
            raise RuntimeError(f"Empty text response from Responses API: {str(data)[:500]}")
        return text

    def generate_multimodal(
        self,
        contents: list[Any],
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_output_tokens: int = 36000,
        temperature: float = 0.2,
    ) -> str:
        """Run a multimodal (text + N images) request via /v1/responses (SSE streaming + reasoning effort)."""
        content_blocks: list[dict[str, Any]] = []
        for item in contents:
            if isinstance(item, str):
                content_blocks.append({"type": "input_text", "text": item})
            elif isinstance(item, (Path, Image.Image)):
                content_blocks.append({"type": "input_image", "image_url": encode_image_to_data_url(item)})
            else:
                raise TypeError(f"Unsupported multimodal content item: {type(item)}")

        payload: dict[str, Any] = {
            "model": model or self.settings.main_model,
            "store": not self.settings.disable_response_storage,
            "max_output_tokens": max_output_tokens,
            "input": [{"role": "user", "content": content_blocks}],
        }
        if system:
            payload["instructions"] = system
        self._apply_reasoning(payload, temperature=temperature)

        data = self._post_responses(payload)
        text = self._extract_output_text(data)
        if not text:
            raise RuntimeError(f"Empty multimodal response from Responses API: {str(data)[:500]}")
        return text

    def generate_image(
        self,
        prompt: str,
        *,
        reference_images: Optional[list[ImageInput]] = None,
        model: Optional[str] = None,
        image_tool_model: Optional[str] = None,
        quality: Optional[str] = None,
        size: Optional[str] = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Generate a Stage-1 high-craft visual anchor using Responses API built-in image_generation tool
        with multi-image visual art-direction references.
        """
        content_blocks: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        if reference_images:
            for idx, ref_img in enumerate(reference_images, 1):
                content_blocks.append(
                    {
                        "type": "input_text",
                        "text": (
                            f"[ART-DIRECTION REFERENCE #{idx} — MIMIC ITS VISUAL CRAFT, ILLUSTRATION RICHNESS, "
                            "COLOR HARMONY, BORDER/BORDERLESS LANGUAGE, ARROW STYLE & WHITESPACE; "
                            "DO NOT COPY ANY REFERENCE WORDS OR MODULES]"
                        ),
                    }
                )
                content_blocks.append({"type": "input_image", "image_url": encode_image_to_data_url(ref_img)})

        tool_def: dict[str, Any] = {
            "type": "image_generation",
            "quality": quality or self.settings.image_quality,
            "size": size or self.settings.image_size,
            "output_format": "png",
            "background": "opaque",
        }
        chosen_img_model = image_tool_model if image_tool_model is not None else self.settings.image_tool_model
        if chosen_img_model:
            tool_def["model"] = chosen_img_model

        payload: dict[str, Any] = {
            "model": model or self.settings.main_model,
            "store": not self.settings.disable_response_storage,
            "tools": [tool_def],
            "tool_choice": {"type": "image_generation"},
            "input": [{"role": "user", "content": content_blocks}],
        }
        # Note: Do not force heavy reasoning effort onto forced tool_choice image_generation calls;
        # the image_generation tool handles visual rendering while reasoning effort powers planning/SVG/critic stages.

        try:
            data = self._post_responses(payload)
        except RuntimeError as exc:
            if any(m in str(exc) for m in HARD_QUOTA_MARKERS) or "额度" in str(exc):
                raise
            # Fallback if gateway rejects explicit tool_choice object
            payload.pop("tool_choice", None)
            data = self._post_responses(payload)

        for item in data.get("output", []):
            if item.get("type") == "image_generation_call" and item.get("result"):
                raw_bytes = base64.b64decode(item["result"])
                img = Image.open(io.BytesIO(raw_bytes))
                img.load()
                meta = {k: v for k, v in item.items() if k != "result"}
                meta["requested_image_model"] = chosen_img_model or "default"
                return img, meta

        raise RuntimeError(
            "Responses API completed without an image_generation_call image payload. "
            f"Output types: {[o.get('type') for o in data.get('output', [])]}"
        )
