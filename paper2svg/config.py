"""Configuration management for Paper2SVG-ultra."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Explicitly prohibited non-academic fonts
BANNED_FONTS = {
    "comic sans ms",
    "comic sans",
    "papyrus",
    "chiller",
    "curlz mt",
    "jokerman",
    "brush script mt",
}

ACADEMIC_DEFAULT_FONT_EN = "Inter, Helvetica, Arial, sans-serif"
ACADEMIC_DEFAULT_FONT_ZH = "Inter, 'Noto Sans SC', 'Microsoft YaHei', Helvetica, Arial, sans-serif"

VALID_REASONING_EFFORTS = {"", "none", "low", "medium", "high", "xhigh"}
VALID_SVG_MODES = {"hybrid_rich", "pure_vector"}


class Settings(BaseModel):
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default="")
    actor_auth: str = Field(default="")

    main_model: str = Field(default="gpt-5.4")
    reasoning_effort: str = Field(default="high")
    # Image model passed to Responses `image_generation` tool; empty = endpoint default
    image_tool_model: str = Field(default="")
    image_quality: str = Field(default="high")
    image_size: str = Field(default="1536x1024")
    # 'hybrid_rich' (editable vector cards/text/arrows + crisp embedded visual assets) or 'pure_vector'
    svg_mode: str = Field(default="hybrid_rich")
    disable_response_storage: bool = Field(default=True)

    critic_rounds: int = Field(default=2, ge=1, le=4)
    font_family: str = Field(default=ACADEMIC_DEFAULT_FONT_EN)
    canvas_width: int = Field(default=1600)
    canvas_height: int = Field(default=680)  # ~2.35:1 wide banner default
    lang: str = Field(default="auto")  # auto | en | zh

    timeout_seconds: float = Field(default=600.0)
    max_retries: int = Field(default=3)

    server_host: str = Field(default="127.0.0.1")
    server_port: int = Field(default=8090)

    @property
    def default_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.actor_auth:
            headers["x-openai-actor-authorization"] = self.actor_auth
        return headers


def sanitize_font_family(font_family: Optional[str], lang: str = "en") -> str:
    """Enforce academic sans-serif typography and block non-academic fonts like Comic Sans MS."""
    fallback = ACADEMIC_DEFAULT_FONT_ZH if lang == "zh" else ACADEMIC_DEFAULT_FONT_EN
    if not font_family or not font_family.strip():
        return fallback
    lowered = font_family.lower()
    for banned in BANNED_FONTS:
        if banned in lowered:
            return fallback
    return font_family.strip()


def load_settings(env_path: Optional[Path] = None) -> Settings:
    """Load settings from .env file and environment variables."""
    target_env = env_path or (PROJECT_ROOT / ".env")
    if target_env.is_file():
        load_dotenv(target_env, override=False)

    def _bool(key: str, default: bool) -> bool:
        val = os.getenv(key)
        if val is None:
            return default
        return val.strip().lower() in ("1", "true", "yes", "on")

    effort = os.getenv("REASONING_EFFORT", "high").strip().lower()
    if effort not in VALID_REASONING_EFFORTS:
        effort = "high"

    svg_mode = os.getenv("SVG_MODE", "hybrid_rich").strip().lower()
    if svg_mode not in VALID_SVG_MODES:
        svg_mode = "hybrid_rich"

    return Settings(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        actor_auth=os.getenv("OPENAI_ACTOR_AUTH", "").strip(),
        main_model=os.getenv("MAIN_MODEL", "gpt-5.4").strip(),
        reasoning_effort=effort,
        image_tool_model=os.getenv("IMAGE_TOOL_MODEL", "").strip(),
        image_quality=os.getenv("IMAGE_QUALITY", "high").strip(),
        image_size=os.getenv("IMAGE_SIZE", "1536x1024").strip(),
        svg_mode=svg_mode,
        disable_response_storage=_bool("DISABLE_RESPONSE_STORAGE", True),
        critic_rounds=int(os.getenv("DEFAULT_CRITIC_ROUNDS", "2")),
        font_family=sanitize_font_family(os.getenv("DEFAULT_FONT_FAMILY", ACADEMIC_DEFAULT_FONT_EN)),
        canvas_width=int(os.getenv("DEFAULT_CANVAS_WIDTH", "1600")),
        canvas_height=int(os.getenv("DEFAULT_CANVAS_HEIGHT", "680")),
        lang=os.getenv("DEFAULT_LANG", "auto").strip().lower(),
        timeout_seconds=float(os.getenv("API_TIMEOUT_SECONDS", "600")),
        server_host=os.getenv("SERVER_HOST", "127.0.0.1").strip(),
        server_port=int(os.getenv("SERVER_PORT", "8090")),
    )
