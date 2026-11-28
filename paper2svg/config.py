"""Configuration management for Paper2SVG-ultra v2.0 (AIA Front-End + AutoFigure-Edit Back-End)."""

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
_EFFORT_ORDER = {"": 0, "none": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}
VALID_SVG_MODES = {"hybrid_rich", "pure_vector"}
VALID_DENSITY_MODES = {"auto", "concise", "detailed"}
VALID_SVG_ENGINES = {"autofigure", "legacy"}
VALID_SEG_BACKENDS = {"auto", "sam3_local", "sam3_fal", "sam3_roboflow", "cc_zero_gpu", "vlm_zero_gpu"}
VALID_MATTING_BACKENDS = {"auto", "rmbg2", "birefnet", "floodfill"}


def resolve_stage_effort(base_effort: Optional[str], max_cap: str) -> str:
    """Cap per-stage reasoning effort so fast utility stages (PDF/Style/Gate/Seg) never stall for 10+ minutes."""
    base = (base_effort or "medium").strip().lower()
    cap = (max_cap or "medium").strip().lower()
    if base not in _EFFORT_ORDER:
        base = "medium"
    if cap not in _EFFORT_ORDER:
        cap = "medium"
    return base if _EFFORT_ORDER[base] <= _EFFORT_ORDER[cap] else cap


def resolve_density_mode(density: Optional[str], inventory_count: int = 0) -> str:
    """Resolve effective diagram density tier ('concise' vs 'detailed').

    - 'concise': airy CVPR/Nature/NeurIPS single-strip overview (<= 30 labels, larger print floor)
    - 'detailed': dense multi-panel IEEE S&P / USENIX / OSDI systems architecture (> 30 labels)
    - 'auto': automatically selects 'detailed' when exact_text_inventory has > 30 labels, else 'concise'
    """
    mode = (density or "auto").strip().lower()
    if mode == "concise":
        return "concise"
    if mode == "detailed":
        return "detailed"
    return "detailed" if inventory_count > 30 else "concise"


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
    # 'autofigure' (v2.0 true AutoFigure-Edit <AF>01 template -> optimize -> icon swap engine) or 'legacy'
    svg_engine: str = Field(default="autofigure")
    # 'auto' (adaptive from label inventory), 'concise' (airy <=30 labels), or 'detailed' (dense systems diagram)
    density: str = Field(default="auto")
    disable_response_storage: bool = Field(default=True)

    # Cut 1 (AIA Stage-4 Aesthetic & Anti-Copy Gate)
    draft_review_rounds: int = Field(default=1, ge=0, le=3)

    # Cut 2 (AutoFigure-Edit native segmentation / matting / 4K upscale)
    seg_backend: str = Field(default="auto")
    matting_backend: str = Field(default="auto")
    sam_prompts: str = Field(default="icon,portrait,diagram,chart,object")
    sam_min_score: float = Field(default=0.45, ge=0.05, le=0.95)
    sam_merge_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    sam_max_masks: int = Field(default=32, ge=1, le=64)
    sam_api_key: str = Field(default="")
    rmbg_model_path: str = Field(default="")
    enable_upscale_4k: bool = Field(default=False)

    critic_rounds: int = Field(default=1, ge=0, le=4)
    font_family: str = Field(default=ACADEMIC_DEFAULT_FONT_EN)
    canvas_width: int = Field(default=1600)
    canvas_height: int = Field(default=680)  # ~2.35:1 wide banner default
    lang: str = Field(default="auto")  # auto | en | zh

    timeout_seconds: float = Field(default=600.0)
    max_retries: int = Field(default=5)

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

    svg_engine = os.getenv("SVG_ENGINE", "autofigure").strip().lower()
    if svg_engine not in VALID_SVG_ENGINES:
        svg_engine = "autofigure"

    density = os.getenv("DEFAULT_DENSITY", "auto").strip().lower()
    if density not in VALID_DENSITY_MODES:
        density = "auto"

    seg_backend = os.getenv("SEG_BACKEND", "auto").strip().lower()
    if seg_backend not in VALID_SEG_BACKENDS:
        seg_backend = "auto"

    matting_backend = os.getenv("MATTING_BACKEND", "auto").strip().lower()
    if matting_backend not in VALID_MATTING_BACKENDS:
        matting_backend = "auto"

    sam_api_key = (
        os.getenv("SAM_API_KEY")
        or os.getenv("FAL_KEY")
        or os.getenv("ROBOFLOW_API_KEY")
        or ""
    ).strip()

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
        svg_engine=svg_engine,
        density=density,
        disable_response_storage=_bool("DISABLE_RESPONSE_STORAGE", True),
        draft_review_rounds=int(os.getenv("DRAFT_REVIEW_ROUNDS", "1")),
        seg_backend=seg_backend,
        matting_backend=matting_backend,
        sam_prompts=os.getenv("SAM_PROMPTS", "icon,portrait,diagram,chart,object").strip(),
        sam_min_score=float(os.getenv("SAM_MIN_SCORE", "0.45")),
        sam_merge_threshold=float(os.getenv("SAM_MERGE_THRESHOLD", "0.85")),
        sam_max_masks=int(os.getenv("SAM_MAX_MASKS", "32")),
        sam_api_key=sam_api_key,
        rmbg_model_path=os.getenv("RMBG_MODEL_PATH", "").strip(),
        enable_upscale_4k=_bool("ENABLE_UPSCALE_4K", False),
        critic_rounds=int(os.getenv("DEFAULT_CRITIC_ROUNDS", "1")),
        font_family=sanitize_font_family(os.getenv("DEFAULT_FONT_FAMILY", ACADEMIC_DEFAULT_FONT_EN)),
        canvas_width=int(os.getenv("DEFAULT_CANVAS_WIDTH", "1600")),
        canvas_height=int(os.getenv("DEFAULT_CANVAS_HEIGHT", "680")),
        lang=os.getenv("DEFAULT_LANG", "auto").strip().lower(),
        timeout_seconds=float(os.getenv("API_TIMEOUT_SECONDS", "600")),
        max_retries=int(os.getenv("API_MAX_RETRIES", "5")),
        server_host=os.getenv("SERVER_HOST", "127.0.0.1").strip(),
        server_port=int(os.getenv("SERVER_PORT", "8090")),
    )
