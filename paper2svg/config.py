"""Configuration management for Paper2SVG-ultra v3.0 (Pure AutoFigure-Edit + Local GPU/Zero-GPU + Deterministic SVG)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
VALID_SEG_BACKENDS = {"auto", "sam3_local", "sam3_fal", "sam3_roboflow", "grounding_dino", "cc_zero_gpu"}
VALID_MATTING_BACKENDS = {"auto", "rmbg2", "birefnet", "floodfill"}


def resolve_stage_effort(base_effort: Optional[str], max_cap: str) -> str:
    """Cap per-stage reasoning effort so utility stages stay snappy while SVG synthesis stays thorough."""
    base = (base_effort or "medium").strip().lower()
    cap = (max_cap or "medium").strip().lower()
    if base not in _EFFORT_ORDER:
        base = "medium"
    if cap not in _EFFORT_ORDER:
        cap = "medium"
    return base if _EFFORT_ORDER[base] <= _EFFORT_ORDER[cap] else cap


def resolve_density_mode(density: Optional[str], inventory_count: int = 0) -> str:
    """Resolve diagram density tier ('concise' vs 'detailed') for soft print-legibility hints."""
    mode = (density or "auto").strip().lower()
    if mode in ("concise", "detailed"):
        return mode
    return "detailed" if inventory_count > 30 else "concise"


class Settings(BaseModel):
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default="")
    actor_auth: str = Field(default="")

    main_model: str = Field(default="gpt-5.4")
    reasoning_effort: str = Field(default="high")
    image_tool_model: str = Field(default="")
    image_quality: str = Field(default="high")
    image_size: str = Field(default="1536x1024")
    disable_response_storage: bool = Field(default=True)

    # Pure AutoFigure-Edit v3.0 SVG settings
    svg_mode: str = Field(default="hybrid_rich")
    density: str = Field(default="auto")
    optimize_iterations: int = Field(default=2, ge=0, le=4)
    font_family: str = Field(default=ACADEMIC_DEFAULT_FONT_EN)
    lang: str = Field(default="auto")

    # Pluggable Step 2 (SAM3 / Local GPU GroundingDINO / Zero-GPU CC) & Step 3 (RMBG-2.0 / BiRefNet / FloodFill)
    seg_backend: str = Field(default="auto")
    matting_backend: str = Field(default="auto")
    gpu_python: str = Field(default="")
    hf_endpoint: str = Field(default="")
    sam_prompts: str = Field(default="icon,person,robot,animal,diagram,chart")
    sam_min_score: float = Field(default=0.45, ge=0.05, le=0.95)
    sam_merge_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    sam_max_masks: int = Field(default=32, ge=1, le=64)
    sam_api_key: str = Field(default="")
    rmbg_model_path: str = Field(default="")
    enable_upscale_4k: bool = Field(default=False)

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
    """Enforce clean academic typography and block novelty fonts such as Comic Sans MS."""
    fallback = ACADEMIC_DEFAULT_FONT_ZH if lang == "zh" else ACADEMIC_DEFAULT_FONT_EN
    if not font_family or not font_family.strip():
        return fallback
    lowered = font_family.lower()
    for banned in BANNED_FONTS:
        if banned in lowered:
            return fallback
    return font_family.strip()


def load_settings(env_path: Optional[Path] = None) -> Settings:
    """Load settings from `.env` and environment variables."""
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

    opt_iters_raw = os.getenv("OPTIMIZE_ITERATIONS") or os.getenv("DEFAULT_CRITIC_ROUNDS") or "2"
    try:
        opt_iters = max(0, min(4, int(opt_iters_raw)))
    except ValueError:
        opt_iters = 2

    return Settings(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        actor_auth=os.getenv("OPENAI_ACTOR_AUTH", "").strip(),
        main_model=os.getenv("MAIN_MODEL", "gpt-5.4").strip(),
        reasoning_effort=effort,
        image_tool_model=os.getenv("IMAGE_TOOL_MODEL", "").strip(),
        image_quality=os.getenv("IMAGE_QUALITY", "high").strip(),
        image_size=os.getenv("IMAGE_SIZE", "1536x1024").strip(),
        disable_response_storage=_bool("DISABLE_RESPONSE_STORAGE", True),
        svg_mode=svg_mode,
        density=density,
        optimize_iterations=opt_iters,
        font_family=sanitize_font_family(os.getenv("DEFAULT_FONT_FAMILY", ACADEMIC_DEFAULT_FONT_EN)),
        lang=os.getenv("DEFAULT_LANG", "auto").strip().lower(),
        seg_backend=seg_backend,
        matting_backend=matting_backend,
        gpu_python=os.getenv("GPU_PYTHON", "").strip(),
        hf_endpoint=os.getenv("HF_ENDPOINT", "").strip(),
        sam_prompts=os.getenv("SAM_PROMPTS", "icon,person,robot,animal,diagram,chart").strip(),
        sam_min_score=float(os.getenv("SAM_MIN_SCORE", "0.45")),
        sam_merge_threshold=float(os.getenv("SAM_MERGE_THRESHOLD", "0.80")),
        sam_max_masks=int(os.getenv("SAM_MAX_MASKS", "32")),
        sam_api_key=sam_api_key,
        rmbg_model_path=os.getenv("RMBG_MODEL_PATH", "").strip(),
        enable_upscale_4k=_bool("ENABLE_UPSCALE_4K", False),
        timeout_seconds=float(os.getenv("API_TIMEOUT_SECONDS", "600")),
        max_retries=int(os.getenv("API_MAX_RETRIES", "5")),
        server_host=os.getenv("SERVER_HOST", "127.0.0.1").strip(),
        server_port=int(os.getenv("SERVER_PORT", "8090")),
    )
