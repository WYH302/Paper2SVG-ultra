"""Cut 2: True AutoFigure-Edit Native Back-End Engine (4K Upscale + Pluggable SAM3/Zero-GPU +
Pluggable RMBG-2.0/BiRefNet/FloodFill + Clean <AF>01 Template SVG -> Optimize -> Deterministic Swap).

Preserves the proven core pipeline of upstream AutoFigure-Edit (`autofigure2.py`) while transplanting
Paper2SVG-ultra's 3 Windows-bulletproof engineering upgrades:
  1. OpenAI Responses API SSE streaming (no HTTP 524 timeouts, automatic transient overload retry)
  2. Pure-Rust zero-DLL `resvg-py` rendering (replacing fragile `cairosvg`)
  3. Zero-GPU VLM+NumPy segmentation & border-floodfill matting fallback when SAM3/RMBG-2.0 aren't installed
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .architect import extract_exact_text_inventory
from .asset_extractor import (
    detect_and_crop_visual_assets,
    smart_remove_border_background,
    strip_edge_text_bleed,
)
from .config import resolve_density_mode
from .svg_builder import extract_svg_block, fix_svg_xml_with_llm, normalize_svg_root
from .svg_linter import TWO_COLUMN_WIDTH_PT, LintReport, lint_svg, validate_svg_xml
from .svg_renderer import render_dual_preview, render_svg_to_png

UPSCALE_TARGET_LONG_EDGE = 3840
FAL_SAM3_URL = "https://fal.run/fal-ai/sam-3/image"
ROBOFLOW_SAM3_URL = "https://serverless.roboflow.com/sam3/concept_segment"


# ============================================================================
# Step 1 Helper: 4K Proportional Upscale & Imported Figure Preparation
# ============================================================================


def upscale_image_to_4k_if_needed(
    image: Image.Image,
    target_long_edge: int = UPSCALE_TARGET_LONG_EDGE,
) -> tuple[Image.Image, bool]:
    """Upscale an image proportionally so its long edge reaches 4K (exact AutoFigure-Edit logic)."""
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= 0 or long_edge >= target_long_edge:
        return image, False

    scale = target_long_edge / float(long_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    upscaled = image.resize((new_width, new_height), resample=resample)
    return upscaled, True


def prepare_imported_figure(
    input_figure_path: Path,
    output_path: Path,
    *,
    enable_upscale: bool = False,
) -> tuple[Path, tuple[int, int]]:
    """Normalize an imported raster figure (EXIF transpose, RGB PNG, optional 4K upscale)."""
    if not input_figure_path.is_file():
        raise FileNotFoundError(f"Imported figure not found: {input_figure_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_figure_path) as imported:
        img = ImageOps.exif_transpose(imported).convert("RGB")
    if enable_upscale:
        img, _ = upscale_image_to_4k_if_needed(img)
    img.save(output_path, format="PNG")
    return output_path, img.size


# ============================================================================
# Step 2: Pluggable Segmentation (SAM3 Local / Fal / Roboflow / Zero-GPU VLM)
# ============================================================================


def _get_hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return None


def _has_hf_cached_repo(repo_slug: str) -> bool:
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    snapshots_dir = hf_home / "hub" / f"models--{repo_slug.replace('/', '--')}" / "snapshots"
    if not snapshots_dir.exists():
        return False
    return any(snapshots_dir.glob("*/config.json")) or any(snapshots_dir.glob("*/*.safetensors")) or any(
        snapshots_dir.glob("*/*.pt")
    )


def _sam3_local_ready() -> bool:
    if importlib.util.find_spec("sam3") is None or importlib.util.find_spec("torch") is None:
        return False
    return _has_hf_cached_repo("facebook/sam3") or (_get_hf_token() is not None)


def _rmbg2_local_ready(rmbg_model_path: str = "") -> bool:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        return False
    if rmbg_model_path and Path(rmbg_model_path).exists():
        return True
    return _has_hf_cached_repo("briaai/RMBG-2.0") or (_get_hf_token() is not None)


def _birefnet_or_rembg_ready() -> bool:
    if importlib.util.find_spec("rembg") is not None:
        return True
    if importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None:
        return _has_hf_cached_repo("ZhengPeng7/BiRefNet")
    return False


def resolve_seg_backend(requested: str = "auto", sam_api_key: str = "") -> str:
    """Resolve effective segmentation backend cleanly without crashing on zero-GPU machines."""
    req = (requested or "auto").strip().lower()
    if req in ("sam3_local", "sam3_fal", "sam3_roboflow", "vlm_zero_gpu"):
        return req
    if _sam3_local_ready():
        return "sam3_local"
    if (sam_api_key and os.environ.get("FAL_KEY")) or os.environ.get("FAL_KEY"):
        return "sam3_fal"
    if os.environ.get("ROBOFLOW_API_KEY"):
        return "sam3_roboflow"
    if sam_api_key:
        return "sam3_fal"
    return "vlm_zero_gpu"


def resolve_matting_backend(requested: str = "auto", rmbg_model_path: str = "") -> str:
    """Resolve effective matting backend (RMBG-2.0 -> open BiRefNet/rembg -> zero-GPU floodfill)."""
    req = (requested or "auto").strip().lower()
    if req in ("rmbg2", "birefnet", "floodfill"):
        return req
    if _rmbg2_local_ready(rmbg_model_path):
        return "rmbg2"
    if _birefnet_or_rembg_ready():
        return "birefnet"
    return "floodfill"


def _get_label_font(box_width: int, box_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Exact AutoFigure-Edit dynamic label font for <AF>01 boxes."""
    min_dim = min(box_width, box_height)
    font_size = max(12, min(48, min_dim // 4))
    for fp in (
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(fp, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def calculate_overlap_ratio(box1: dict[str, Any], box2: dict[str, Any]) -> float:
    """Exact AutoFigure-Edit overlap ratio: intersection / min(area1, area2)."""
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    a2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    if a1 <= 0 or a2 <= 0:
        return 0.0
    return inter / float(min(a1, a2))


def merge_overlapping_boxes(boxes: list[dict[str, Any]], overlap_threshold: float = 0.85) -> list[dict[str, Any]]:
    """Iteratively merge overlapping detected boxes and assign canonical <AF>01..NN labels."""
    if not boxes:
        return []
    working = [b.copy() for b in boxes]
    if overlap_threshold > 0 and len(working) > 1:
        merged = True
        while merged:
            merged = False
            n = len(working)
            for i in range(n):
                if merged:
                    break
                for j in range(i + 1, n):
                    if calculate_overlap_ratio(working[i], working[j]) >= overlap_threshold:
                        b1, b2 = working[i], working[j]
                        best = b1 if b1.get("score", 0.0) >= b2.get("score", 0.0) else b2
                        nb = {
                            "x1": min(b1["x1"], b2["x1"]),
                            "y1": min(b1["y1"], b2["y1"]),
                            "x2": max(b1["x2"], b2["x2"]),
                            "y2": max(b1["y2"], b2["y2"]),
                            "score": max(float(b1.get("score", 0.0)), float(b2.get("score", 0.0))),
                            "prompt": best.get("prompt", "icon"),
                            "bg_mode": best.get("bg_mode", "auto"),
                            "rounded_rx": int(best.get("rounded_rx", 0)),
                            "description": best.get("description", ""),
                        }
                        working = [working[k] for k in range(n) if k != i and k != j]
                        working.append(nb)
                        merged = True
                        break

    # Deterministic reading-order sort (top-to-bottom row bands, left-to-right)
    working.sort(key=lambda b: (int(b["y1"]) // 60, int(b["x1"])))

    result: list[dict[str, Any]] = []
    for idx, box in enumerate(working):
        result.append(
            {
                "id": idx,
                "label": f"<AF>{idx + 1:02d}",
                "label_clean": f"AF{idx + 1:02d}",
                "x1": int(box["x1"]),
                "y1": int(box["y1"]),
                "x2": int(box["x2"]),
                "y2": int(box["y2"]),
                "score": round(float(box.get("score", 1.0)), 4),
                "prompt": str(box.get("prompt", "icon")),
                "bg_mode": str(box.get("bg_mode", "auto")),
                "rounded_rx": int(box.get("rounded_rx", 0)),
                "description": str(box.get("description", "")),
            }
        )
    return result


def draw_samed_placeholder_image(
    figure_img: Image.Image,
    boxes: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Draw exact AutoFigure-Edit `samed.png`: #808080 gray fill, black 3px outline, white centered <AF>01."""
    samed = figure_img.convert("RGB").copy()
    draw = ImageDraw.Draw(samed)

    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        label = box["label"]
        draw.rectangle([x1, y1, x2, y2], fill="#808080", outline="black", width=3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        font = _get_label_font(max(1, x2 - x1), max(1, y2 - y1))
        try:
            draw.text((cx, cy), label, fill="white", anchor="mm", font=font)
        except TypeError:
            tb = draw.textbbox((0, 0), label, font=font)
            draw.text((cx - (tb[2] - tb[0]) // 2, cy - (tb[3] - tb[1]) // 2), label, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    samed.save(output_path, format="PNG")
    return output_path


def _cxcywh_norm_to_xyxy(box: list | tuple, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    cx, cy, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if max(cx, cy, bw, bh) <= 1.5:
        cx, bw = cx * width, bw * width
        cy, bh = cy * height, bh * height
    x1 = max(0, min(width - 1, int(round(cx - bw / 2.0))))
    y1 = max(0, min(height - 1, int(round(cy - bh / 2.0))))
    x2 = max(x1 + 1, min(width, int(round(cx + bw / 2.0))))
    y2 = max(y1 + 1, min(height, int(round(cy + bh / 2.0))))
    return (x1, y1, x2, y2)


def _run_sam3_local(
    image: Image.Image,
    prompt_list: list[str],
    min_score: float,
) -> list[dict[str, Any]]:
    import sam3
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    sam3_dir = Path(sam3.__path__[0]) if hasattr(sam3, "__path__") else Path(sam3.__file__).parent
    bpe_path = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(device=device, bpe_path=str(bpe_path) if bpe_path.exists() else None)
    processor = Sam3Processor(model, device=device)
    state = processor.set_image(image)

    detected: list[dict[str, Any]] = []
    for prompt in prompt_list:
        out = processor.set_text_prompt(state=state, prompt=prompt)
        boxes = out["boxes"].cpu().numpy() if isinstance(out["boxes"], torch.Tensor) else out["boxes"]
        scores = out["scores"].cpu().numpy() if isinstance(out["scores"], torch.Tensor) else out["scores"]
        for box, score in zip(boxes, scores):
            if float(score) >= min_score:
                detected.append(
                    {
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3]),
                        "score": float(score),
                        "prompt": prompt,
                        "bg_mode": "auto",
                    }
                )
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return detected


def _run_sam3_fal(
    image: Image.Image,
    prompt_list: list[str],
    min_score: float,
    api_key: str,
    max_masks: int = 32,
) -> list[dict[str, Any]]:
    key = api_key or os.environ.get("FAL_KEY", "")
    if not key:
        raise ValueError("SAM3 fal.ai backend requires FAL_KEY or --sam-api-key.")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    w, h = image.size
    detected: list[dict[str, Any]] = []
    with httpx.Client(timeout=120.0) as http:
        for prompt in prompt_list:
            resp = http.post(
                FAL_SAM3_URL,
                headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
                json={
                    "image_url": data_uri,
                    "prompt": prompt,
                    "include_boxes": True,
                    "include_scores": True,
                    "return_multiple_masks": True,
                    "max_masks": max(1, min(32, max_masks)),
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            raw_boxes = payload.get("boxes") or []
            raw_scores = payload.get("scores") or []
            for idx, rb in enumerate(raw_boxes):
                sc = float(raw_scores[idx]) if idx < len(raw_scores) else 0.8
                if sc < min_score:
                    continue
                xyxy = _cxcywh_norm_to_xyxy(rb, w, h)
                if xyxy:
                    detected.append(
                        {
                            "x1": xyxy[0],
                            "y1": xyxy[1],
                            "x2": xyxy[2],
                            "y2": xyxy[3],
                            "score": sc,
                            "prompt": prompt,
                            "bg_mode": "auto",
                        }
                    )
    return detected


def _run_sam3_roboflow(
    image: Image.Image,
    prompt_list: list[str],
    min_score: float,
    api_key: str,
) -> list[dict[str, Any]]:
    key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        raise ValueError("SAM3 Roboflow backend requires ROBOFLOW_API_KEY or --sam-api-key.")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    w, h = image.size
    detected: list[dict[str, Any]] = []
    with httpx.Client(timeout=120.0) as http:
        for prompt in prompt_list:
            resp = http.post(
                ROBOFLOW_SAM3_URL,
                params={"api_key": key},
                json={
                    "format": "polygon",
                    "image": {"type": "base64", "value": img_b64},
                    "prompts": [{"type": "text", "text": prompt}],
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            for res in payload.get("prompt_results") or []:
                for pred in res.get("predictions") or []:
                    sc = float(pred.get("confidence", 0.0))
                    if sc < min_score:
                        continue
                    xs: list[float] = []
                    ys: list[float] = []
                    for poly in pred.get("masks") or []:
                        for pt in poly:
                            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                                xs.append(float(pt[0]))
                                ys.append(float(pt[1]))
                    if xs and ys:
                        detected.append(
                            {
                                "x1": max(0, int(min(xs))),
                                "y1": max(0, int(min(ys))),
                                "x2": min(w, int(max(xs))),
                                "y2": min(h, int(max(ys))),
                                "score": sc,
                                "prompt": prompt,
                                "bg_mode": "auto",
                            }
                        )
    return detected


def segment_icons_autofigure(
    client: Any,
    figure_path: Path,
    output_dir: Path,
    *,
    seg_backend: str = "auto",
    sam_prompts: str = "icon,portrait,diagram,chart,object",
    min_score: float = 0.45,
    merge_threshold: float = 0.85,
    sam_api_key: str = "",
    sam_max_masks: int = 32,
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[Path, Path, list[dict[str, Any]], str]:
    """Run AutoFigure-Edit Step 2 icon segmentation -> `samed.png` + `boxlib.json`."""
    with Image.open(figure_path) as im:
        fig_img = im.convert("RGB")
    w, h = fig_img.size

    resolved = resolve_seg_backend(seg_backend, sam_api_key)
    prompt_list = [p.strip() for p in sam_prompts.split(",") if p.strip()] or ["icon"]
    if on_progress:
        on_progress(f"AutoFigure Step 2: Segmenting visual assets via `{resolved}` ({w}x{h})...")

    raw_boxes: list[dict[str, Any]] = []
    if resolved == "sam3_local":
        raw_boxes = _run_sam3_local(fig_img, prompt_list, min_score)
    elif resolved == "sam3_fal":
        raw_boxes = _run_sam3_fal(fig_img, prompt_list, min_score, sam_api_key, sam_max_masks)
    elif resolved == "sam3_roboflow":
        raw_boxes = _run_sam3_roboflow(fig_img, prompt_list, min_score, sam_api_key)
    else:
        # Zero-GPU fallback using our calibrated 1:1 VLM + pure-NumPy pixel-snapper!
        extracted = detect_and_crop_visual_assets(
            client,
            figure_path,
            output_dir,
            (w, h),
            max_assets=28,
        )
        for a in extracted:
            x1 = max(0, int(round(a.svg_x)))
            y1 = max(0, int(round(a.svg_y)))
            x2 = min(w, int(round(a.svg_x + a.svg_w)))
            y2 = min(h, int(round(a.svg_y + a.svg_h)))
            if x2 - x1 >= 12 and y2 - y1 >= 12:
                raw_boxes.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": 1.0,
                        "prompt": a.bg_mode,
                        "bg_mode": a.bg_mode,
                        "rounded_rx": a.rounded_rx,
                        "description": a.description,
                    }
                )

    # Filter giant full-canvas false positives (> 65% of entire canvas area)
    canvas_area = float(max(1, w * h))
    filtered = [
        b
        for b in raw_boxes
        if (b["x2"] - b["x1"]) >= 14
        and (b["y2"] - b["y1"]) >= 14
        and ((b["x2"] - b["x1"]) * (b["y2"] - b["y1"])) / canvas_area <= 0.65
    ]

    valid_boxes = merge_overlapping_boxes(filtered, overlap_threshold=merge_threshold)
    samed_path = draw_samed_placeholder_image(fig_img, valid_boxes, output_dir / "samed.png")
    # Keep Web UI alias synchronized
    shutil.copyfile(samed_path, output_dir / "stage1_draft_masked.png")

    boxlib_data = {
        "image_size": {"width": w, "height": h},
        "seg_backend": resolved,
        "prompts_used": prompt_list,
        "boxes": valid_boxes,
        "no_icon_mode": len(valid_boxes) == 0,
    }
    boxlib_path = output_dir / "boxlib.json"
    boxlib_path.write_text(json.dumps(boxlib_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return samed_path, boxlib_path, valid_boxes, resolved


# ============================================================================
# Step 3: Pluggable Matting (RMBG-2.0 / BiRefNet / Zero-GPU FloodFill)
# ============================================================================


def _looks_like_opaque_photo_tile(crop_img: Image.Image) -> bool:
    """Detect whether a cropped tile is a rectangular photograph/heatmap (should NOT have its
    studio backdrop chewed away by matting) vs an icon/diagram on a uniform card/white canvas.
    """
    rgb = np.asarray(crop_img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    if h < 16 or w < 16:
        return False
    border = np.concatenate([rgb[0, :, :], rgb[h - 1, :, :], rgb[:, 0, :], rgb[:, w - 1, :]], axis=0)
    med = np.median(border, axis=0)
    # If border is bright neutral white/near-white or soft light pastel (>215 luminance) AND uniform -> diagram/icon!
    lum = float(0.299 * med[0] + 0.587 * med[1] + 0.114 * med[2])
    dists = np.linalg.norm(border - med, axis=1)
    uniform_ratio = float(np.mean(dists < 22.0))
    if uniform_ratio >= 0.55 and lum >= 205.0:
        return False
    # Photographic portrait/scene/spectrogram borders have varied colors or darker natural backdrop
    if uniform_ratio < 0.45:
        return True
    # Check if top/bottom/left/right corners differ significantly (photographic gradient/background)
    corners = np.array([rgb[2, 2], rgb[2, w - 3], rgb[h - 3, 2], rgb[h - 3, w - 3]])
    corner_spread = float(np.max(np.linalg.norm(corners - med, axis=1)))
    return corner_spread > 38.0


def _load_neural_matte_fn(backend: str, rmbg_model_path: str = "") -> Callable[[Image.Image], Image.Image]:
    """Load neural background remover (RMBG-2.0 or BiRefNet / rembg)."""
    if backend == "birefnet" and importlib.util.find_spec("rembg") is not None:
        import rembg

        def _rembg_fn(img: Image.Image) -> Image.Image:
            return rembg.remove(img.convert("RGBA"))

        return _rembg_fn

    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = _get_hf_token()
    if backend == "rmbg2":
        repo = rmbg_model_path if (rmbg_model_path and Path(rmbg_model_path).exists()) else "briaai/RMBG-2.0"
        model = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True, token=hf_token).eval().to(device)
    else:
        model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).eval().to(device)

    tf = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def _torch_matte(img: Image.Image) -> Image.Image:
        rgb = img.convert("RGB")
        inp = tf(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(inp)[-1].sigmoid().cpu()
        mask = transforms.ToPILImage()(preds[0].squeeze()).resize(rgb.size)
        out = rgb.copy()
        out.putalpha(mask)
        return out

    return _torch_matte


def crop_and_matte_icons(
    figure_path: Path,
    boxes: list[dict[str, Any]],
    output_dir: Path,
    *,
    matting_backend: str = "auto",
    rmbg_model_path: str = "",
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[list[dict[str, Any]], str]:
    """Run AutoFigure-Edit Step 3: Crop boxes and remove backgrounds cleanly."""
    icons_dir = output_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_matting_backend(matting_backend, rmbg_model_path)

    if not boxes:
        return [], resolved

    if on_progress:
        on_progress(f"AutoFigure Step 3: Cropping & matting {len(boxes)} visual asset(s) via `{resolved}`...")

    neural_fn: Optional[Callable[[Image.Image], Image.Image]] = None
    if resolved in ("rmbg2", "birefnet"):
        try:
            neural_fn = _load_neural_matte_fn(resolved, rmbg_model_path)
        except Exception:
            resolved = "floodfill"
            neural_fn = None

    with Image.open(figure_path) as im:
        fig_img = im.convert("RGB")

    icon_infos: list[dict[str, Any]] = []
    for box in boxes:
        label = box["label"]
        label_clean = box.get("label_clean") or label.replace("<", "").replace(">", "")
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        cropped = fig_img.crop((x1, y1, x2, y2))
        crop_path = icons_dir / f"icon_{label_clean}.png"
        cropped.save(crop_path, format="PNG")

        bg_mode = str(box.get("bg_mode", "auto")).lower()
        is_opaque_photo = (bg_mode == "opaque_rect") or (
            bg_mode == "auto" and _looks_like_opaque_photo_tile(cropped)
        )

        adj_x1, adj_y1, adj_x2, adj_y2 = x1, y1, x2, y2
        if is_opaque_photo:
            nobg_img = cropped.convert("RGBA")
        elif neural_fn is not None:
            nobg_img = neural_fn(cropped)
        else:
            stripped, bleed_trims = strip_edge_text_bleed(cropped)
            adj_x1 += bleed_trims[0]
            adj_y1 += bleed_trims[1]
            adj_x2 -= bleed_trims[2]
            adj_y2 -= bleed_trims[3]
            nobg_img, ff_trims = smart_remove_border_background(stripped)
            adj_x1 += ff_trims[0]
            adj_y1 += ff_trims[1]
            adj_x2 -= ff_trims[2]
            adj_y2 -= ff_trims[3]

        nobg_path = icons_dir / f"icon_{label_clean}_nobg.png"
        nobg_img.save(nobg_path, format="PNG")

        icon_infos.append(
            {
                "id": box["id"],
                "label": label,
                "label_clean": label_clean,
                "x1": adj_x1,
                "y1": adj_y1,
                "x2": adj_x2,
                "y2": adj_y2,
                "width": max(1, adj_x2 - adj_x1),
                "height": max(1, adj_y2 - adj_y1),
                "orig_x1": x1,
                "orig_y1": y1,
                "orig_width": max(1, x2 - x1),
                "orig_height": max(1, y2 - y1),
                "bg_mode": "opaque_rect" if is_opaque_photo else "remove_bg",
                "rounded_rx": int(box.get("rounded_rx", 0)),
                "crop_path": str(crop_path),
                "nobg_path": str(nobg_path),
            }
        )

    return icon_infos, resolved


# ============================================================================
# Step 4: Razor-Clean AutoFigure-Edit `<AF>01` Template SVG Reconstruction
# ============================================================================


def generate_autofigure_svg_template(
    client: Any,
    figure_path: Path,
    samed_path: Path,
    boxes: list[dict[str, Any]],
    output_path: Path,
    *,
    font_family: str = "Inter, Helvetica, Arial, sans-serif",
    exact_text_inventory: Optional[list[str]] = None,
    resolved_density: str = "concise",
) -> str:
    """Exact AutoFigure-Edit Step 4 (`generate_svg_template`): focused, distraction-free pixel replication."""
    with Image.open(figure_path) as im:
        w, h = im.size

    min_pt = 4.3 if resolved_density == "detailed" else 6.4
    min_px = int(math.ceil(min_pt * float(w) / TWO_COLUMN_WIDTH_PT))

    no_icon_mode = len(boxes) == 0
    spelling_block = (
        "\nEXACT SPELLING REFERENCE (use these exact spellings for matching visible labels; do NOT add unshown text):\n"
        + json.dumps(exact_text_inventory, ensure_ascii=False)
        if exact_text_inventory
        else ""
    )

    if no_icon_mode:
        prompt = f"""Write clean, editable SVG code to achieve pixel-level faithful replication of Image 1 (`figure.png`).

CRITICAL DIMENSION REQUIREMENT:
- Original image dimensions: {w} x {h} pixels.
- Your SVG root tag MUST be:
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">`
- Replicate EVERY shape, panel, text label (using `{font_family}`, minimum `font-size >= {min_px}px`), arrow (exact direction, color, thickness, arrowhead), border, and color with exact visual parity.
- Do NOT add any gray `<AF>01` placeholder boxes.
{spelling_block}

Output ONLY the complete valid SVG code starting with `<svg` and ending with `</svg>`. NO markdown fences, NO commentary."""
    else:
        coord_lines = "\n".join(
            f"- {b['label_clean']} ({b['label']}): x={b['x1']}, y={b['y1']}, width={b['x2'] - b['x1']}, height={b['y2'] - b['y1']}"
            for b in boxes
        )
        prompt = f"""Write clean, editable SVG code to achieve **pixel-level faithful replication** of Image 1 (`figure.png`), replacing ONLY the icon/image areas covered by gray boxes in Image 2 (`samed.png`) with exact `<g id="AF..">` placeholders!

CRITICAL DIMENSION REQUIREMENT:
- Original image dimensions: {w} x {h} pixels.
- Your SVG root tag MUST be:
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">`
- DO NOT scale or resize the coordinate system.

EXACT PLACEHOLDER REQUIREMENT (matching Image 2 `samed.png`):
- For EVERY gray placeholder `<AF>01`, `<AF>02`, ... listed below, emit an exact `<g id="AF01">` group using the exact pixel coordinates:
```xml
<g id="AF01">
  <rect x="100" y="50" width="80" height="80" fill="#808080" stroke="black" stroke-width="2"/>
  <text x="140" y="90" text-anchor="middle" dominant-baseline="middle" fill="white" font-size="16">&lt;AF&gt;01</text>
</g>
```
- **ZERO DUPLICATE VECTOR ARTWORK BEHIND PLACEHOLDERS**: NEVER draw vector shapes, 3D slab edges, polygons, or duplicate lines INSIDE or BEHIND any `<g id="AF..">` area — our next step pastes the real transparent PNG there, so anything drawn behind it will peek out as ugly ghost lines! (Only clean flat background panel fills are allowed behind icons sitting inside a tinted zone.)
- Replicate ALL other visual components from Image 1 faithfully: background, soft tint zones, borders, crisp sans-serif `<text>` (`font-family="{font_family}"`, minimum `font-size >= {min_px}px` for 2-column print legibility, matching weight & color), and EVERY arrow (exact start/end, exact arrowhead direction UP/DOWN/LEFT/RIGHT, exact stroke width & color).
- Escape XML entities strictly (`&amp;`, `&lt;`, `&gt;`). NEVER use `textLength` squashing.

EXACT PLACEHOLDER COORDINATES ({len(boxes)} boxes):
{coord_lines}
{spelling_block}

Output ONLY the complete valid SVG code starting with `<svg` and ending with `</svg>`. NO markdown fences, NO commentary."""

    raw = client.generate_multimodal(
        [
            prompt,
            f"Image 1 — Original Target Figure (`figure.png`, {w}x{h}):",
            figure_path,
            f"Image 2 — Icon Placeholder Reference (`samed.png`, {w}x{h}):",
            samed_path,
        ],
        max_output_tokens=64000,
        temperature=0.15,
    )
    svg_code = normalize_svg_root(extract_svg_block(raw), w, h)
    if not svg_code:
        raise RuntimeError(f"AutoFigure template generation returned no <svg> block. Preview: {raw[:400]}")

    ok, errs = validate_svg_xml(svg_code)
    if not ok:
        svg_code = fix_svg_xml_with_llm(client, svg_code, errs, width=w, height=h)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_code, encoding="utf-8")
    return svg_code


# ============================================================================
# Step 4.6: AutoFigure-Edit Visual Template Optimization (via Windows resvg-py)
# ============================================================================


def _count_af_placeholders(svg_code: str, boxes: list[dict[str, Any]]) -> tuple[int, list[str]]:
    missing = []
    for b in boxes:
        lc = b.get("label_clean") or b["label"].replace("<", "").replace(">", "")
        if not re.search(rf'\bid=["\']?{re.escape(lc)}["\']?', svg_code, re.IGNORECASE):
            missing.append(lc)
    return len(boxes) - len(missing), missing


def optimize_autofigure_template(
    client: Any,
    figure_path: Path,
    samed_path: Path,
    template_svg: str,
    boxes: list[dict[str, Any]],
    output_dir: Path,
    *,
    max_iterations: int = 2,
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Exact AutoFigure-Edit Step 4.6 (`optimize_svg_with_llm`) running on compact `template.svg`
    using Windows zero-DLL `resvg-py` instead of `cairosvg`.
    """
    with Image.open(figure_path) as im:
        w, h = im.size

    current_svg = template_svg
    round_logs: list[dict[str, Any]] = []
    no_icon_mode = len(boxes) == 0

    if max_iterations <= 0:
        (output_dir / "optimized_template.svg").write_text(current_svg, encoding="utf-8")
        return current_svg, round_logs

    for r in range(1, max_iterations + 1):
        if on_progress:
            on_progress(f"AutoFigure Optimize Round {r}/{max_iterations}: Rendering template via resvg-py...")

        iter_svg = output_dir / f"iter_{r - 1}.svg"
        iter_png = output_dir / f"iter_{r - 1}_full.png"
        iter_svg.write_text(current_svg, encoding="utf-8")
        renderer_used = render_svg_to_png(current_svg, iter_png, width=min(w, 2200))

        if no_icon_mode:
            opt_prompt = f"""You are an expert SVG optimizer (AutoFigure-Edit). Compare Image 3 (our current SVG rendered as PNG) against Image 1 (`figure.png`, original target) and fix EVERY visual alignment or style discrepancy.

Check and fix ALL 8 points:
## POSITION: (1) Graphics alignment, (2) Text positions & zero overlaps/clipping, (3) Arrow endpoints & exact directions (UP/DOWN/LEFT/RIGHT!), (4) Borders & dividers.
## STYLE: (5) Graphic proportions, (6) Font sizes, weights & dark crisp contrast, (7) Arrow thicknesses & colors, (8) Border colors & stroke widths.

CURRENT SVG CODE:
```xml
{current_svg}
```

Output ONLY the complete optimized SVG code starting with `<svg` and ending with `</svg>`. NO markdown fences, NO commentary."""
        else:
            opt_prompt = f"""You are an expert SVG optimizer (AutoFigure-Edit). Compare Image 3 (our current template SVG rendered as PNG) against Image 1 (`figure.png`, target figure) and Image 2 (`samed.png`, target icon placeholders `<AF>01`..`<AF>{len(boxes):02d}`) and refine the SVG to match with pixel-level precision!

Check and fix **TWO MAJOR ASPECTS with EIGHT KEY POINTS**:
## ASPECT 1: POSITION
1. **Icons (`<g id="AF..">`)**: Are every gray placeholder's `(x, y, width, height)` aligned pixel-for-pixel with Image 2 (`samed.png`)?
2. **Text**: Are ALL text labels properly centered/aligned with ZERO overlap against placeholders, arrows, or borders, and ZERO clipping outside `[0, {w}] x [0, {h}]`?
3. **Arrows**: Check EVERY arrow's exact start/end point AND arrowhead direction (`UP` vs `DOWN`, `LEFT` vs `RIGHT`) against Image 1! Fix ANY inverted, crooked, missing, or colliding arrow!
4. **Lines/Borders**: Are containers, dividers, and panels cleanly aligned to Image 1?

## ASPECT 2: STYLE
5. **Icons**: Keep EVERY `<g id="AF01">` .. `<g id="AF{len(boxes):02d}">` placeholder intact (`fill="#808080"`, `stroke="black"`, centered `&lt;AF&gt;..` text). NEVER delete any placeholder AND **REMOVE any duplicate vector shapes/slab-edges/polygons drawn behind or inside an `<AF>` box**!
6. **Text**: Match Image 1's font sizes, bold weights, and crisp ink colors (NO `textLength` squashing!).
7. **Arrows**: Match Image 1's arrow styles, thicknesses, arrowheads, and colors.
8. **Lines/Borders**: Match Image 1's subtle fills, corner radii, and stroke widths.

CURRENT SVG CODE:
```xml
{current_svg}
```

Output ONLY the complete optimized SVG code starting with `<svg` and ending with `</svg>`. NO markdown fences, NO commentary."""

        if on_progress:
            on_progress(f"AutoFigure Optimize Round {r}/{max_iterations}: Consulting {client.settings.main_model}...")

        try:
            raw = client.generate_multimodal(
                [
                    opt_prompt,
                    f"Image 1 — Original Target Figure (`figure.png`, {w}x{h}):",
                    figure_path,
                    f"Image 2 — Icon Placeholder Reference (`samed.png`, {w}x{h}):",
                    samed_path,
                    f"Image 3 — Current Template SVG Render (`iter_{r - 1}_full.png`):",
                    iter_png,
                ],
                max_output_tokens=64000,
                temperature=0.15,
            )
        except Exception as exc:
            round_logs.append(
                {
                    "round": r,
                    "renderer": renderer_used,
                    "accepted": False,
                    "reject_reason": f"Upstream API error during optimize round {r}: {exc}",
                }
            )
            break

        candidate = normalize_svg_root(extract_svg_block(raw), w, h)
        accepted = False
        reject_reason = ""
        if not candidate:
            reject_reason = "No <svg> block extracted from optimization reply."
        else:
            ok, errs = validate_svg_xml(candidate)
            if not ok:
                candidate = fix_svg_xml_with_llm(client, candidate, errs, width=w, height=h)
                ok, errs = validate_svg_xml(candidate)
            if not ok:
                reject_reason = f"Invalid XML after repair: {errs[:2]}"
            else:
                kept, missing = _count_af_placeholders(candidate, boxes)
                if boxes and len(missing) > max(0, int(len(boxes) * 0.2)):
                    reject_reason = f"Optimization dropped too many <AF> placeholders ({missing}); keeping previous template."
                else:
                    current_svg = candidate
                    accepted = True

        round_logs.append(
            {
                "round": r,
                "renderer": renderer_used,
                "accepted": accepted,
                "reject_reason": reject_reason,
            }
        )

    (output_dir / "optimized_template.svg").write_text(current_svg, encoding="utf-8")
    return current_svg, round_logs


# ============================================================================
# Step 5: Exact AutoFigure-Edit Coordinate-Aligned Icon Replacement
# ============================================================================


def get_svg_dimensions(svg_code: str) -> tuple[Optional[float], Optional[float]]:
    """Extract SVG coordinate dimensions from viewBox or width/height (exact AutoFigure-Edit logic)."""
    m = re.search(r'viewBox=["\']([^"\']+)["\']', svg_code, re.IGNORECASE)
    if m:
        parts = re.split(r"[\s,]+", m.group(1).strip())
        if len(parts) >= 4:
            try:
                return float(parts[2]), float(parts[3])
            except ValueError:
                pass
    w_m = re.search(r'\bwidth=["\']([\d.]+)', svg_code, re.IGNORECASE)
    h_m = re.search(r'\bheight=["\']([\d.]+)', svg_code, re.IGNORECASE)
    if w_m and h_m:
        try:
            return float(w_m.group(1)), float(h_m.group(1))
        except ValueError:
            pass
    return None, None


def replace_icons_in_svg(
    template_svg: str,
    icon_infos: list[dict[str, Any]],
    figure_size: tuple[int, int],
) -> tuple[str, int]:
    """Deterministically replace `<g id="AF01">...</g>` placeholders with transparent base64 `<image>` tags."""
    if not icon_infos:
        return template_svg, 0

    fig_w, fig_h = figure_size
    svg_w, svg_h = get_svg_dimensions(template_svg)
    scale_x = (svg_w / float(fig_w)) if (svg_w and fig_w > 0) else 1.0
    scale_y = (svg_h / float(fig_h)) if (svg_h and fig_h > 0) else 1.0

    svg_content = template_svg
    clip_defs: list[str] = []
    replaced_count = 0

    for info in icon_infos:
        label = info.get("label", "")
        label_clean = info.get("label_clean") or label.replace("<", "").replace(">", "")
        nobg_path = Path(info["nobg_path"])
        if not nobg_path.is_file():
            continue

        b64 = base64.b64encode(nobg_path.read_bytes()).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"
        rounded_rx = int(info.get("rounded_rx", 0))
        is_opaque = info.get("bg_mode") == "opaque_rect"

        def _make_image_tag(x: float, y: float, w: float, h: float) -> str:
            clip_attr = ""
            if is_opaque and rounded_rx > 0:
                cid = f"clip_{label_clean}"
                clip_defs.append(
                    f'  <clipPath id="{cid}"><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rounded_rx}"/></clipPath>'
                )
                clip_attr = f' clip-path="url(#{cid})"'
            return (
                f'<image id="icon_{label_clean}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'href="{data_uri}"{clip_attr} preserveAspectRatio="xMidYMid meet"/>'
            )

        # Compute exact adjusted pixel offset inside the original placeholder box (if trim shifted it)
        dx_trim = (info["x1"] - info.get("orig_x1", info["x1"])) * scale_x
        dy_trim = (info["y1"] - info.get("orig_y1", info["y1"])) * scale_y
        trim_w_ratio = info["width"] / float(max(1, info.get("orig_width", info["width"])))
        trim_h_ratio = info["height"] / float(max(1, info.get("orig_height", info["height"])))

        replaced = False

        # Method 1: Match `<g id="AF01">...</g>`
        g_pattern = rf'<g[^>]*\bid=["\']?{re.escape(label_clean)}["\']?[^>]*>[\s\S]*?</g>'
        g_match = re.search(g_pattern, svg_content, re.IGNORECASE)
        if g_match:
            g_block = g_match.group(0)
            tx, ty = 0.0, 0.0
            t_match = re.search(
                r'transform=["\'][^"\']*translate\s*\(\s*([\d.-]+)[\s,]+([\d.-]+)\s*\)',
                g_block.split(">", 1)[0],
                re.IGNORECASE,
            )
            if t_match:
                tx, ty = float(t_match.group(1)), float(t_match.group(2))

            rect_m = re.search(r"<rect\b[^>]*>", g_block, re.IGNORECASE)
            if rect_m:
                r_tag = rect_m.group(0)
                xm = re.search(r'\bx=["\']?([\d.-]+)', r_tag)
                ym = re.search(r'\by=["\']?([\d.-]+)', r_tag)
                wm = re.search(r'\bwidth=["\']?([\d.]+)', r_tag)
                hm = re.search(r'\bheight=["\']?([\d.]+)', r_tag)
                if xm and ym and wm and hm:
                    rx = float(xm.group(1)) + tx + dx_trim
                    ry = float(ym.group(1)) + ty + dy_trim
                    rw = float(wm.group(1)) * trim_w_ratio
                    rh = float(hm.group(1)) * trim_h_ratio
                    svg_content = svg_content.replace(g_block, _make_image_tag(rx, ry, rw, rh), 1)
                    replaced = True

        # Method 2: Match `<text>&lt;AF&gt;01</text>` near a `<rect>`
        if not replaced and label:
            for tp in (
                rf"<text[^>]*>[^<]*{re.escape(label)}[^<]*</text>",
                rf"<text[^>]*>[^<]*&lt;AF&gt;{re.escape(label_clean[2:])}[^<]*</text>",
            ):
                tm = re.search(tp, svg_content, re.IGNORECASE)
                if tm:
                    before = svg_content[: tm.start()]
                    rects = list(re.finditer(r"<rect[^>]*/?\s*>", before, re.IGNORECASE))
                    if rects:
                        r_tag = rects[-1].group(0)
                        xm = re.search(r'\bx=["\']?([\d.-]+)', r_tag)
                        ym = re.search(r'\by=["\']?([\d.-]+)', r_tag)
                        wm = re.search(r'\bwidth=["\']?([\d.]+)', r_tag)
                        hm = re.search(r'\bheight=["\']?([\d.]+)', r_tag)
                        if xm and ym and wm and hm:
                            rx = float(xm.group(1)) + dx_trim
                            ry = float(ym.group(1)) + dy_trim
                            rw = float(wm.group(1)) * trim_w_ratio
                            rh = float(hm.group(1)) * trim_h_ratio
                            svg_content = svg_content.replace(tm.group(0), "", 1)
                            svg_content = svg_content.replace(r_tag, _make_image_tag(rx, ry, rw, rh), 1)
                            replaced = True
                            break

        # Method 3: Deterministic coordinate fallback (append cleanly before </svg>)
        if not replaced:
            rx = info["x1"] * scale_x
            ry = info["y1"] * scale_y
            rw = info["width"] * scale_x
            rh = info["height"] * scale_y
            svg_content = svg_content.replace("</svg>", f"  {_make_image_tag(rx, ry, rw, rh)}\n</svg>", 1)

        replaced_count += 1

    if clip_defs:
        defs_chunk = "\n".join(clip_defs)
        if "</defs>" in svg_content:
            svg_content = svg_content.replace("</defs>", f"{defs_chunk}\n</defs>", 1)
        else:
            svg_content = re.sub(
                r"(<svg\b[^>]*>)",
                rf"\1\n<defs>\n{defs_chunk}\n</defs>",
                svg_content,
                count=1,
                flags=re.IGNORECASE,
            )

    return svg_content, replaced_count


# ============================================================================
# Step 5.5: Post-Swap Visual Polish Pass (Catches Ghost Lines / Double Text!)
# ============================================================================


def _dehydrate_base64_images(svg_code: str) -> tuple[str, dict[str, str]]:
    """Temporarily replace `data:image/png;base64,...` strings with compact `__B64_IMG_001__` tokens."""
    mapping: dict[str, str] = {}
    counter = [0]

    def _repl(m: re.Match[str]) -> str:
        counter[0] += 1
        token = f"__B64_IMG_{counter[0]:03d}__"
        mapping[token] = m.group(0)
        return token

    compact = re.sub(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", _repl, svg_code)
    return compact, mapping


def _rehydrate_base64_images(compact_svg: str, mapping: dict[str, str]) -> tuple[str, list[str]]:
    """Restore all `__B64_IMG_...__` tokens and report any accidentally dropped tokens."""
    missing = [tok for tok in mapping if tok not in compact_svg]
    out = compact_svg
    for tok, uri in mapping.items():
        out = out.replace(tok, uri)
    return out, missing


def polish_final_svg_after_icon_swap(
    client: Any,
    figure_path: Path,
    hydrated_svg: str,
    viewbox: tuple[int, int],
    output_dir: Path,
    *,
    inventory: Optional[list[str]] = None,
    resolved_density: str = "concise",
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[str, dict[str, Any]]:
    """Run 1 surgical post-swap visual inspection on the REAL hydrated `final_pre_polish.png`.

    Because `optimize_autofigure_template` only sees grey `<AF>` boxes, this post-swap pass inspects
    the ACTUAL transparent icons composited onto the SVG and eliminates:
      1. Duplicate vector ghost lines / 3D edges peeking out behind transparent `<image>` assets
      2. Double/overlapping text between vector `<text>` and nearby icons
      3. Minor arrow collisions or sub-floor font sizes reported by `lint_svg`
    """
    w, h = viewbox
    pre_png = output_dir / "final_pre_polish.png"
    renderer = render_svg_to_png(hydrated_svg, pre_png, width=min(w, 2200))
    lint_before = lint_svg(
        hydrated_svg,
        exact_text_inventory=inventory or None,
        expected_viewbox=viewbox,
        rendered_png_path=pre_png,
        density=resolved_density,
    )

    compact_svg, b64_map = _dehydrate_base64_images(hydrated_svg)
    if on_progress:
        on_progress(
            f"AutoFigure Step 5.5 (Post-Swap Polish): Auditing final composited render ({len(b64_map)} icons, "
            f"{len(lint_before.errors)} lint errors)..."
        )

    polish_prompt = f"""You are a Principal Scientific SVG Finisher performing a **Final Post-Swap Visual Audit** on our composited editable SVG (`viewBox="0 0 {w} {h}"`).

Compare **Image 2 (`final_pre_polish.png`, our real composited SVG with all transparent icons embedded)** side-by-side against **Image 1 (`figure.png`, original target)**.

FIX EVERY REAL VISUAL DEFECT YOU SEE IN IMAGE 2:
1. **DELETE GHOST / DUPLICATE VECTOR ARTWORK BEHIND TRANSPARENT ICONS**: Inspect around EVERY embedded `<image href="__B64_IMG_...__"/>`! Are there ANY stray black zig-zag lines, duplicate 3D block outlines, stray arrows, or extra shapes peeking out behind/around a transparent icon? **DELETE those stray vector elements immediately!**
2. **FIX DOUBLE / OVERLAPPING TEXT (NEVER DELETE INVENTORY LABELS!)**: Does ANY vector `<text>` overlap an icon, collide with another word, or sit on top of partially clipped raster letters? NEVER delete a required `[EXACT TEXT INVENTORY]` vector `<text>`! Instead, trim/reposition the colliding `<image>` slightly or place a tiny clean background patch behind the crisp editable vector `<text>` so every single required label renders 100% sharp and unclipped!
3. **FIX EVERY PROGRAMMATIC LINTER ERROR BELOW** (e.g., bump sub-floor `font-size` up to the required minimum px so 2-column print stays crisp!).
4. **PROTECT EVERY SINGLE EMBEDDED ICON**: Keep ALL `{len(b64_map)}` `href="__B64_IMG_...__"` `<image>` elements intact! Do NOT delete or alter any `__B64_IMG_...__` token!

=== PROGRAMMATIC LINTER REPORT ===
{lint_before.to_prompt_feedback()}

=== CURRENT SVG SOURCE (WITH COMPACT ICON TOKENS) ===
```xml
{compact_svg}
```

Output ONLY the complete polished valid SVG code starting with `<svg` and ending with `</svg>`. NO markdown fences, NO commentary."""

    try:
        raw = client.generate_multimodal(
            [
                polish_prompt,
                f"Image 1 — Original Target Figure (`figure.png`, {w}x{h}):",
                figure_path,
                f"Image 2 — Current Composited Final SVG (`final_pre_polish.png`):",
                pre_png,
            ],
            max_output_tokens=64000,
            temperature=0.1,
        )
    except Exception as exc:
        return hydrated_svg, {
            "round": "post_swap_polish",
            "renderer": renderer,
            "accepted": False,
            "reject_reason": f"Post-swap polish skipped on API error: {exc}",
        }

    candidate_compact = normalize_svg_root(extract_svg_block(raw), w, h)
    if not candidate_compact:
        return hydrated_svg, {
            "round": "post_swap_polish",
            "renderer": renderer,
            "accepted": False,
            "reject_reason": "No <svg> block extracted in post-swap polish.",
        }

    ok, errs = validate_svg_xml(candidate_compact)
    if not ok:
        candidate_compact = fix_svg_xml_with_llm(client, candidate_compact, errs, width=w, height=h)
        ok, errs = validate_svg_xml(candidate_compact)
    if not ok:
        return hydrated_svg, {
            "round": "post_swap_polish",
            "renderer": renderer,
            "accepted": False,
            "reject_reason": f"Post-swap polish produced invalid XML: {errs[:2]}",
        }

    rehydrated, missing_tokens = _rehydrate_base64_images(candidate_compact, b64_map)
    if missing_tokens:
        return hydrated_svg, {
            "round": "post_swap_polish",
            "renderer": renderer,
            "accepted": False,
            "reject_reason": f"Post-swap polish dropped {len(missing_tokens)} icon token(s); keeping pre-polish SVG.",
        }

    return rehydrated, {
        "round": "post_swap_polish",
        "renderer": renderer,
        "accepted": True,
        "reject_reason": "",
    }


# ============================================================================
# Unified AutoFigure-Edit Engine Entry Point (Stage 5 + 6 Replacement)
# ============================================================================


def run_autofigure_svg_engine(
    client: Any,
    figure_path: Path,
    output_dir: Path,
    *,
    schema_md: str = "",
    font_family: str = "Inter, Helvetica, Arial, sans-serif",
    svg_mode: str = "hybrid_rich",
    density: str = "auto",
    seg_backend: str = "auto",
    matting_backend: str = "auto",
    sam_prompts: str = "icon,portrait,diagram,chart,object",
    sam_min_score: float = 0.45,
    sam_merge_threshold: float = 0.85,
    sam_api_key: str = "",
    sam_max_masks: int = 32,
    rmbg_model_path: str = "",
    enable_upscale_4k: bool = False,
    optimize_iterations: int = 2,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Run the full native AutoFigure-Edit figure -> editable SVG engine.

    Steps:
      1. Optional 4K proportional upscale (auto-enabled when SAM3 is active or `--upscale-4k` is set)
      2. Icon detection -> `samed.png` + `boxlib.json` (SAM3 local/fal/roboflow or Zero-GPU VLM fallback)
      3. Crop + transparent matting -> `icons/icon_AF.._nobg.png` (RMBG-2.0 / BiRefNet / FloodFill)
      4. Clean `<AF>01` Template SVG generation -> `template.svg`
      5. Multi-round visual template optimization using Windows zero-DLL `resvg-py` -> `optimized_template.svg`
      6. Deterministic coordinate-aligned `<AF>01` -> base64 `<image>` replacement -> `final.svg` + 2× PNG + 2-col preview
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = extract_exact_text_inventory(schema_md) if schema_md else []
    resolved_density = resolve_density_mode(density, len(inventory))

    resolved_seg = resolve_seg_backend(seg_backend, sam_api_key)
    should_upscale = enable_upscale_4k or resolved_seg in ("sam3_local", "sam3_fal", "sam3_roboflow")

    work_figure_path = output_dir / "figure.png"
    with Image.open(figure_path) as src_im:
        img = src_im.convert("RGB")
    if should_upscale:
        img, did_upscale = upscale_image_to_4k_if_needed(img)
        if did_upscale and on_progress:
            on_progress(f"AutoFigure Step 1: Upscaled anchor proportionally to 4K ({img.width}x{img.height}).")
    img.save(work_figure_path, format="PNG")
    viewbox = (img.width, img.height)

    # Step 2 + 3: Segmentation & Matting (skipped in pure_vector mode)
    boxes: list[dict[str, Any]] = []
    icon_infos: list[dict[str, Any]] = []
    used_seg = "disabled_pure_vector"
    used_matting = "disabled_pure_vector"
    samed_path = output_dir / "samed.png"

    if svg_mode == "hybrid_rich":
        samed_path, _boxlib_path, boxes, used_seg = segment_icons_autofigure(
            client,
            work_figure_path,
            output_dir,
            seg_backend=seg_backend,
            sam_prompts=sam_prompts,
            min_score=sam_min_score,
            merge_threshold=sam_merge_threshold,
            sam_api_key=sam_api_key,
            sam_max_masks=sam_max_masks,
            on_progress=on_progress,
        )
        icon_infos, used_matting = crop_and_matte_icons(
            work_figure_path,
            boxes,
            output_dir,
            matting_backend=matting_backend,
            rmbg_model_path=rmbg_model_path,
            on_progress=on_progress,
        )
    else:
        shutil.copyfile(work_figure_path, samed_path)

    # Step 4: Clean <AF>01 Template SVG Reconstruction
    if on_progress:
        on_progress(
            f"AutoFigure Step 4: Reconstructing clean <AF> template SVG ({viewbox[0]}x{viewbox[1]}, {len(boxes)} icons)..."
        )
    template_path = output_dir / "template.svg"
    template_svg = generate_autofigure_svg_template(
        client,
        work_figure_path,
        samed_path,
        boxes,
        template_path,
        font_family=font_family,
        exact_text_inventory=inventory or None,
        resolved_density=resolved_density,
    )
    (output_dir / "raw.svg").write_text(template_svg, encoding="utf-8")

    # Step 4.6: Multi-round Visual Template Optimization (via resvg-py)
    # When optimize_iterations >= 2, spend (optimize_iterations - 1) round(s) on grey-box template alignment
    # and reserve the final round for Step 5.5 Post-Swap Visual Polish on the REAL composited icons!
    opt_iters = max(0, int(optimize_iterations))
    template_rounds = max(0, opt_iters - 1) if (opt_iters >= 2 and len(icon_infos) > 0) else opt_iters
    do_post_swap_polish = opt_iters >= 1 and len(icon_infos) > 0

    optimized_svg, round_logs = optimize_autofigure_template(
        client,
        work_figure_path,
        samed_path,
        template_svg,
        boxes,
        output_dir,
        max_iterations=template_rounds,
        on_progress=on_progress,
    )

    # Step 5: Deterministic Coordinate-Aligned Icon Replacement
    if on_progress:
        on_progress(f"AutoFigure Step 5: Deterministically embedding {len(icon_infos)} transparent icon(s) into final.svg...")
    final_svg, replaced_count = replace_icons_in_svg(optimized_svg, icon_infos, viewbox)

    ok_xml, xml_errs = validate_svg_xml(final_svg)
    if not ok_xml:
        final_svg = fix_svg_xml_with_llm(client, final_svg, xml_errs, width=viewbox[0], height=viewbox[1])

    # Step 5.5: Post-Swap Visual Polish Pass on the REAL composited figure (catches ghost lines/double text!)
    if do_post_swap_polish:
        final_svg, polish_log = polish_final_svg_after_icon_swap(
            client,
            work_figure_path,
            final_svg,
            viewbox,
            output_dir,
            inventory=inventory or None,
            resolved_density=resolved_density,
            on_progress=on_progress,
        )
        round_logs.append(polish_log)

    final_svg_path = output_dir / "final.svg"
    final_svg_path.write_text(final_svg, encoding="utf-8")

    final_png_path = output_dir / "final.png"
    twocol_png_path = output_dir / "print_preview_2col.png"
    renderer_used = render_dual_preview(final_svg, final_png_path, twocol_png_path, viewbox=viewbox, full_zoom=2.0)

    final_lint: LintReport = lint_svg(
        final_svg,
        exact_text_inventory=inventory or None,
        expected_viewbox=viewbox,
        rendered_png_path=final_png_path,
        density=resolved_density,
    )
    qa_md = _build_autofigure_qa_report(
        final_lint=final_lint,
        round_logs=round_logs,
        renderer=renderer_used,
        viewbox=viewbox,
        seg_backend=used_seg,
        matting_backend=used_matting,
        embedded_count=replaced_count,
    )
    qa_path = output_dir / "qa.md"
    qa_path.write_text(qa_md, encoding="utf-8")

    return {
        "final_svg": final_svg_path,
        "final_png": final_png_path,
        "print_preview_2col": twocol_png_path,
        "qa_report": qa_path,
        "samed_path": samed_path,
        "template_svg": template_path,
        "viewbox": viewbox,
        "embedded_assets_count": replaced_count,
        "seg_backend": used_seg,
        "matting_backend": used_matting,
        "round_logs": round_logs,
    }


def _build_autofigure_qa_report(
    *,
    final_lint: LintReport,
    round_logs: list[dict[str, Any]],
    renderer: str,
    viewbox: tuple[int, int],
    seg_backend: str,
    matting_backend: str,
    embedded_count: int,
) -> str:
    w, h = viewbox
    lines = [
        "# Paper2SVG-ultra v2.0 Quality Audit Report (`qa.md`)",
        "",
        f"- **SVG Engine**: `AutoFigure-Edit Native (<AF> Template -> Optimize -> Deterministic Swap)`",
        f"- **Segmentation Backend**: `{seg_backend}`",
        f"- **Matting Backend**: `{matting_backend}`",
        f"- **SVG Renderer**: `{renderer}` (zero-DLL Windows native)",
        f"- **Final ViewBox**: `{w} x {h}` (aspect ratio `{w / max(1, h):.2f}:1`)",
        f"- **Embedded Visual Assets**: `{embedded_count}`",
        f"- **XML Valid**: `{final_lint.valid_xml}`",
        f"- **Hard Linter Errors**: `{len(final_lint.errors)}` | **Warnings**: `{len(final_lint.warnings)}`",
        f"- **Stats**: `{json.dumps(final_lint.stats, ensure_ascii=False)}`",
        "",
        "## Template Visual Optimization Rounds",
    ]
    if not round_logs:
        lines.append("- Optimization skipped (`optimize_iterations=0`).")
    for entry in round_logs:
        lines.append(
            f"- **Round {entry['round']}**: Accepted=`{entry['accepted']}` | Renderer=`{entry.get('renderer', renderer)}`"
            + (f" | Note: {entry['reject_reason']}" if entry.get("reject_reason") else "")
        )
    if final_lint.errors:
        lines.append("\n## Linter Errors")
        lines.extend(f"- ❌ {e}" for e in final_lint.errors)
    if final_lint.warnings:
        lines.append("\n## Linter Warnings")
        lines.extend(f"- ⚠️ {wn}" for wn in final_lint.warnings)
    return "\n".join(lines) + "\n"
