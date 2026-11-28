"""Paper2SVG-ultra v3.0 — Pure AutoFigure-Edit Native Vector Engine.

Clean, single-backbone pipeline (Step 2 -> Step 3 -> Step 4 -> Step 4.6 -> Step 5 -> Step 5.5):
  - Step 2 (Icon Segmentation -> `samed.png` + `boxlib.json`):
      `sam3_local` -> `sam3_fal` -> `sam3_roboflow` -> local CUDA `grounding_dino` + frame-stripped CC -> zero-GPU CC
      (with strict VLM semantic icon filter preventing text/panel swallowing).
  - Step 3 (Photo-Protected Matting -> `icons/icon_AF.._nobg.png`):
      Rectangular photos/portraits/heatmaps protected (`opaque_rect`), transparent icons batch-matted via
      `rmbg2` -> zero-token `birefnet` -> zero-GPU `floodfill`.
  - Step 4 (`<AF>` Template SVG) + Step 4.6 (Template Alignment via Windows zero-DLL `resvg-py`)
  - Step 5 (Deterministic `<AF>` -> base64 `<image>` Swap) + `postprocess_svg_deterministic` (0.05s)
  - Step 5.5 (1-Round Post-Swap Visual Polish on the REAL composited render) + final `postprocess_svg_deterministic`
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import PROJECT_ROOT, resolve_density_mode, resolve_stage_effort
from .icon_utils import (
    detect_cc_frame_stripped,
    looks_like_opaque_photo_tile,
    smart_remove_border_background,
    strip_edge_text_bleed,
    strip_text_edges_on_box,
)
from .svg_linter import LintReport, lint_svg, validate_svg_xml
from .svg_postprocess import (
    extract_svg_block,
    fix_svg_xml_with_llm,
    normalize_svg_root,
    postprocess_svg_deterministic,
    sanitize_svg_numeric_typos,
)
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
    """Upscale an image proportionally so its long edge reaches 4K (upstream AutoFigure-Edit logic)."""
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= 0 or long_edge >= target_long_edge:
        return image, False

    scale = target_long_edge / float(long_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return image.resize((new_width, new_height), resample=resample), True


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
# GPU Worker Probe & Backend Resolution
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
    return (
        any(snapshots_dir.glob("*/config.json"))
        or any(snapshots_dir.glob("*/*.safetensors"))
        or any(snapshots_dir.glob("*/*.pt"))
    )


def _worker_env(hf_endpoint: str = "") -> dict[str, str]:
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    repo_str = str(PROJECT_ROOT)
    env["PYTHONPATH"] = f"{repo_str}{os.pathsep}{existing_pp}" if existing_pp else repo_str
    ep = (hf_endpoint or env.get("HF_ENDPOINT") or "").strip()
    if ep:
        env["HF_ENDPOINT"] = ep
    return env


@lru_cache(maxsize=8)
def probe_gpu_capabilities(gpu_python: str = "") -> dict[str, Any]:
    """Probe CUDA/PyTorch/SAM3/Transformers/Kornia capabilities in-process or via configured `GPU_PYTHON`."""
    py = (gpu_python or os.environ.get("GPU_PYTHON") or "").strip()
    if py and Path(py).is_file():
        try:
            proc = subprocess.run(
                [py, "-m", "paper2svg.gpu_worker", "--task", "probe"],
                capture_output=True,
                text=True,
                timeout=30,
                env=_worker_env(),
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            pass

    # In-process check
    has_torch = importlib.util.find_spec("torch") is not None
    cuda = False
    gpu_name = ""
    if has_torch:
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
            if cuda:
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda = False
    return {
        "python": sys.executable,
        "torch": has_torch,
        "cuda": cuda,
        "gpu_name": gpu_name,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "sam3": importlib.util.find_spec("sam3") is not None,
        "kornia": importlib.util.find_spec("kornia") is not None,
        "timm": importlib.util.find_spec("timm") is not None,
        "rembg": importlib.util.find_spec("rembg") is not None,
    }


def _should_use_external_gpu_python(gpu_python: str = "") -> Optional[Path]:
    py = (gpu_python or os.environ.get("GPU_PYTHON") or "").strip()
    if not py:
        return None
    p = Path(py)
    if not p.is_file():
        return None
    try:
        if p.resolve() == Path(sys.executable).resolve():
            return None
    except Exception:
        pass
    return p


def resolve_seg_backend(
    requested: str = "auto",
    sam_api_key: str = "",
    *,
    gpu_python: str = "",
) -> str:
    """Resolve effective Step 2 backend: SAM3 (local/fal/roboflow) -> local CUDA GroundingDINO -> zero-GPU CC."""
    req = (requested or "auto").strip().lower()
    if req in ("sam3_local", "sam3_fal", "sam3_roboflow", "grounding_dino", "cc_zero_gpu"):
        return req

    caps = probe_gpu_capabilities(gpu_python)
    if caps.get("sam3") and caps.get("torch") and (_has_hf_cached_repo("facebook/sam3") or _get_hf_token()):
        return "sam3_local"
    if os.environ.get("FAL_KEY"):
        return "sam3_fal"
    if os.environ.get("ROBOFLOW_API_KEY"):
        return "sam3_roboflow"
    if sam_api_key:
        return "sam3_fal"
    if caps.get("cuda") and caps.get("transformers"):
        return "grounding_dino"
    return "cc_zero_gpu"


def resolve_matting_backend(
    requested: str = "auto",
    rmbg_model_path: str = "",
    *,
    gpu_python: str = "",
) -> str:
    """Resolve effective Step 3 backend: RMBG-2.0 -> zero-token BiRefNet (CUDA) -> zero-GPU floodfill."""
    req = (requested or "auto").strip().lower()
    if req in ("rmbg2", "birefnet", "floodfill"):
        return req

    caps = probe_gpu_capabilities(gpu_python)
    has_neural_deps = bool(caps.get("torch") and caps.get("transformers") and caps.get("kornia"))
    if has_neural_deps and (
        (rmbg_model_path and Path(rmbg_model_path).exists())
        or _has_hf_cached_repo("briaai/RMBG-2.0")
        or _get_hf_token()
    ):
        return "rmbg2"
    if (caps.get("cuda") and has_neural_deps) or caps.get("rembg"):
        return "birefnet"
    return "floodfill"


# ============================================================================
# Step 2: Icon Segmentation & Strict VLM Filter -> `samed.png` + `boxlib.json`
# ============================================================================


def _get_label_font(box_width: int, box_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def merge_overlapping_boxes(boxes: list[dict[str, Any]], overlap_threshold: float = 0.80) -> list[dict[str, Any]]:
    """Iteratively merge overlapping detected boxes and assign canonical `<AF>01`..`<AF>NN` labels."""
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
                        best = b1 if float(b1.get("score", 0.0)) >= float(b2.get("score", 0.0)) else b2
                        nb = {
                            "x1": min(b1["x1"], b2["x1"]),
                            "y1": min(b1["y1"], b2["y1"]),
                            "x2": max(b1["x2"], b2["x2"]),
                            "y2": max(b1["y2"], b2["y2"]),
                            "score": max(float(b1.get("score", 0.0)), float(b2.get("score", 0.0))),
                            "prompt": best.get("prompt", "icon"),
                            "bg_mode": best.get("bg_mode", "auto"),
                            "rounded_rx": int(best.get("rounded_rx", 0)),
                        }
                        working = [working[k] for k in range(n) if k != i and k != j]
                        working.append(nb)
                        merged = True
                        break

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
            }
        )
    return result


def draw_samed_placeholder_image(
    figure_img: Image.Image,
    boxes: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Draw exact AutoFigure-Edit `samed.png`: `#808080` gray fill, black 3px outline, white centered `<AF>01`."""
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


def _run_gpu_worker_detection(
    task: str,
    figure_path: Path,
    *,
    prompts: list[str],
    min_score: float,
    gpu_python: str = "",
    hf_endpoint: str = "",
) -> list[dict[str, Any]]:
    ext_py = _should_use_external_gpu_python(gpu_python)
    if ext_py is not None:
        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "boxes.json"
            cmd = [
                str(ext_py),
                "-m",
                "paper2svg.gpu_worker",
                "--task",
                task,
                "--image",
                str(figure_path),
                "--prompts",
                ",".join(prompts),
                "--min-score",
                str(min_score),
                "--out",
                str(out_json),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=_worker_env(hf_endpoint))
            if proc.returncode != 0 or not out_json.is_file():
                raise RuntimeError(f"GPU worker `{task}` failed: {(proc.stderr or proc.stdout)[-500:]}")
            return json.loads(out_json.read_text(encoding="utf-8"))

    from .gpu_worker import run_grounding_dino_gpu, run_sam3_local_gpu

    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
    if task == "sam3":
        return run_sam3_local_gpu(figure_path, prompts, min_score=min_score)
    return run_grounding_dino_gpu(figure_path)


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
                        {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "score": sc, "prompt": prompt}
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
                            }
                        )
    return detected


def vlm_filter_strict_icons(
    client: Any,
    figure_path: Path,
    candidates: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Fast 10s low-effort multimodal filter that keeps ONLY pure non-text visual icons/photos/3D graphics
    and drops every text card, arrow, or half-clipped fragment.
    """
    if not candidates:
        return []
    with Image.open(figure_path) as base_im:
        rgba = base_im.convert("RGBA")
        w, h = rgba.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        fnt = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 16)
    except OSError:
        fnt = ImageFont.load_default()

    for b in candidates:
        idx = b["id"] + 1
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        draw.rectangle([x1, y1, x2, y2], fill=(239, 68, 68, 38), outline=(220, 38, 38, 255), width=3)
        tag = f"#{idx}"
        tb = draw.textbbox((0, 0), tag, font=fnt)
        draw.rectangle([x1, max(0, y1 - 20), x1 + (tb[2] - tb[0]) + 8, max(20, y1)], fill=(220, 38, 38, 235))
        draw.text((x1 + 4, max(0, y1 - 19)), tag, fill=(255, 255, 255, 255), font=fnt)

    cand_preview = output_dir / "samed_candidates.png"
    Image.alpha_composite(rgba, overlay).convert("RGB").save(cand_preview, format="PNG")

    prompt = f"""You are a STRICT semantic icon filter for AutoFigure-Edit.
Image 1 is the clean figure. Image 2 shows {len(candidates)} numbered red candidate boxes (#1..#{len(candidates)}).

KEEP ONLY boxes covering a **pure non-text visual icon / 3D graphic / photograph / waveform / heatmap / scatter plot / chart** with **ZERO readable words inside**:
- KEEP: realistic photos (faces, chips, hardware), 3D voxel cubes/grids, cars/LiDAR, antenna towers, scatter/curve plots, waveforms, waterfalls, heatmaps, bar/radar charts, standalone icons (robot, calendar, document, books, database, signal antenna, magnifying glass, shield, satellite dish, gear, microchip, 3D cubes cluster, circular arrow WITHOUT swallowing words inside).
- STRICTLY DROP:
  * ANY box containing readable text words, titles, bullets, formulas, or card names!
  * ANY box covering an arrow line, border line, or plain flat colored rectangle/timeline bar/1D token strip!
  * ANY box that slices only half of an icon!

Return ONLY a compact JSON object like: {{"keep_ids": [2, 3, 6]}}"""

    raw = client.generate_multimodal(
        [prompt, "Image 1 (clean):", figure_path, "Image 2 (red candidates):", cand_preview],
        max_output_tokens=4000,
        reasoning_effort="low",
    )
    m = re.search(r"\{[^{}]*keep_ids[^{}]*\}", raw, re.DOTALL) or re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return candidates
    try:
        keep_set = {int(i) for i in json.loads(m.group(0)).get("keep_ids", [])}
    except Exception:
        return candidates
    kept = [b for b in candidates if (b["id"] + 1) in keep_set]
    return merge_overlapping_boxes(kept, overlap_threshold=0.72)


def segment_icons_autofigure(
    client: Any,
    figure_path: Path,
    output_dir: Path,
    *,
    seg_backend: str = "auto",
    sam_prompts: str = "icon,person,robot,animal,diagram,chart",
    min_score: float = 0.45,
    merge_threshold: float = 0.80,
    sam_api_key: str = "",
    sam_max_masks: int = 32,
    gpu_python: str = "",
    hf_endpoint: str = "",
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[Path, Path, list[dict[str, Any]], str]:
    """Run AutoFigure-Edit Step 2 icon segmentation -> `samed.png` + `boxlib.json`."""
    with Image.open(figure_path) as im:
        fig_img = im.convert("RGB")
    arr = np.asarray(fig_img, dtype=np.float32)
    w, h = fig_img.size

    resolved = resolve_seg_backend(seg_backend, sam_api_key, gpu_python=gpu_python)
    prompt_list = [p.strip() for p in sam_prompts.split(",") if p.strip()] or ["icon"]
    if on_progress:
        on_progress(f"Step 2: Segmenting visual icons via `{resolved}` ({w}x{h})...")

    raw_boxes: list[dict[str, Any]] = []
    needs_vlm_filter = False

    if resolved == "sam3_local":
        raw_boxes = _run_gpu_worker_detection(
            "sam3", figure_path, prompts=prompt_list, min_score=min_score, gpu_python=gpu_python, hf_endpoint=hf_endpoint
        )
    elif resolved == "sam3_fal":
        raw_boxes = _run_sam3_fal(fig_img, prompt_list, min_score, sam_api_key, sam_max_masks)
    elif resolved == "sam3_roboflow":
        raw_boxes = _run_sam3_roboflow(fig_img, prompt_list, min_score, sam_api_key)
    elif resolved == "grounding_dino":
        try:
            gd_boxes = _run_gpu_worker_detection(
                "grounding_dino",
                figure_path,
                prompts=prompt_list,
                min_score=min_score,
                gpu_python=gpu_python,
                hf_endpoint=hf_endpoint,
            )
            cc_boxes = detect_cc_frame_stripped(fig_img)
            raw_boxes = gd_boxes + cc_boxes
            needs_vlm_filter = True
        except Exception as exc:
            if on_progress:
                on_progress(f"Step 2: Local GroundingDINO failed ({exc}); falling back to `cc_zero_gpu`...")
            resolved = "cc_zero_gpu"
            raw_boxes = detect_cc_frame_stripped(fig_img)
            needs_vlm_filter = True
    else:
        raw_boxes = detect_cc_frame_stripped(fig_img)
        needs_vlm_filter = True

    canvas_area = float(max(1, w * h))
    filtered = [
        b
        for b in raw_boxes
        if (b["x2"] - b["x1"]) >= 16
        and (b["y2"] - b["y1"]) >= 16
        and ((b["x2"] - b["x1"]) * (b["y2"] - b["y1"])) / canvas_area <= 0.45
    ]

    merged = merge_overlapping_boxes(filtered, overlap_threshold=0.60 if needs_vlm_filter else merge_threshold)
    stripped = [strip_text_edges_on_box(arr, b) for b in merged]
    valid_boxes = merge_overlapping_boxes(stripped, overlap_threshold=0.65 if needs_vlm_filter else merge_threshold)

    if needs_vlm_filter and valid_boxes and client is not None:
        if on_progress:
            on_progress(f"Step 2: VLM semantic icon filter auditing {len(valid_boxes)} candidate box(es)...")
        valid_boxes = vlm_filter_strict_icons(client, figure_path, valid_boxes, output_dir)

    samed_path = draw_samed_placeholder_image(fig_img, valid_boxes, output_dir / "samed.png")
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
# Step 3: Photo-Protected Batch Neural Matting (RMBG-2.0 / BiRefNet / FloodFill)
# ============================================================================


def _run_gpu_worker_matting(
    pairs: list[tuple[Path, Path]],
    *,
    backend: str,
    rmbg_model_path: str = "",
    gpu_python: str = "",
    hf_endpoint: str = "",
) -> int:
    if not pairs:
        return 0
    ext_py = _should_use_external_gpu_python(gpu_python)
    if ext_py is not None:
        with tempfile.TemporaryDirectory() as td:
            man_path = Path(td) / "matte_manifest.json"
            man_path.write_text(
                json.dumps([{"src": str(s), "dst": str(d)} for s, d in pairs], ensure_ascii=False),
                encoding="utf-8",
            )
            cmd = [
                str(ext_py),
                "-m",
                "paper2svg.gpu_worker",
                "--task",
                backend,
                "--manifest",
                str(man_path),
                "--rmbg-model-path",
                rmbg_model_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=_worker_env(hf_endpoint))
            if proc.returncode != 0:
                raise RuntimeError(f"GPU worker matting `{backend}` failed: {(proc.stderr or proc.stdout)[-500:]}")
            return len(pairs)

    if backend == "birefnet" and importlib.util.find_spec("rembg") is not None:
        import rembg

        for src, dst in pairs:
            rembg.remove(Image.open(src).convert("RGBA")).save(dst, format="PNG")
        return len(pairs)

    from .gpu_worker import run_neural_matting_batch_gpu

    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
    return run_neural_matting_batch_gpu(pairs, backend=backend, rmbg_model_path=rmbg_model_path)


def crop_and_matte_icons(
    figure_path: Path,
    boxes: list[dict[str, Any]],
    output_dir: Path,
    *,
    matting_backend: str = "auto",
    rmbg_model_path: str = "",
    gpu_python: str = "",
    hf_endpoint: str = "",
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[list[dict[str, Any]], str]:
    """Run AutoFigure-Edit Step 3: Crop boxes, protect real photos (`opaque_rect`), and batch-matte transparent icons."""
    icons_dir = output_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_matting_backend(matting_backend, rmbg_model_path, gpu_python=gpu_python)

    if not boxes:
        return [], resolved

    if on_progress:
        on_progress(f"Step 3: Cropping & matting {len(boxes)} visual icon(s) via `{resolved}`...")

    with Image.open(figure_path) as im:
        fig_img = im.convert("RGB")

    # First pass: crop all boxes, separate opaque photo tiles from transparent-background icons
    icon_infos: list[dict[str, Any]] = []
    neural_batch: list[tuple[int, Path, Path]] = []

    for box in boxes:
        label = box["label"]
        label_clean = box.get("label_clean") or label.replace("<", "").replace(">", "")
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        cropped = fig_img.crop((x1, y1, x2, y2))
        crop_path = icons_dir / f"icon_{label_clean}.png"
        nobg_path = icons_dir / f"icon_{label_clean}_nobg.png"
        cropped.save(crop_path, format="PNG")

        bg_mode = str(box.get("bg_mode", "auto")).lower()
        is_opaque_photo = (bg_mode == "opaque_rect") or (bg_mode == "auto" and looks_like_opaque_photo_tile(cropped))

        info = {
            "id": box["id"],
            "label": label,
            "label_clean": label_clean,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": max(1, x2 - x1),
            "height": max(1, y2 - y1),
            "orig_x1": x1,
            "orig_y1": y1,
            "orig_width": max(1, x2 - x1),
            "orig_height": max(1, y2 - y1),
            "bg_mode": "opaque_rect" if is_opaque_photo else "remove_bg",
            "rounded_rx": int(box.get("rounded_rx", 4 if is_opaque_photo else 0)),
            "crop_path": str(crop_path),
            "nobg_path": str(nobg_path),
        }

        if is_opaque_photo:
            cropped.convert("RGBA").save(nobg_path, format="PNG")
        elif resolved in ("rmbg2", "birefnet"):
            # Pre-strip any tiny edge caption sliver before neural matting
            stripped, trims = strip_edge_text_bleed(cropped)
            if trims != (0, 0, 0, 0):
                stripped.convert("RGB").save(crop_path, format="PNG")
                info["x1"] += trims[0]
                info["y1"] += trims[1]
                info["x2"] -= trims[2]
                info["y2"] -= trims[3]
                info["width"] = max(1, info["x2"] - info["x1"])
                info["height"] = max(1, info["y2"] - info["y1"])
            neural_batch.append((len(icon_infos), crop_path, nobg_path))
        else:
            stripped, bleed_trims = strip_edge_text_bleed(cropped)
            nobg_img, ff_trims = smart_remove_border_background(stripped)
            nobg_img.save(nobg_path, format="PNG")
            info["x1"] += bleed_trims[0] + ff_trims[0]
            info["y1"] += bleed_trims[1] + ff_trims[1]
            info["x2"] -= bleed_trims[2] + ff_trims[2]
            info["y2"] -= bleed_trims[3] + ff_trims[3]
            info["width"] = max(1, info["x2"] - info["x1"])
            info["height"] = max(1, info["y2"] - info["y1"])

        icon_infos.append(info)

    # Single batched GPU matting call for all non-photo icons (loads weights ONCE!)
    if neural_batch:
        try:
            _run_gpu_worker_matting(
                [(src, dst) for _, src, dst in neural_batch],
                backend=resolved,
                rmbg_model_path=rmbg_model_path,
                gpu_python=gpu_python,
                hf_endpoint=hf_endpoint,
            )
        except Exception as exc:
            if on_progress:
                on_progress(f"Step 3: Neural matting `{resolved}` failed ({exc}); falling back to zero-GPU `floodfill`...")
            resolved = "floodfill"
            for idx, src, dst in neural_batch:
                im_crop = Image.open(src).convert("RGB")
                nobg_img, ff_trims = smart_remove_border_background(im_crop)
                nobg_img.save(dst, format="PNG")
                info = icon_infos[idx]
                info["x1"] += ff_trims[0]
                info["y1"] += ff_trims[1]
                info["x2"] -= ff_trims[2]
                info["y2"] -= ff_trims[3]
                info["width"] = max(1, info["x2"] - info["x1"])
                info["height"] = max(1, info["y2"] - info["y1"])

    return icon_infos, resolved


# ============================================================================
# Step 4: Clean AutoFigure-Edit `<AF>01` Template SVG Reconstruction
# ============================================================================


def generate_autofigure_svg_template(
    client: Any,
    figure_path: Path,
    samed_path: Path,
    boxes: list[dict[str, Any]],
    output_path: Path,
    *,
    font_family: str = "Inter, Helvetica, Arial, sans-serif",
) -> str:
    """Exact AutoFigure-Edit Step 4 (`template.svg`) with deterministic arrow & box-fit guardrails."""
    with Image.open(figure_path) as im:
        w, h = im.size

    if not boxes:
        prompt = f"""Write clean, editable SVG code to achieve pixel-level faithful replication of Image 1 (`figure.png`).

CRITICAL ENGINEERING RULES:
1. Exact 1:1 dimensions: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">`.
2. **CLEAN ARROWS (NEVER COLLAPSE INTO WEDGES!)**:
   - Thin arrows: use `<line ... marker-end="url(#arrow)"/>` or open `<polyline fill="none" ... marker-end="url(#arrow)"/>`.
   - Thick block arrows: use a proper **7-point symmetric closed `<polygon>`**. NEVER write `M...h...l...v...z` in one short `<path>`!
3. **FIT ALL TEXT INSIDE ITS BOX**: Match Image 1's font-family (serif vs sans, default `{font_family}`), weights, and compact font-sizes so EVERY label fits cleanly inside its enclosing `<rect>` with >= 6px horizontal margin.
4. **CLEAN SUBSCRIPTS**: Reset baseline after every `<tspan dy="4">` via `<tspan dy="-4">`.
5. Do NOT add any gray `<AF>01` placeholder boxes.

Output ONLY valid SVG starting with `<svg` and ending with `</svg>`. NO markdown fences, NO commentary."""
    else:
        coord_lines = "\n".join(
            f"- {b['label_clean']} ({b['label']}): x={b['x1']}, y={b['y1']}, width={b['x2'] - b['x1']}, height={b['y2'] - b['y1']}"
            for b in boxes
        )
        prompt = f"""Write clean, editable SVG code to achieve **pixel-level faithful replication** of Image 1 (`figure.png`), replacing ONLY the icon/image areas covered by gray boxes in Image 2 (`samed.png`) with exact `<g id="AF..">` placeholders!

CRITICAL ENGINEERING RULES:
1. Exact 1:1 dimensions: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">`.
2. **EXACT `<g id="AF..">` PLACEHOLDERS** (matching Image 2 `samed.png`):
```xml
<g id="AF01">
  <rect x="100" y="50" width="80" height="80" fill="#808080" stroke="black" stroke-width="2"/>
  <text x="140" y="90" text-anchor="middle" dominant-baseline="middle" fill="white" font-size="16">&lt;AF&gt;01</text>
</g>
```
   **ZERO DUPLICATE ARTWORK BEHIND PLACEHOLDERS**: Never draw icons, 3D cubes, or lines behind any `<g id="AF..">` box.
3. **CLEAN ARROWS (NEVER COLLAPSE INTO WEDGES!)**:
   - Thin arrows: ALWAYS use `<line ... marker-end="url(#arrow)"/>` or open `<polyline fill="none" ... marker-end="url(#arrow)"/>`.
   - Thick block arrows: ALWAYS draw a proper **7-point symmetric closed `<polygon>`** (top shaft -> head base -> wing -> tip -> opposite wing -> head base -> bottom shaft). **NEVER combine a shaft + head + `z` in one short `<path d="M...h...l...v...z"/>`!**
4. **FIT ALL TEXT INSIDE ITS BOX**: Match Image 1's font-family (serif vs sans, fallback `{font_family}`), weights, and compact font-sizes so EVERY label fits cleanly INSIDE its enclosing `<rect>` with >= 6px horizontal margin and ZERO overlap against any `<g id="AF..">` placeholder!
5. **CLEAN SUBSCRIPTS**: Reset baseline after every `<tspan dy="4">` via `<tspan dy="-4">`. Escape XML entities strictly (`&amp;`, `&lt;`, `&gt;`).

EXACT PLACEHOLDER COORDINATES ({len(boxes)} boxes):
{coord_lines}

Output ONLY valid SVG starting with `<svg` and ending with `</svg>`. NO markdown fences, NO commentary."""

    base_eff = getattr(getattr(client, "settings", None), "reasoning_effort", "medium")
    svg_eff = resolve_stage_effort(base_eff, "medium")
    raw = client.generate_multimodal(
        [
            prompt,
            f"Image 1 — Target Figure (`figure.png`, {w}x{h}):",
            figure_path,
            f"Image 2 — Icon Placeholders (`samed.png`, {w}x{h}):",
            samed_path,
        ],
        max_output_tokens=48000,
        temperature=0.15,
        reasoning_effort=svg_eff,
    )
    svg_code = normalize_svg_root(extract_svg_block(raw), w, h)
    if not svg_code:
        raise RuntimeError(f"AutoFigure Step 4 returned no <svg> block. Preview: {raw[:400]}")

    ok, errs = validate_svg_xml(svg_code)
    if not ok:
        svg_code = fix_svg_xml_with_llm(client, svg_code, errs, width=w, height=h)

    svg_code, _ = postprocess_svg_deterministic(svg_code)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_code, encoding="utf-8")
    return svg_code


# ============================================================================
# Step 4.6: Template Alignment Optimization (Windows Zero-DLL `resvg-py`)
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
    max_iterations: int = 1,
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Upstream AutoFigure-Edit Step 4.6 running on `template.svg` via Windows zero-DLL `resvg-py`."""
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
            on_progress(f"Step 4.6 (Template Align {r}/{max_iterations}): Rendering template via resvg-py...")

        iter_svg = output_dir / f"iter_{r - 1}.svg"
        iter_png = output_dir / f"iter_{r - 1}_full.png"
        iter_svg.write_text(current_svg, encoding="utf-8")
        renderer_used = render_svg_to_png(current_svg, iter_png, width=min(w, 3840))

        placeholder_rule = (
            "- No icon placeholders exist; keep pure vector layout clean."
            if no_icon_mode
            else f"- Keep ALL {len(boxes)} `<g id=\"AF01\">`..`<g id=\"AF{len(boxes):02d}\">` placeholders intact (`#808080`, black border) and delete any vector art drawn behind them."
        )
        opt_prompt = f"""You are an expert SVG alignment optimizer (AutoFigure-Edit).
Compare Image 3 (current SVG rendered as PNG) against Image 1 (`figure.png`, target) and Image 2 (`samed.png`, icon placeholders).
Fix EVERY position/alignment/style mismatch across 8 checks:
- POSITION: (1) Icon `<AF>` alignment, (2) Text placement with zero overlaps/clipping, (3) Arrow endpoints & exact direction (`UP`/`DOWN`/`LEFT`/`RIGHT`), (4) Borders & dividers.
- STYLE: (5) Placeholder proportions, (6) Font family/sizes/weights matching Image 1, (7) Clean arrows (NEVER collapsed `M...h...l...v...z` wedges!), (8) Panel fills & stroke widths.
{placeholder_rule}

CURRENT SVG:
```xml
{current_svg}
```

Output ONLY valid SVG starting with `<svg` and ending with `</svg>`. NO markdown fences, NO commentary."""

        base_eff = getattr(getattr(client, "settings", None), "reasoning_effort", "medium")
        opt_eff = resolve_stage_effort(base_eff, "medium")
        try:
            raw = client.generate_multimodal(
                [
                    opt_prompt,
                    f"Image 1 — Target (`figure.png`, {w}x{h}):",
                    figure_path,
                    f"Image 2 — Placeholders (`samed.png`, {w}x{h}):",
                    samed_path,
                    f"Image 3 — Current Render (`iter_{r - 1}_full.png`):",
                    iter_png,
                ],
                max_output_tokens=48000,
                temperature=0.15,
                reasoning_effort=opt_eff,
            )
        except Exception as exc:
            round_logs.append({"round": f"template_{r}", "renderer": renderer_used, "accepted": False, "reject_reason": str(exc)})
            break

        candidate = normalize_svg_root(extract_svg_block(raw), w, h)
        accepted = False
        reject_reason = ""
        if not candidate:
            reject_reason = "No <svg> block extracted."
        else:
            ok, errs = validate_svg_xml(candidate)
            if not ok:
                candidate = fix_svg_xml_with_llm(client, candidate, errs, width=w, height=h)
                ok, errs = validate_svg_xml(candidate)
            if not ok:
                reject_reason = f"Invalid XML: {errs[:2]}"
            else:
                _, missing = _count_af_placeholders(candidate, boxes)
                if boxes and len(missing) > max(0, int(len(boxes) * 0.2)):
                    reject_reason = f"Dropped too many <AF> placeholders ({missing})."
                else:
                    current_svg, _ = postprocess_svg_deterministic(candidate)
                    accepted = True

        round_logs.append({"round": f"template_{r}", "renderer": renderer_used, "accepted": accepted, "reject_reason": reject_reason})

    (output_dir / "optimized_template.svg").write_text(current_svg, encoding="utf-8")
    return current_svg, round_logs


# ============================================================================
# Step 5: Deterministic Coordinate-Aligned Icon Replacement
# ============================================================================


def get_svg_dimensions(svg_code: str) -> tuple[Optional[float], Optional[float]]:
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

        dx_trim = (info["x1"] - info.get("orig_x1", info["x1"])) * scale_x
        dy_trim = (info["y1"] - info.get("orig_y1", info["y1"])) * scale_y
        trim_w_ratio = info["width"] / float(max(1, info.get("orig_width", info["width"])))
        trim_h_ratio = info["height"] / float(max(1, info.get("orig_height", info["height"])))

        replaced = False
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

        if not replaced:
            rx, ry = info["x1"] * scale_x, info["y1"] * scale_y
            rw, rh = info["width"] * scale_x, info["height"] * scale_y
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
# Step 5.5: Post-Swap Visual Polish Pass on REAL Composited Render
# ============================================================================


def _dehydrate_base64_images(svg_code: str) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}
    counter = [0]

    def _repl(m: re.Match[str]) -> str:
        counter[0] += 1
        token = f"__B64_IMG_{counter[0]:03d}__"
        mapping[token] = m.group(0)
        return token

    return re.sub(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", _repl, svg_code), mapping


def _rehydrate_base64_images(compact_svg: str, mapping: dict[str, str]) -> tuple[str, list[str]]:
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
    resolved_density: str = "detailed",
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[str, dict[str, Any]]:
    """Run 1 surgical post-swap visual inspection on the REAL composited `final_pre_polish.png`."""
    w, h = viewbox
    pre_png = output_dir / "final_pre_polish.png"
    renderer = render_svg_to_png(hydrated_svg, pre_png, width=min(w, 3840))
    lint_before = lint_svg(hydrated_svg, expected_viewbox=viewbox, rendered_png_path=pre_png, density=resolved_density)
    geo_errs = [
        e for e in lint_before.errors if any(k in e for k in ("Canvas clipping", "Text collision", "overflows its box", "XML"))
    ]
    issues_txt = "\n".join(f"- {e}" for e in geo_errs[:12]) if geo_errs else "- None."

    compact_svg, b64_map = _dehydrate_base64_images(hydrated_svg)
    if on_progress:
        on_progress(f"Step 5.5 (Post-Swap Polish): Auditing composited render ({len(b64_map)} icons, {len(geo_errs)} geo lint issues)...")

    polish_prompt = f"""You are a Principal Scientific SVG Finisher performing 1 surgical post-swap visual audit (`viewBox="0 0 {w} {h}"`).
Compare **Image 2 (`final_pre_polish.png`)** against **Image 1 (`figure.png`, target)** and fix ONLY real visual discrepancies:
1. Delete any ghost vector artwork peeking out behind `<image href="__B64_IMG_...__"/>` or double/overlapping text.
2. Fix any misaligned arrow endpoints, missing small icons/arrows, or text touching a nearby icon/border.
3. Keep ALL {len(b64_map)} `href="__B64_IMG_...__"` tokens intact!
4. NEVER write collapsed wedge arrow paths `M...h...l...v...z`.

DETECTED GEOMETRIC ISSUES:
{issues_txt}

CURRENT SVG:
```xml
{compact_svg}
```

Output ONLY valid SVG starting with `<svg` and ending with `</svg>`. NO markdown fences, NO commentary."""

    base_eff = getattr(getattr(client, "settings", None), "reasoning_effort", "medium")
    pol_eff = resolve_stage_effort(base_eff, "medium")
    try:
        raw = client.generate_multimodal(
            [
                polish_prompt,
                f"Image 1 — Target (`figure.png`, {w}x{h}):",
                figure_path,
                "Image 2 — Current Composited SVG (`final_pre_polish.png`):",
                pre_png,
            ],
            max_output_tokens=48000,
            temperature=0.1,
            reasoning_effort=pol_eff,
        )
    except Exception as exc:
        return hydrated_svg, {"round": "post_swap_polish", "renderer": renderer, "accepted": False, "reject_reason": str(exc)}

    candidate = sanitize_svg_numeric_typos(normalize_svg_root(extract_svg_block(raw), w, h))
    if not candidate:
        return hydrated_svg, {"round": "post_swap_polish", "renderer": renderer, "accepted": False, "reject_reason": "Empty SVG."}

    ok, errs = validate_svg_xml(candidate)
    if not ok:
        candidate = fix_svg_xml_with_llm(client, candidate, errs, width=w, height=h)
        ok, errs = validate_svg_xml(candidate)
    if not ok:
        return hydrated_svg, {"round": "post_swap_polish", "renderer": renderer, "accepted": False, "reject_reason": f"Invalid XML: {errs[:2]}"}

    rehydrated, missing = _rehydrate_base64_images(candidate, b64_map)
    if missing:
        return hydrated_svg, {"round": "post_swap_polish", "renderer": renderer, "accepted": False, "reject_reason": f"Dropped {len(missing)} icon(s)."}

    return rehydrated, {"round": "post_swap_polish", "renderer": renderer, "accepted": True, "reject_reason": ""}


# ============================================================================
# Unified AutoFigure-Edit Engine Entry Point (Steps 2 -> 5.5)
# ============================================================================


def run_autofigure_svg_engine(
    client: Any,
    figure_path: Path,
    output_dir: Path,
    *,
    font_family: str = "Inter, Helvetica, Arial, sans-serif",
    svg_mode: str = "hybrid_rich",
    density: str = "auto",
    seg_backend: str = "auto",
    matting_backend: str = "auto",
    gpu_python: str = "",
    hf_endpoint: str = "",
    sam_prompts: str = "icon,person,robot,animal,diagram,chart",
    sam_min_score: float = 0.45,
    sam_merge_threshold: float = 0.80,
    sam_api_key: str = "",
    sam_max_masks: int = 32,
    rmbg_model_path: str = "",
    enable_upscale_4k: bool = False,
    optimize_iterations: int = 2,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Run the full v3.0 figure -> editable SVG engine with deterministic post-processing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_density = resolve_density_mode(density, 40)

    resolved_seg = resolve_seg_backend(seg_backend, sam_api_key, gpu_python=gpu_python)
    should_upscale = enable_upscale_4k or resolved_seg in ("sam3_local", "sam3_fal", "sam3_roboflow")

    work_figure_path = output_dir / "figure.png"
    with Image.open(figure_path) as src_im:
        img = src_im.convert("RGB")
    if should_upscale:
        img, did_up = upscale_image_to_4k_if_needed(img)
        if did_up and on_progress:
            on_progress(f"Step 1: Upscaled figure proportionally to 4K ({img.width}x{img.height}).")
    img.save(work_figure_path, format="PNG")
    viewbox = (img.width, img.height)

    # Step 2 + 3: Icon Segmentation & Photo-Protected Matting
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
            gpu_python=gpu_python,
            hf_endpoint=hf_endpoint,
            on_progress=on_progress,
        )
        icon_infos, used_matting = crop_and_matte_icons(
            work_figure_path,
            boxes,
            output_dir,
            matting_backend=matting_backend,
            rmbg_model_path=rmbg_model_path,
            gpu_python=gpu_python,
            hf_endpoint=hf_endpoint,
            on_progress=on_progress,
        )
    else:
        img.save(samed_path, format="PNG")

    # Step 4: Clean `<AF>` Template SVG
    if on_progress:
        on_progress(f"Step 4: Generating editable <AF> template SVG ({viewbox[0]}x{viewbox[1]}, {len(boxes)} icons)...")
    template_path = output_dir / "template.svg"
    template_svg = generate_autofigure_svg_template(
        client,
        work_figure_path,
        samed_path,
        boxes,
        template_path,
        font_family=font_family,
    )

    # Step 4.6 + Step 5.5 Budget (default optimize_iterations=2 -> 1 template round + 1 post-swap polish!)
    opt_iters = max(0, int(optimize_iterations))
    if len(icon_infos) > 0:
        template_rounds = max(0, opt_iters - 1)
        do_post_swap_polish = opt_iters >= 1
    else:
        template_rounds = opt_iters
        do_post_swap_polish = False

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

    # Step 5: Deterministic Icon Replacement + Deterministic Geometry Post-Processor (0.05s)
    if on_progress:
        on_progress(f"Step 5: Embedding {len(icon_infos)} icon(s) + running deterministic arrow/text repair...")
    final_svg, replaced_count = replace_icons_in_svg(optimized_svg, icon_infos, viewbox)
    ok_xml, xml_errs = validate_svg_xml(final_svg)
    if not ok_xml:
        final_svg = fix_svg_xml_with_llm(client, final_svg, xml_errs, width=viewbox[0], height=viewbox[1])
    final_svg, pre_post_stats = postprocess_svg_deterministic(final_svg)
    (output_dir / "final_pre_polish.svg").write_text(final_svg, encoding="utf-8")

    # Step 5.5: Post-Swap Visual Polish + Final Deterministic Pass
    if do_post_swap_polish:
        final_svg, polish_log = polish_final_svg_after_icon_swap(
            client,
            work_figure_path,
            final_svg,
            viewbox,
            output_dir,
            resolved_density=resolved_density,
            on_progress=on_progress,
        )
        round_logs.append(polish_log)

    final_svg, final_post_stats = postprocess_svg_deterministic(final_svg)
    final_svg_path = output_dir / "final.svg"
    final_svg_path.write_text(final_svg, encoding="utf-8")

    final_png_path = output_dir / "final.png"
    twocol_png_path = output_dir / "print_preview_2col.png"
    renderer_used = render_dual_preview(final_svg, final_png_path, twocol_png_path, viewbox=viewbox, full_zoom=2.0)

    final_lint: LintReport = lint_svg(
        final_svg,
        expected_viewbox=viewbox,
        rendered_png_path=final_png_path,
        density=resolved_density,
    )
    qa_md = _build_qa_report(
        final_lint=final_lint,
        round_logs=round_logs,
        renderer=renderer_used,
        viewbox=viewbox,
        seg_backend=used_seg,
        matting_backend=used_matting,
        embedded_count=replaced_count,
        pre_post_stats=pre_post_stats,
        final_post_stats=final_post_stats,
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
        "postprocess_stats": {"pre_polish": pre_post_stats, "final": final_post_stats},
        "round_logs": round_logs,
    }


def _build_qa_report(
    *,
    final_lint: LintReport,
    round_logs: list[dict[str, Any]],
    renderer: str,
    viewbox: tuple[int, int],
    seg_backend: str,
    matting_backend: str,
    embedded_count: int,
    pre_post_stats: dict[str, int],
    final_post_stats: dict[str, int],
) -> str:
    w, h = viewbox
    lines = [
        "# Paper2SVG-ultra v3.0 Quality Audit Report (`qa.md`)",
        "",
        "- **Pipeline**: `Pure AutoFigure-Edit (Method Section + 1 Style Ref -> Step 1 -> Seg/Matting -> <AF> SVG -> Deterministic Repair -> Polish)`",
        f"- **Segmentation Backend**: `{seg_backend}`",
        f"- **Matting Backend**: `{matting_backend}`",
        f"- **SVG Renderer**: `{renderer}` (Windows zero-DLL native)",
        f"- **Final ViewBox**: `{w} x {h}` (aspect ratio `{w / max(1, h):.2f}:1`)",
        f"- **Embedded Visual Icons**: `{embedded_count}`",
        f"- **Deterministic Post-Processor**: pre-polish=`{pre_post_stats}` | final=`{final_post_stats}`",
        f"- **XML Valid**: `{final_lint.valid_xml}`",
        f"- **Hard Linter Errors**: `{len(final_lint.errors)}` | **Warnings**: `{len(final_lint.warnings)}`",
        "",
        "## Visual Optimization Rounds",
    ]
    if not round_logs:
        lines.append("- Optimization skipped (`optimize_iterations=0`).")
    for entry in round_logs:
        lines.append(
            f"- **Round `{entry['round']}`**: Accepted=`{entry['accepted']}` | Renderer=`{entry.get('renderer', renderer)}`"
            + (f" | Note: {entry['reject_reason']}" if entry.get("reject_reason") else "")
        )
    if final_lint.errors:
        lines.append("\n## Linter Errors")
        lines.extend(f"- {e}" for e in final_lint.errors)
    if final_lint.warnings:
        lines.append("\n## Linter Warnings")
        lines.extend(f"- {wn}" for wn in final_lint.warnings)
    return "\n".join(lines) + "\n"
