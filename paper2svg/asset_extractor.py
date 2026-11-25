"""Zero-GPU Visual Asset Localizer, Border-Flood-Fill Background Remover & SVG Hydration Engine.

Achieves AutoFigure-Edit's hybrid editable SVG richness (editable vector cards/text/arrows +
lossless high-res photos/3D visuals/textures/bespoke icons) WITHOUT needing SAM3, RMBG-2.0, or PyTorch.
"""

from __future__ import annotations

import base64
import io
import json
import re
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class ExtractedAsset:
    asset_id: str
    description: str
    bg_mode: str  # "opaque_rect" | "remove_bg"
    bbox_1000: list[int]  # [x1, y1, x2, y2] in 0..1000 normalized draft coords
    svg_x: float
    svg_y: float
    svg_w: float
    svg_h: float
    rounded_rx: int
    crop_path: str
    placeholder: str
    data_url: str

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "description": self.description,
            "bg_mode": self.bg_mode,
            "x": round(self.svg_x, 1),
            "y": round(self.svg_y, 1),
            "width": round(self.svg_w, 1),
            "height": round(self.svg_h, 1),
            "rounded_rx": self.rounded_rx,
            "href_placeholder": self.placeholder,
        }


def _load_ui_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a clean cross-platform sans-serif font (Windows / macOS / Linux)."""
    candidates = (
        [
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ]
        if bold
        else [
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Helvetica.ttc"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        ]
    )
    for fp in candidates:
        if fp.is_file():
            try:
                return ImageFont.truetype(str(fp), size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_coordinate_ruler_overlay(draft_path: Path, out_path: Path) -> Path:
    """Create a 1:1 identical-dimension inspection copy of stage1_draft.png with an edge-to-edge
    0..1000 coordinate grid (ZERO outer white margin!) so Image 1 and Image 2 share the EXACT same
    pixel coordinate space and VLMs never double-subtract a margin offset.
    """
    with Image.open(draft_path) as base_img:
        rgb = base_img.convert("RGBA")
        w, h = rgb.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_ui_font(max(11, min(15, int(round(min(w, h) * 0.017)))), bold=True)

    # Subtle outer border frame marking exact 0 and 1000 boundaries
    draw.rectangle([0, 0, w - 1, h - 1], outline=(220, 38, 38, 170), width=2)

    def _draw_tick_pill(x: int, y: int, label: str) -> None:
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad_x, pad_y = 3, 1
        bx0 = max(1, min(w - tw - pad_x * 2 - 2, x))
        by0 = max(1, min(h - th - pad_y * 2 - 2, y))
        bx1 = bx0 + tw + pad_x * 2
        by1 = by0 + th + pad_y * 2
        draw.rectangle([bx0, by0, bx1, by1], fill=(255, 255, 255, 225), outline=(220, 38, 38, 185), width=1)
        draw.text((bx0 + pad_x - tb[0], by0 + pad_y - tb[1]), label, fill=(185, 28, 28, 255), font=font)

    # Draw grid every 50 units in exact edge-to-edge 0..1000 space (red major @100, blue minor @50)
    for tick in range(50, 1000, 50):
        is_major = (tick % 100 == 0)
        alpha = 115 if is_major else 55
        color = (220, 38, 38, alpha) if is_major else (59, 130, 246, alpha)

        gx = int(round((w - 1) * tick / 1000.0))
        gy = int(round((h - 1) * tick / 1000.0))

        draw.line([(gx, 0), (gx, h - 1)], fill=color, width=1)
        draw.line([(0, gy), (w - 1, gy)], fill=color, width=1)

        if is_major:
            lbl = str(tick)
            tb = draw.textbbox((0, 0), lbl, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            # Place compact red ticks on ALL 4 edges for dead-accurate local coordinate reading
            _draw_tick_pill(gx - (tw + 6) // 2, 3, lbl)
            _draw_tick_pill(gx - (tw + 6) // 2, h - th - 7, lbl)
            _draw_tick_pill(3, gy - (th + 4) // 2, lbl)
            _draw_tick_pill(w - tw - 10, gy - (th + 4) // 2, lbl)

    combined = Image.alpha_composite(rgb, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path, format="PNG")
    return out_path


def smart_remove_border_background(crop_img: Image.Image, tolerance: float = 18.0) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Remove uniform card/canvas backdrop around an isolated visual asset using edge-seeded flood fill.

    Because flood-fill starts ONLY from the outer perimeter, interior light regions (e.g. white teeth,
    white marble veins, white robot bodies, bright icons) are 100% preserved!
    Returns (rgba_trimmed_img, (trim_left, trim_top, trim_right, trim_bottom)).
    """
    rgba = crop_img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32)
    h, w, _ = arr.shape
    if h < 8 or w < 8:
        return rgba, (0, 0, 0, 0)

    rgb = arr[:, :, :3]
    # Sample perimeter pixels to estimate the local card/canvas background color
    border_pixels = np.concatenate(
        [rgb[0, :, :], rgb[h - 1, :, :], rgb[:, 0, :], rgb[:, w - 1, :]],
        axis=0,
    )
    bg_color = np.median(border_pixels, axis=0)

    # Check if perimeter actually has a reasonably uniform background (skip if photo fills the whole frame)
    border_dists = np.linalg.norm(border_pixels - bg_color, axis=1)
    uniform_ratio = float(np.mean(border_dists < tolerance * 1.1))
    if uniform_ratio < 0.65:
        return rgba, (0, 0, 0, 0)

    dist_map = np.linalg.norm(rgb - bg_color, axis=2)
    bg_mask = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    # Seed from all perimeter pixels matching background
    for x in range(w):
        for y in (0, h - 1):
            if dist_map[y, x] <= tolerance and not visited[y, x]:
                visited[y, x] = True
                bg_mask[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if dist_map[y, x] <= tolerance and not visited[y, x]:
                visited[y, x] = True
                bg_mask[y, x] = True
                q.append((y, x))

    # 4-neighbor BFS flood-fill from perimeter
    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                if dist_map[ny, nx] <= tolerance:
                    bg_mask[ny, nx] = True
                    q.append((ny, nx))

    out = arr.copy()
    out[bg_mask, 3] = 0.0

    # Soft 1px anti-alias feathering on foreground pixels touching the removed background
    padded = np.pad(bg_mask, 1, mode="constant", constant_values=False)
    touches_bg = (
        padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
    ) & (~bg_mask)
    soft_zone = touches_bg & (dist_map < tolerance * 1.8)
    if np.any(soft_zone):
        alpha_scale = np.clip((dist_map[soft_zone] - tolerance * 0.85) / (tolerance * 0.95), 0.15, 1.0)
        out[soft_zone, 3] = np.round(out[soft_zone, 3] * alpha_scale)

    # Trim empty transparent edges so SVG placement aligns tightly to visible artwork
    alpha = out[:, :, 3]
    ys, xs = np.where(alpha > 15)
    if len(xs) < 16 or len(ys) < 16:
        return rgba, (0, 0, 0, 0)

    # Strip tiny disconnected text-bleed slivers touching the exact top/bottom raw crop edge
    # (e.g. top 4px of a caption grazed at y2, separated from the main graphic by a blank transparent gap).
    fg_mask = alpha > 15
    row_counts = np.sum(fg_mask, axis=1)
    total_fg = int(np.sum(fg_mask))
    # Find contiguous non-empty horizontal bands
    bands: list[tuple[int, int, int]] = []  # (r0, r1_incl, fg_pixels)
    in_band = False
    b_start = 0
    for r in range(h):
        if row_counts[r] > 0 and not in_band:
            in_band = True
            b_start = r
        elif row_counts[r] == 0 and in_band:
            in_band = False
            bands.append((b_start, r - 1, int(np.sum(row_counts[b_start:r]))))
    if in_band:
        bands.append((b_start, h - 1, int(np.sum(row_counts[b_start:h]))))

    if len(bands) >= 2 and total_fg > 0:
        # Check bottom-most band touching bottom edge (h-1) and top-most band touching top edge (0)
        main_band = max(bands, key=lambda b: b[2])
        for b0, b1, b_fg in list(bands):
            if (b0, b1, b_fg) == main_band:
                continue
            b_h = b1 - b0 + 1
            touches_top = b0 <= 1
            touches_bottom = b1 >= h - 2
            if (touches_top or touches_bottom) and b_h <= max(10, int(h * 0.14)) and b_fg < main_band[2] * 0.08:
                out[b0 : b1 + 1, :, 3] = 0.0
        alpha = out[:, :, 3]
        ys, xs = np.where(alpha > 15)
        if len(xs) < 16 or len(ys) < 16:
            return rgba, (0, 0, 0, 0)

    tx0, tx1 = int(xs.min()), int(xs.max()) + 1
    ty0, ty1 = int(ys.min()), int(ys.max()) + 1
    cleaned = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGBA")
    trimmed = cleaned.crop((tx0, ty0, tx1, ty1))
    return trimmed, (tx0, ty0, w - tx1, h - ty1)


def strip_edge_text_bleed(crop_img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Deterministically strip tiny clipped text slivers touching the top or bottom crop edge
    (e.g. top 4-6px of a caption grazed at y2, separated from the main graphic by >=2px of uniform
    white/transparent backdrop). Preserves photos and multi-row graphics untouched.
    """
    rgba = crop_img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32)
    h, w, _ = arr.shape
    if h < 28 or w < 24:
        return rgba, (0, 0, 0, 0)

    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    # Estimate canvas/card backdrop from the lightest 15% opaque pixels (usually white #FFFFFF or pale card)
    opaque_rgb = rgb[alpha > 15]
    if len(opaque_rgb) < 20:
        return rgba, (0, 0, 0, 0)
    bg_est = np.percentile(opaque_rgb, 92, axis=0)
    is_ink = (alpha > 15) & (np.linalg.norm(rgb - bg_est, axis=2) > 40.0)

    row_ink = np.sum(is_ink, axis=1)
    bands: list[tuple[int, int, int]] = []  # (r0, r1_incl, ink_pixels)
    in_band = False
    b_start = 0
    for r in range(h):
        if row_ink[r] > 0 and not in_band:
            in_band = True
            b_start = r
        elif row_ink[r] == 0 and in_band:
            in_band = False
            bands.append((b_start, r - 1, int(np.sum(row_ink[b_start:r]))))
    if in_band:
        bands.append((b_start, h - 1, int(np.sum(row_ink[b_start:h]))))

    if len(bands) < 2:
        return rgba, (0, 0, 0, 0)

    main_band = max(bands, key=lambda b: b[2])
    main_band_h = main_band[1] - main_band[0] + 1
    max_sliver_h = max(8, int(round(h * 0.13)))
    new_top = 0
    new_bottom = h

    # Check top-most sliver separated by >= 2 blank rows from next band
    first = bands[0]
    if first != main_band and first[0] <= 1:
        sliver_h = first[1] - first[0] + 1
        gap_after = bands[1][0] - first[1] - 1
        if (
            sliver_h <= max_sliver_h
            and sliver_h <= main_band_h * 0.25
            and gap_after >= 2
            and first[2] < main_band[2] * 0.25
        ):
            new_top = bands[1][0]

    # Check bottom-most sliver separated by >= 2 blank rows from previous band
    last = bands[-1]
    if last != main_band and last[1] >= h - 2:
        sliver_h = last[1] - last[0] + 1
        gap_before = last[0] - bands[-2][1] - 1
        if (
            sliver_h <= max_sliver_h
            and sliver_h <= main_band_h * 0.25
            and gap_before >= 2
            and last[2] < main_band[2] * 0.25
        ):
            new_bottom = bands[-2][1] + 1

    if new_top == 0 and new_bottom == h:
        return rgba, (0, 0, 0, 0)
    if new_bottom - new_top < 16:
        return rgba, (0, 0, 0, 0)

    trimmed = rgba.crop((0, new_top, w, new_bottom))
    return trimmed, (0, new_top, 0, h - new_bottom)


def _encode_pil_png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _crop_assets_from_specs(
    raw_assets: list[dict[str, Any]],
    draft_rgb: Image.Image,
    assets_dir: Path,
    viewbox: tuple[int, int],
    *,
    max_assets: int = 8,
) -> list[ExtractedAsset]:
    """Crop and package visual assets from normalized 0..1000 bbox specifications."""
    dw, dh = draft_rgb.size
    vb_w, vb_h = viewbox

    # Clean stale crops before writing the latest pass
    for old_png in assets_dir.glob("*.png"):
        try:
            old_png.unlink()
        except Exception:
            pass

    extracted: list[ExtractedAsset] = []
    used_ids: set[str] = set()

    for idx, item in enumerate(raw_assets[:max_assets], 1):
        bbox = item.get("bbox_1000")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1_n, y1_n, x2_n, y2_n = [int(round(float(v))) for v in bbox]
        except Exception:
            continue

        x1_n = max(0, min(999, x1_n))
        y1_n = max(0, min(999, y1_n))
        x2_n = max(x1_n + 10, min(1000, x2_n))
        y2_n = max(y1_n + 10, min(1000, y2_n))

        # Reject accidental whole-figure crops (> 55% of width & height)
        if (x2_n - x1_n) > 550 and (y2_n - y1_n) > 550:
            continue

        px1 = int(round(dw * x1_n / 1000.0))
        py1 = int(round(dh * y1_n / 1000.0))
        px2 = int(round(dw * x2_n / 1000.0))
        py2 = int(round(dh * y2_n / 1000.0))
        if (px2 - px1) < 20 or (py2 - py1) < 20:
            continue

        slug = re.sub(r"[^a-z0-9_]+", "_", str(item.get("asset_id") or f"asset_{idx}").lower()).strip("_")
        if not slug:
            slug = f"asset_{idx}"
        while slug in used_ids:
            slug = f"{slug}_{idx}"
        used_ids.add(slug)

        bg_mode = "remove_bg" if str(item.get("bg_mode", "")).strip().lower() == "remove_bg" else "opaque_rect"
        rounded_rx = int(item.get("rounded_rx") or (6 if bg_mode == "opaque_rect" else 0))

        crop = draft_rgb.crop((px1, py1, px2, py2))
        if bg_mode == "remove_bg":
            final_crop, (tl, tt, tr, tb) = smart_remove_border_background(crop)
            px1 += tl
            py1 += tt
            px2 -= tr
            py2 -= tb
        else:
            final_crop = crop.convert("RGBA")

        # Deterministically strip any tiny grazed caption/header text slivers separated by blank gap
        final_crop, (btl, btt, btr, btb) = strip_edge_text_bleed(final_crop)
        px1 += btl
        py1 += btt
        px2 -= btr
        py2 -= btb

        if px2 <= px1 or py2 <= py1:
            continue

        crop_file = assets_dir / f"{slug}.png"
        final_crop.save(crop_file, format="PNG", optimize=True)

        svg_x = (px1 / float(dw)) * vb_w
        svg_y = (py1 / float(dh)) * vb_h
        svg_w = ((px2 - px1) / float(dw)) * vb_w
        svg_h = ((py2 - py1) / float(dh)) * vb_h
        actual_bbox_1000 = [
            int(round(px1 * 1000.0 / float(dw))),
            int(round(py1 * 1000.0 / float(dh))),
            int(round(px2 * 1000.0 / float(dw))),
            int(round(py2 * 1000.0 / float(dh))),
        ]

        extracted.append(
            ExtractedAsset(
                asset_id=slug,
                description=str(item.get("description") or slug),
                bg_mode=bg_mode,
                bbox_1000=actual_bbox_1000,
                svg_x=svg_x,
                svg_y=svg_y,
                svg_w=svg_w,
                svg_h=svg_h,
                rounded_rx=max(0, min(20, rounded_rx)),
                crop_path=str(crop_file),
                placeholder=f"__ASSET_{slug}__",
                data_url=_encode_pil_png_data_url(final_crop),
            )
        )
    return extracted


def render_verification_overlay(
    draft_path: Path,
    assets: list[ExtractedAsset],
    viewbox: tuple[int, int],
    out_path: Path,
) -> Path:
    """Render a see-through red-bordered verification overlay on the exact (dw, dh) canvas
    so the VLM can inspect INSIDE and AROUND every proposed crop box for clipped headers or shifts.
    """
    with Image.open(draft_path) as base_img:
        rgba = base_img.convert("RGBA")
        dw, dh = rgba.size

    overlay = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_ui_font(13, bold=True)

    vb_w, vb_h = viewbox
    for idx, a in enumerate(assets, 1):
        x1 = int(round((a.svg_x / float(vb_w)) * dw))
        y1 = int(round((a.svg_y / float(vb_h)) * dh))
        x2 = int(round(((a.svg_x + a.svg_w) / float(vb_w)) * dw))
        y2 = int(round(((a.svg_y + a.svg_h) / float(vb_h)) * dh))
        draw.rectangle([x1, y1, x2, y2], fill=(239, 68, 68, 45), outline=(220, 38, 38, 255), width=3)
        # Compact badge tucked strictly INSIDE top-left corner so it NEVER hides titles above the box!
        tag = f"#{idx}"
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        lx = max(2, min(dw - tw - 8, x1 + 3))
        ly = max(2, min(dh - th - 6, y1 + 3))
        draw.rectangle([lx, ly, lx + tw + 6, ly + th + 4], fill=(220, 38, 38, 225))
        draw.text((lx + 3 - tb[0], ly + 2 - tb[1]), tag, fill=(255, 255, 255, 255), font=font)

    merged = Image.alpha_composite(rgba, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path, format="PNG")
    return out_path


def _verify_and_calibrate_asset_boxes(
    client: Any,
    draft_path: Path,
    ruler_path: Path,
    verify_path: Path,
    extracted: list[ExtractedAsset],
    *,
    max_assets: int = 8,
) -> tuple[bool, list[dict[str, Any]]]:
    """Perform a 1-pass visual alignment inspection on proposed crop boxes.

    Catches vertical/horizontal shifts and clipped adjacent text headers BEFORE Stage 5 runs.
    """
    current_specs = [
        {
            "index": idx,
            "asset_id": a.asset_id,
            "description": a.description,
            "bbox_1000": a.bbox_1000,
            "bg_mode": a.bg_mode,
            "rounded_rx": a.rounded_rx,
        }
        for idx, a in enumerate(extracted, 1)
    ]

    prompt = f"""You are performing a **Precision Crop Box Alignment Audit** before Hybrid SVG assembly.

Look carefully at ALL THREE attached images (ALL THREE share the EXACT same pixel dimensions, `0..1000` edge-to-edge, ZERO outer margins!):
- **Image 1**: Clean `stage1_draft.png`
- **Image 2**: `stage1_draft_ruler.png` (exact edge-to-edge `0..1000` red/blue grid)
- **Image 3**: `stage1_draft_verify.png` (the current proposed crop boxes drawn as translucent RED rectangles with compact `#1`, `#2`, ... corner badges!)

CURRENT PROPOSED BOXES:
```json
{json.dumps(current_specs, ensure_ascii=False, indent=2)}
```

INSPECT EVERY SINGLE RED RECTANGLE IN IMAGE 3 STRICTLY:
1. **TEXT CLIPPING CHECK (CRITICAL!)**: Does ANY red rectangle touch, graze, or cover even 2 pixels of ANY text title, card header, axis label, or caption directly above or below? If YES, tighten `y1`/`y2`/`x1`/`x2` by 5–15 units using Image 2's grid so ZERO text letters remain inside the red box!
2. **EDGE ALIGNMENT CHECK**: Does ANY red rectangle cut off the left/right/top/bottom border of a stacked card, spectrogram, or illustration, or include excess blank margin? If YES, snap `bbox_1000` tightly around the exact artwork bounds!
3. **DROP NON-ARTWORK**: If a red rectangle accidentally covers mostly plain text pills, plain boxes, or simple arrows that should be pure vector SVG instead of a raster crop, REMOVE that item from `assets`!
4. If ALL red rectangles are already tight, well-centered around pure visual artwork, and clip ZERO text, set `"needs_adjustment": false` and keep them unchanged.

Return ONLY valid JSON (no markdown fences):
{{
  "needs_adjustment": true,
  "audit_notes": ["brief note on what was fixed or why all boxes are clean"],
  "assets": [
    {{
      "asset_id": "short_snake_case_id",
      "description": "brief description",
      "bbox_1000": [x1, y1, x2, y2],
      "bg_mode": "opaque_rect | remove_bg",
      "rounded_rx": 6
    }}
  ]
}}
"""

    # Cap auxiliary mask-alignment audit at 'high' when global effort is 'xhigh' to keep turnaround crisp
    cur_eff = getattr(getattr(client, "settings", None), "reasoning_effort", "")
    audit_effort = "high" if str(cur_eff).strip().lower() == "xhigh" else None
    mm_kwargs: dict[str, Any] = {"max_output_tokens": 6000, "temperature": 0.1}
    if audit_effort:
        mm_kwargs["reasoning_effort"] = audit_effort

    raw = client.generate_multimodal(
        [
            prompt,
            "Image 1 — Clean Stage-1 Draft:",
            draft_path,
            "Image 2 — Edge-to-Edge 0..1000 Coordinate Ruler:",
            ruler_path,
            "Image 3 — Current Proposed Crop Boxes (Translucent Red Outlines):",
            verify_path,
        ],
        **mm_kwargs,
    )
    parsed = _parse_assets_json(raw)
    if not isinstance(parsed, dict) or "needs_adjustment" not in parsed:
        return False, current_specs
    needs_adj = bool(parsed.get("needs_adjustment"))
    new_assets = parsed.get("assets")
    if not needs_adj or not isinstance(new_assets, list):
        return False, current_specs
    return True, new_assets[:max_assets]


def detect_and_crop_visual_assets(
    client: Any,
    draft_path: Path,
    output_dir: Path,
    viewbox: tuple[int, int],
    *,
    max_assets: int = 8,
    verify_masks: bool = True,
) -> list[ExtractedAsset]:
    """Detect rich visual assets in stage1_draft.png using the multimodal model + zero-margin
    coordinate ruler, verify alignment via a translucent see-through mask overlay, and crop them
    losslessly at full resolution for hybrid editable SVG assembly.
    """
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    ruler_path = output_dir / "stage1_draft_ruler.png"
    render_coordinate_ruler_overlay(draft_path, ruler_path)

    with Image.open(draft_path) as d_img:
        draft_rgb = d_img.convert("RGB")
        dw, dh = draft_rgb.size

    prompt = f"""You are a Precision Visual Asset Segmenter converting a scientific overview figure into a **Hybrid Editable SVG** (like AutoFigure-Edit).

Look at BOTH attached images:
- **Image 1**: Clean `stage1_draft.png` ({dw} x {dh} px).
- **Image 2**: `stage1_draft_ruler.png` — the **EXACT SAME {dw} x {dh} px image** (ZERO outer white margin!) overlaid directly with an **edge-to-edge normalized 0..1000 coordinate grid** (red major lines every 100 units with red tick badges on all 4 edges; blue minor lines every 50 units).
  **DIRECT 1:1 COORDINATE RULE**: `0` is the exact left/top edge of the image and `1000` is the exact right/bottom edge of the image. Read `[x1, y1, x2, y2]` directly from the red/blue grid lines — DO NOT add or subtract any margin offset!

YOUR JOB:
Locate ONLY the genuinely rich visual artwork regions inside the figure that should be cropped losslessly from Image 1 and embedded as draggable `<image>` elements inside the editable SVG — **specifically**:
- Realistic photos or image samples (e.g., human faces, natural photos, medical scans, stone/marble slabs, satellite/scene crops, real screen reconstructions)
- 3D/isometric rendered objects (e.g., 3D meshes, voxels, optical setups, molecules, shaded geometric volumes, layered 3D tensors)
- Tactile material textures, spatial fields, real spectrograms/waterfalls/heatmaps/dense plots
- Complex multi-part physical equipment illustrations or expressive bespoke icons that cannot be drawn cleanly as simple SVG primitives

CRITICAL EXCLUSION RULES (DO NOT CROP THESE!):
1. **ZERO TEXT INSIDE CROPS (STRICT!)**: NEVER include ANY title, card header, sub-label, caption, or equation inside `bbox_1000`! Always start `y1` strictly BELOW any title/header above the graphic and end `y2` strictly ABOVE any caption below the graphic so 100% of text stays crisp native vector `<text>`!
2. **NEVER crop an entire module box, pastel panel, or text pill!** Only crop the pure graphic artwork inside.
3. **NEVER crop simple flat geometric vectors** (plain arrows, plain rectangles, simple 3-layer circle node-link graphs, simple 1-color lines/dots) — those render sharper as native SVG vectors!
4. If the figure is purely geometric vectors with no complex raster/photo/spectrogram/3D artwork worth cropping, return `{{"assets": []}}`.

For each asset, choose `bg_mode`:
- `"opaque_rect"`: for rectangular framed photos, spectrograms, heatmaps, or screen thumbnails.
- `"remove_bg"`: for irregular/floating illustrations, 3D objects, or equipment icons sitting on a uniform white or pale tinted card background.

Return ONLY valid JSON (no markdown fences) with at most {max_assets} assets:
{{
  "assets": [
    {{
      "asset_id": "short_snake_case_id",
      "description": "brief description of the graphic",
      "bbox_1000": [x1, y1, x2, y2],
      "bg_mode": "opaque_rect | remove_bg",
      "rounded_rx": 6
    }}
  ]
}}
"""

    raw = client.generate_multimodal(
        [
            prompt,
            "Image 1 — Clean Stage-1 Draft:",
            draft_path,
            "Image 2 — Edge-to-Edge 0..1000 Coordinate Ruler Overlay (Identical Dimensions, Zero Margin):",
            ruler_path,
        ],
        max_output_tokens=8000,
        temperature=0.1,
    )

    parsed = _parse_assets_json(raw)
    raw_assets = parsed.get("assets") or []

    extracted = _crop_assets_from_specs(
        raw_assets, draft_rgb, assets_dir, viewbox, max_assets=max_assets
    )

    if extracted and verify_masks:
        verify_path = output_dir / "stage1_draft_verify.png"
        render_verification_overlay(draft_path, extracted, viewbox, verify_path)
        adjusted, refined_specs = _verify_and_calibrate_asset_boxes(
            client, draft_path, ruler_path, verify_path, extracted, max_assets=max_assets
        )
        if adjusted:
            extracted = _crop_assets_from_specs(
                refined_specs, draft_rgb, assets_dir, viewbox, max_assets=max_assets
            )
            if extracted:
                render_verification_overlay(draft_path, extracted, viewbox, output_dir / "stage1_draft_verify_calibrated.png")

    # Save manifest (without giant base64 strings) for inspection & resume
    manifest_data = [
        {k: v for k, v in asdict(a).items() if k != "data_url"} for a in extracted
    ]
    (output_dir / "assets_manifest.json").write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if extracted:
        render_masked_draft_preview(draft_path, extracted, viewbox, output_dir / "stage1_draft_masked.png")
    return extracted


def render_masked_draft_preview(
    draft_path: Path,
    assets: list[ExtractedAsset],
    viewbox: tuple[int, int],
    out_path: Path,
) -> Path:
    """Create an AutoFigure-Edit style `samed.png` mask overlay showing exact cropped asset boxes
    so Stage 5 SVG reconstruction NEVER double-draws cartoon shapes or collides text over embedded assets.
    """
    with Image.open(draft_path) as base_img:
        rgba = base_img.convert("RGBA")
        dw, dh = rgba.size

    overlay = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_ui_font(16, bold=True)

    vb_w, vb_h = viewbox
    for idx, a in enumerate(assets, 1):
        x1 = int(round((a.svg_x / float(vb_w)) * dw))
        y1 = int(round((a.svg_y / float(vb_h)) * dh))
        x2 = int(round(((a.svg_x + a.svg_w) / float(vb_w)) * dw))
        y2 = int(round(((a.svg_y + a.svg_h) / float(vb_h)) * dh))
        draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 215), outline=(15, 23, 42, 255), width=2)
        box_w = max(1, x2 - x1)
        tag = f"ASSET:{a.asset_id}"
        bbox = draw.textbbox((0, 0), tag, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw > box_w - 8:
            tag = f"#{idx}"
            bbox = draw.textbbox((0, 0), tag, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.text((max(x1 + 3, cx - tw // 2), max(y1 + 2, cy - th // 2)), tag, fill=(255, 255, 255, 255), font=font)

    merged = Image.alpha_composite(rgba, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path, format="PNG")
    return out_path


def load_cached_assets(output_dir: Path) -> list[ExtractedAsset]:
    """Reload previously cropped assets from assets_manifest.json (for --from-stage 5/6 resume)."""
    manifest_path = output_dir / "assets_manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        items = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    loaded: list[ExtractedAsset] = []
    for item in items:
        cpath = Path(item.get("crop_path", ""))
        if not cpath.is_file():
            continue
        with Image.open(cpath) as im:
            data_url = _encode_pil_png_data_url(im.convert("RGBA"))
        loaded.append(ExtractedAsset(**item, data_url=data_url))
    return loaded


def hydrate_svg_assets(svg_code: str, assets: list[ExtractedAsset]) -> str:
    """Replace compact __ASSET_<id>__ placeholders with real base64 PNG data URLs,
    adding BOTH SVG2 `href` and SVG1.1 `xlink:href` + `xmlns:xlink` for universal compatibility
    across svg-edit, Adobe Illustrator, Figma, Inkscape, and browsers.
    """
    if not assets:
        return svg_code
    out = svg_code
    if 'xmlns:xlink="http://www.w3.org/1999/xlink"' not in out and "xmlns:xlink=" not in out:
        out = re.sub(
            r"<svg\b",
            '<svg xmlns:xlink="http://www.w3.org/1999/xlink"',
            out,
            count=1,
        )
    for asset in assets:
        # Ensure both href and xlink:href are populated without duplicating if already present
        out = re.sub(
            rf'\bhref="{re.escape(asset.placeholder)}"(?!\s+xlink:href=)',
            f'href="{asset.data_url}" xlink:href="{asset.data_url}"',
            out,
        )
        out = out.replace(asset.placeholder, asset.data_url)
    return out


def dehydrate_svg_assets(svg_code: str, assets: list[ExtractedAsset]) -> str:
    """Swap base64 PNG data URLs back to compact __ASSET_<id>__ placeholders before LLM inspection."""
    out = svg_code
    for asset in assets:
        if asset.data_url in out:
            # Collapse duplicate xlink:href back to a single clean href placeholder
            out = out.replace(
                f'href="{asset.data_url}" xlink:href="{asset.data_url}"',
                f'href="{asset.placeholder}"',
            )
            out = out.replace(asset.data_url, asset.placeholder)
    return out


def verify_assets_preserved(svg_code: str, assets: list[ExtractedAsset]) -> tuple[bool, list[str]]:
    """Verify every extracted asset placeholder (or hydrated data URL) is present in the SVG."""
    missing: list[str] = []
    for asset in assets:
        if asset.placeholder not in svg_code and asset.data_url not in svg_code:
            missing.append(asset.asset_id)
    return len(missing) == 0, missing


def inject_missing_assets(svg_code: str, assets: list[ExtractedAsset]) -> str:
    """Safety net: if the LLM forgot to include any asset placeholder, cleanly inject it before </svg>."""
    ok, missing = verify_assets_preserved(svg_code, assets)
    if ok:
        return svg_code
    missing_set = set(missing)
    snippets: list[str] = ['  <g id="layer-visual-assets-autofill">']
    for a in assets:
        if a.asset_id in missing_set:
            snippets.append(
                f'    <image id="asset-{a.asset_id}" x="{a.svg_x:.1f}" y="{a.svg_y:.1f}" '
                f'width="{a.svg_w:.1f}" height="{a.svg_h:.1f}" href="{a.placeholder}" '
                f'preserveAspectRatio="xMidYMid meet"/>'
            )
    snippets.append("  </g>")
    block = "\n".join(snippets) + "\n</svg>"
    return re.sub(r"</svg>\s*$", block, svg_code, count=1, flags=re.IGNORECASE)


def _parse_assets_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except Exception:
        return {"assets": []}
