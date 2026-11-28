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
    uniform_ratio = float(np.mean(border_dists < tolerance * 1.15))
    if uniform_ratio < 0.45:
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


def _find_1d_bands(counts: np.ndarray) -> list[tuple[int, int, int]]:
    """Find contiguous non-zero index spans as (start, end_incl, total_sum)."""
    bands: list[tuple[int, int, int]] = []
    in_band = False
    b_start = 0
    n = len(counts)
    for i in range(n):
        if counts[i] > 0 and not in_band:
            in_band = True
            b_start = i
        elif counts[i] == 0 and in_band:
            in_band = False
            bands.append((b_start, i - 1, int(np.sum(counts[b_start:i]))))
    if in_band:
        bands.append((b_start, n - 1, int(np.sum(counts[b_start:n]))))
    return bands


def strip_edge_text_bleed(crop_img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Deterministically strip stray clipped text captions or card-border slivers touching the outer
    edges of a cropped icon (e.g. a single bold caption line like `VLM` / `SSIM /` / `Stop &` /
    `Refined` directly below an icon, or a stray letter / 3px card-border sliver at left/right),
    separated by a clean blank white/transparent gap from the main graphic.

    Preserves full-bleed photos/spectrograms and equal-height multi-row graphics (e.g. 3 stacked
    waveforms or 3 skill bars) 100% untouched.
    """
    rgba = crop_img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32)
    h, w, _ = arr.shape
    if h < 28 or w < 24:
        return rgba, (0, 0, 0, 0)

    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    opaque_rgb = rgb[alpha > 15]
    if len(opaque_rgb) < 20:
        return rgba, (0, 0, 0, 0)
    bg_est = np.percentile(opaque_rgb, 92, axis=0)
    is_ink = (alpha > 15) & (np.linalg.norm(rgb - bg_est, axis=2) > 40.0)

    new_top, new_bottom = 0, h
    new_left, new_right = 0, w

    # --- Step 1: Peel an open C-shaped grazed card border wrapping top+bottom+left (open on right!) ---
    # Must run BEFORE band-stripping so any stray letter sliver on the right edge (e.g. grazed 'T') is stripped next!
    if h >= 40 and w >= 40:
        top_edge_dense = float(np.mean(is_ink[:4, :])) > 0.45
        top_gap_clean = float(np.mean(is_ink[5:9, 6 : w - 4])) < 0.03
        bot_edge_dense = float(np.mean(is_ink[h - 4 :, :])) > 0.45
        bot_gap_clean = float(np.mean(is_ink[h - 9 : h - 5, 6 : w - 4])) < 0.03
        left_edge_dense = float(np.mean(is_ink[:, :4])) > 0.45
        left_gap_clean = float(np.mean(is_ink[6 : h - 6, 5:9])) < 0.03
        # Crucial distinction: a C-shaped grazed card frame is OPEN on the right edge, whereas a framed
        # mini-screen/thumbnail is CLOSED on all 4 sides! Never peel a closed 4-sided picture frame.
        right_edge_open = float(np.mean(is_ink[6 : h - 6, w - 4 :])) < 0.20
        if (
            top_edge_dense
            and top_gap_clean
            and bot_edge_dense
            and bot_gap_clean
            and left_edge_dense
            and left_gap_clean
            and right_edge_open
        ):
            inner = is_ink[7 : h - 7, 7:w]
            iys, ixs = np.where(inner)
            if len(ixs) >= 24 and len(iys) >= 24:
                ix0, ix1 = int(ixs.min()), int(ixs.max()) + 1
                iy0, iy1 = int(iys.min()), int(iys.max()) + 1
                new_left = 7 + ix0
                new_right = 7 + ix1
                new_top = 7 + iy0
                new_bottom = 7 + iy1

    # --- Step 2: Strip disconnected top/bottom caption lines & left/right stray letter slivers ---
    for _ in range(2):
        sub_ink = is_ink[new_top:new_bottom, new_left:new_right]
        sh, sw = sub_ink.shape
        if sh < 28 or sw < 24:
            break
        row_ink = np.sum(sub_ink, axis=1)
        rbands = _find_1d_bands(row_ink)
        changed = False
        if len(rbands) >= 2:
            main_b = max(rbands, key=lambda b: b[2])
            main_bh = main_b[1] - main_b[0] + 1
            max_sliver_h = max(24, int(round(sh * 0.28)))

            first = rbands[0]
            if first != main_b and first[0] <= 2:
                sliver_h = first[1] - first[0] + 1
                gap_after = rbands[1][0] - first[1] - 1
                if (
                    sliver_h <= max_sliver_h
                    and sliver_h <= main_bh * 0.38
                    and gap_after >= 1
                    and first[2] < main_b[2] * 0.42
                ):
                    new_top += rbands[1][0]
                    changed = True

            last = rbands[-1]
            if not changed and last != main_b and last[1] >= sh - 3:
                sliver_h = last[1] - last[0] + 1
                gap_before = last[0] - rbands[-2][1] - 1
                if (
                    sliver_h <= max_sliver_h
                    and sliver_h <= main_bh * 0.38
                    and gap_before >= 1
                    and last[2] < main_b[2] * 0.42
                ):
                    new_bottom -= (sh - (rbands[-2][1] + 1))
                    changed = True

        # Check vertical left/right stray letter or border slivers (e.g. grazed 'T' on right of doc icon)
        sub_ink = is_ink[new_top:new_bottom, new_left:new_right]
        sh, sw = sub_ink.shape
        col_ink = np.sum(sub_ink, axis=0)
        cbands = _find_1d_bands(col_ink)
        if len(cbands) >= 2:
            main_c = max(cbands, key=lambda b: b[2])
            main_cw = main_c[1] - main_c[0] + 1
            max_sliver_w = max(12, int(round(sw * 0.16)))

            cfirst = cbands[0]
            if cfirst != main_c and cfirst[0] <= 2:
                sliver_w = cfirst[1] - cfirst[0] + 1
                gap_after = cbands[1][0] - cfirst[1] - 1
                if (
                    sliver_w <= max_sliver_w
                    and sliver_w <= main_cw * 0.25
                    and gap_after >= 2
                    and cfirst[2] < main_c[2] * 0.25
                ):
                    new_left += cbands[1][0]
                    changed = True

            clast = cbands[-1]
            if clast != main_c and clast[1] >= sw - 3:
                sliver_w = clast[1] - clast[0] + 1
                gap_before = clast[0] - cbands[-2][1] - 1
                if (
                    sliver_w <= max_sliver_w
                    and sliver_w <= main_cw * 0.25
                    and gap_before >= 2
                    and clast[2] < main_c[2] * 0.25
                ):
                    new_right -= (sw - (cbands[-2][1] + 1))
                    changed = True

        if not changed:
            break

    if (new_top, new_bottom, new_left, new_right) == (0, h, 0, w):
        return rgba, (0, 0, 0, 0)
    if (new_bottom - new_top) < 16 or (new_right - new_left) < 16:
        return rgba, (0, 0, 0, 0)

    trimmed = rgba.crop((new_left, new_top, new_right, new_bottom))
    return trimmed, (new_left, new_top, w - new_right, h - new_bottom)


def snap_bbox_to_foreground(
    draft_arr: np.ndarray,
    px1: int,
    py1: int,
    px2: int,
    py2: int,
    *,
    max_expand: int = 14,
    ink_threshold: float = 36.0,
) -> tuple[int, int, int, int]:
    """Zero-GPU pixel-level boundary snapper: if a VLM coarse crop edge slices across a graphic's
    foreground pixels (including light studio portrait frames), step outward (up to `max_expand` px)
    until hitting a clean blank background column/row, stopping immediately before any panel wall.
    """
    h, w, _ = draft_arr.shape
    rx0 = max(0, px1 - max_expand)
    ry0 = max(0, py1 - max_expand)
    rx1 = min(w, px2 + max_expand)
    ry1 = min(h, py2 + max_expand)
    roi = draft_arr[ry0:ry1, rx0:rx1, :3]
    if roi.size == 0:
        return px1, py1, px2, py2

    # Estimate local card/canvas backdrop from the brightest 15% pixels in local ROI
    lum = 0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 2]
    thr_lum = float(np.percentile(lum, 85))
    bright_px = roi[lum >= thr_lum]
    if len(bright_px) < 16:
        return px1, py1, px2, py2
    bg_color = np.median(bright_px, axis=0)

    def _is_ink_line(slice_rgb: np.ndarray) -> float:
        if slice_rgb.size == 0:
            return 0.0
        dists = np.linalg.norm(slice_rgb - bg_color, axis=-1)
        return float(np.mean(dists > ink_threshold))

    def _is_panel_wall_v(col_x: int, y0: int, y1: int) -> bool:
        """True if vertical column `col_x` continues as a solid line far above AND below [y0, y1]."""
        above = draft_arr[max(0, y0 - 12) : y0, col_x, :3]
        below = draft_arr[y1 : min(h, y1 + 12), col_x, :3]
        return above.shape[0] >= 4 and below.shape[0] >= 4 and _is_ink_line(above) > 0.75 and _is_ink_line(below) > 0.75

    def _is_panel_wall_h(row_y: int, x0: int, x1: int) -> bool:
        """True if horizontal row `row_y` continues as a solid line far left AND right of [x0, x1]."""
        left = draft_arr[row_y, max(0, x0 - 12) : x0, :3]
        right = draft_arr[row_y, x1 : min(w, x1 + 12), :3]
        return left.shape[0] >= 4 and right.shape[0] >= 4 and _is_ink_line(left) > 0.75 and _is_ink_line(right) > 0.75

    nx1, ny1, nx2, ny2 = px1, py1, px2, py2

    # Expand LEFT if current left edge slices through graphic ink (stopping at blank bg or outer panel wall)
    if _is_ink_line(draft_arr[ny1:ny2, nx1, :3]) > 0.06:
        for _ in range(max_expand):
            if nx1 <= 1 or _is_panel_wall_v(nx1 - 1, ny1, ny2):
                break
            if _is_ink_line(draft_arr[ny1:ny2, nx1 - 1, :3]) <= 0.02:
                break
            nx1 -= 1

    # Expand RIGHT
    if _is_ink_line(draft_arr[ny1:ny2, max(0, nx2 - 1), :3]) > 0.06:
        for _ in range(max_expand):
            if nx2 >= w - 1 or _is_panel_wall_v(nx2, ny1, ny2):
                break
            if _is_ink_line(draft_arr[ny1:ny2, nx2, :3]) <= 0.02:
                break
            nx2 += 1

    # Expand TOP
    if _is_ink_line(draft_arr[ny1, nx1:nx2, :3]) > 0.06:
        for _ in range(max_expand):
            if ny1 <= 1 or _is_panel_wall_h(ny1 - 1, nx1, nx2):
                break
            if _is_ink_line(draft_arr[ny1 - 1, nx1:nx2, :3]) <= 0.02:
                break
            ny1 -= 1

    # Expand BOTTOM
    if _is_ink_line(draft_arr[max(0, ny2 - 1), nx1:nx2, :3]) > 0.06:
        for _ in range(max_expand):
            if ny2 >= h - 1 or _is_panel_wall_h(ny2, nx1, nx2):
                break
            if _is_ink_line(draft_arr[ny2, nx1:nx2, :3]) <= 0.02:
                break
            ny2 += 1

    return nx1, ny1, nx2, ny2


def _trim_uniform_margin_opaque(crop_img: Image.Image, tolerance: float = 20.0) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Trim redundant uniform white/card outer padding around a rectangular opaque crop."""
    rgba = crop_img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32)
    h, w, _ = arr.shape
    if h < 16 or w < 16:
        return rgba, (0, 0, 0, 0)
    rgb = arr[:, :, :3]
    border_pixels = np.concatenate([rgb[0, :, :], rgb[h - 1, :, :], rgb[:, 0, :], rgb[:, w - 1, :]], axis=0)
    bg_color = np.median(border_pixels, axis=0)
    if float(np.mean(np.linalg.norm(border_pixels - bg_color, axis=1) < tolerance)) < 0.70:
        return rgba, (0, 0, 0, 0)
    fg = np.linalg.norm(rgb - bg_color, axis=2) > tolerance * 1.25
    ys, xs = np.where(fg)
    if len(xs) < 16 or len(ys) < 16:
        return rgba, (0, 0, 0, 0)
    tx0, tx1 = int(xs.min()), int(xs.max()) + 1
    ty0, ty1 = int(ys.min()), int(ys.max()) + 1
    if tx0 <= 1 and ty0 <= 1 and (w - tx1) <= 1 and (h - ty1) <= 1:
        return rgba, (0, 0, 0, 0)
    return rgba.crop((tx0, ty0, tx1, ty1)), (tx0, ty0, w - tx1, h - ty1)


def _encode_pil_png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _deduplicate_overlapping_specs(raw_assets: list[dict[str, Any]], overlap_thresh: float = 0.72) -> list[dict[str, Any]]:
    """Drop near-duplicate heavily overlapping asset specs (keeping the earlier/larger clean one)."""
    kept: list[dict[str, Any]] = []
    kept_boxes: list[tuple[float, float, float, float]] = []
    for item in raw_assets:
        bbox = item.get("bbox_1000")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            continue
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            continue
        area = (x2 - x1) * (y2 - y1)
        duplicate = False
        for kx1, ky1, kx2, ky2 in kept_boxes:
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                karea = (kx2 - kx1) * (ky2 - ky1)
                if inter / max(1.0, min(area, karea)) >= overlap_thresh:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(item)
            kept_boxes.append((x1, y1, x2, y2))
    return kept


def _crop_assets_from_specs(
    raw_assets: list[dict[str, Any]],
    draft_rgb: Image.Image,
    assets_dir: Path,
    viewbox: tuple[int, int],
    *,
    max_assets: int = 28,
) -> list[ExtractedAsset]:
    """Crop and package visual assets from normalized 0..1000 bbox specifications with
    zero-GPU pixel-boundary snapping, edge flood-fill background removal, and text-bleed stripping.
    """
    dw, dh = draft_rgb.size
    vb_w, vb_h = viewbox
    draft_arr = np.asarray(draft_rgb, dtype=np.float32)

    # Clean stale crops before writing the latest pass
    for old_png in assets_dir.glob("*.png"):
        try:
            old_png.unlink()
        except Exception:
            pass

    deduped = _deduplicate_overlapping_specs(raw_assets)[:max_assets]
    extracted: list[ExtractedAsset] = []

    for idx, item in enumerate(deduped, 1):
        bbox = item.get("bbox_1000")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1_n, y1_n, x2_n, y2_n = [int(round(float(v))) for v in bbox]
        except Exception:
            continue

        x1_n = max(0, min(999, x1_n))
        y1_n = max(0, min(999, y1_n))
        x2_n = max(x1_n + 8, min(1000, x2_n))
        y2_n = max(y1_n + 8, min(1000, y2_n))

        # Reject accidental whole-figure crops (> 55% of width & height)
        if (x2_n - x1_n) > 550 and (y2_n - y1_n) > 550:
            continue

        px1 = int(round(dw * x1_n / 1000.0))
        py1 = int(round(dh * y1_n / 1000.0))
        px2 = int(round(dw * x2_n / 1000.0))
        py2 = int(round(dh * y2_n / 1000.0))
        if (px2 - px1) < 16 or (py2 - py1) < 16:
            continue

        bg_mode = "remove_bg" if str(item.get("bg_mode", "")).strip().lower() == "remove_bg" else "opaque_rect"
        rounded_rx = int(item.get("rounded_rx") or (4 if bg_mode == "opaque_rect" else 0))

        # Zero-GPU pixel-boundary snap: recover sliced portrait/photo edges cleanly (up to 72px for photos!)
        if bg_mode == "opaque_rect":
            px1, py1, px2, py2 = snap_bbox_to_foreground(
                draft_arr, px1, py1, px2, py2, max_expand=72, ink_threshold=14.0
            )
        else:
            px1, py1, px2, py2 = snap_bbox_to_foreground(
                draft_arr, px1, py1, px2, py2, max_expand=10, ink_threshold=34.0
            )

        # Canonical AutoFigure-Edit style ID (AF01, AF02, ...) for unmistakable 1-to-1 mask matching
        slug = f"AF{len(extracted) + 1:02d}"

        crop = draft_rgb.crop((px1, py1, px2, py2))
        if bg_mode == "remove_bg":
            final_crop, (tl, tt, tr, tb) = smart_remove_border_background(crop, tolerance=20.0)
            px1 += tl
            py1 += tt
            px2 -= tr
            py2 -= tb
        else:
            final_crop, (tl, tt, tr, tb) = _trim_uniform_margin_opaque(crop)
            px1 += tl
            py1 += tt
            px2 -= tr
            py2 -= tb

        # Deterministically strip any tiny grazed caption/header text slivers separated by blank gap
        final_crop, (btl, btt, btr, btb) = strip_edge_text_bleed(final_crop)
        px1 += btl
        py1 += btt
        px2 -= btr
        py2 -= btb

        if (px2 - px1) < 12 or (py2 - py1) < 12:
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
    max_assets: int = 28,
) -> tuple[bool, list[dict[str, Any]]]:
    """Perform a 1-pass visual alignment & completeness audit on proposed crop boxes.

    1. Catches any MISSED icons/graphics (so Stage 5 NEVER has to hand-draw ugly SVG doodles!)
    2. Catches any clipped adjacent text headers/captions or misaligned borders BEFORE Stage 5 runs.
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

    prompt = f"""You are performing a **Full-Coverage Icon & Crop Box Alignment Audit** (AutoFigure-Edit standard) before Hybrid SVG assembly.

Look carefully at ALL THREE attached images (ALL THREE share the EXACT same pixel dimensions, `0..1000` edge-to-edge, ZERO outer margins!):
- **Image 1**: Clean `stage1_draft.png`
- **Image 2**: `stage1_draft_ruler.png` (exact edge-to-edge `0..1000` red/blue grid)
- **Image 3**: `stage1_draft_verify.png` (the current proposed crop boxes drawn as translucent RED rectangles with compact `#1`, `#2`, ... corner badges!)

CURRENT PROPOSED BOXES ({len(current_specs)} items):
```json
{json.dumps(current_specs, ensure_ascii=False, indent=2)}
```

AUDIT IMAGE 3 STRICTLY AGAINST IMAGE 1 & IMAGE 2:
1. **MISSED VISUAL GRAPHICS CHECK**: Scan left-to-right, top-to-bottom. Add any missed non-text visual graphic (`bbox_1000`) so the SVG builder never hand-draws ugly icons.
2. **ZERO TEXT INSIDE BOXES (SPLIT STACKED BARS!)**: Does ANY red rectangle cover text labels (`c_sem`, `c_id`, `c_geo`, titles, captions, equations)? If a single tall box covers 3 stacked color bars AND swallows `c_id`/`c_geo` labels between them, **SPLIT IT into 3 separate tight horizontal bar boxes** so ZERO text sits inside any red box!
3. **ZERO SLICED PORTRAITS OR CLIPPED OUTER ARROWS (CRITICAL!)**: Inspect EVERY portrait/face photo (`opaque_rect`) and EVERY radial vector field / scatter plot / 3D block: does ANY red box cut off part of a person's hair/ear/cheek/studio frame, or leave outer arrows/petals sticking OUTSIDE the red border? If YES, expand `bbox_1000` cleanly to include 100% of the photo/graphic!
4. **DROP PLAIN BOXES / ARROWS**: Remove any red box covering a plain text pill, outer background panel, or simple arrow.
5. Only set `"needs_adjustment": false` when EVERY box has zero clipped faces/arrows and zero swallowed text!

Return ONLY valid JSON (no markdown fences, up to {max_assets} assets):
{{
  "needs_adjustment": true,
  "audit_notes": ["brief note on added/adjusted boxes"],
  "assets": [
    {{
      "asset_id": "AF01",
      "description": "brief description",
      "bbox_1000": [x1, y1, x2, y2],
      "bg_mode": "opaque_rect | remove_bg",
      "rounded_rx": 4
    }}
  ]
}}
"""

    # Cap auxiliary mask-alignment audit at 'high' when global effort is 'xhigh' to keep turnaround crisp
    cur_eff = getattr(getattr(client, "settings", None), "reasoning_effort", "")
    audit_effort = "high" if str(cur_eff).strip().lower() == "xhigh" else None
    mm_kwargs: dict[str, Any] = {"max_output_tokens": 8000, "temperature": 0.1}
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
    max_assets: int = 28,
    verify_masks: bool = True,
) -> list[ExtractedAsset]:
    """Detect EVERY non-text visual graphic/icon in stage1_draft.png (AutoFigure-Edit full-coverage
    paradigm without GPU/SAM3), snap boundaries pixel-accurately with pure NumPy, verify coverage &
    alignment via a translucent overlay, and crop losslessly at full native resolution.
    """
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    ruler_path = output_dir / "stage1_draft_ruler.png"
    render_coordinate_ruler_overlay(draft_path, ruler_path)

    with Image.open(draft_path) as d_img:
        draft_rgb = d_img.convert("RGB")
        dw, dh = draft_rgb.size

    prompt = f"""You are a Precision Visual Asset Segmenter converting a high-craft academic figure into an **AutoFigure-Edit Style Hybrid Editable SVG**.

Look at BOTH attached images:
- **Image 1**: Clean `stage1_draft.png` ({dw} x {dh} px).
- **Image 2**: `stage1_draft_ruler.png` — the **EXACT SAME {dw} x {dh} px image** (ZERO outer white margin!) overlaid directly with an **edge-to-edge normalized 0..1000 coordinate grid** (red major lines every 100 units with red tick badges on all 4 edges; blue minor lines every 50 units).
  **DIRECT 1:1 COORDINATE RULE**: `0` is the exact left/top edge of the image and `1000` is the exact right/bottom edge of the image. Read `[x1, y1, x2, y2]` directly from the red/blue grid lines — DO NOT add or subtract any margin offset!

CORE DESIGN PRINCIPLE (WHY THIS MATTERS FOR BEAUTY!):
An SVG code generator excels at crisp rectangular panels, white inset cards, dashed borders, text pills, geometric arrows, and editable `<text>`, **but produces UGLY amateur doodles whenever forced to hand-draw icons, logos, neural networks, waveforms, devices, DAGs, or mini-screens!**
Therefore, your job is to **LOCATE EVERY SINGLE NON-TEXT VISUAL GRAPHIC / ICON** in Image 1 (typically **10 to 25 items** on a rich diagram!) so ALL of them are cropped losslessly at native HD resolution and embedded cleanly as draggable `<image>` assets!

MUST CROP ALL OF THESE VISUAL GRAPHICS (SCAN EVERY SINGLE PANEL SYSTEMATICALLY!):
- **Hardware / Device / Scene Illustrations** (monitors, laptops, cables, antennas/probes, sensors, robots, microscopes, 3D objects, faces, natural photos)
- **Icons & Geometric Emblems** (VLM/LLM swirl or rosette logos, Critic icons, document/page icons, lock/gear/flame/code symbols)
- **Waveforms, Signals, Spectrograms & Plots** (multi-frequency modulated RF waveforms, grayscale spectrograms, heatmaps, stacked waveform/signal cards, mini-charts)
- **Neural Networks & Structured Micro-Diagrams** (black or colored node-link neural networks, compact workflow DAGs on grids, shaded table/matrix icons, multi-row colored skill-bar graphics, driver/tool badge strips)
- **Framed Preview Thumbnails** (mini screen/output preview cards like `Initial Output`, `Refined Output`, `Verified Output` inner visual graphic)

STRICT EXCLUSION RULES (DO NOT INCLUDE THESE INSIDE ANY CROP BBOX!):
1. **ZERO EDITABLE TEXT INSIDE CROPS (CRITICAL — SPLIT MULTI-ROW GRAPHICS!)**: NEVER include section titles, card headers, pill text, math symbols (`c_sem`, `c_id`, `c_geo`, `y`, `x_0 -> x_1`), legend labels, or captions inside any `bbox_1000`! If multiple horizontal color bars / thumbnails have text labels BETWEEN rows (e.g., `c_sem` above bar 1, `c_id` above bar 2, `c_geo` above bar 3), **EMIT ONE SEPARATE TIGHT BOX PER BAR** — NEVER one tall box that swallows the text labels in between!
2. **NEVER SLICE FACES, PORTRAITS, OR OUTER ARROW RINGS**: For human portraits/photos (`opaque_rect`), ensure the box covers 100% of the rectangular photo edge-to-edge (all hair, ears, shoulders, studio backdrop — NEVER cut off the right/left side of a face!). For radial vector fields or plots, include 100% of the outer arrow ring and axes!
3. **NEVER crop outer pastel/tinted panels, white card backgrounds, dashed border frames, or colored text pills** — those are rebuilt cleanly as native vector `<rect>`!
4. **NEVER crop inter-panel block arrows or simple straight connector arrows** — those are rebuilt cleanly as native vector arrows!
5. **NO DUPLICATE OVERLAPPING BOXES**: Each graphic gets ONE clean non-overlapping tight bbox.

For each asset, choose `bg_mode`:
- `"opaque_rect"`: for crisp rectangular framed spectrograms, photos, heatmaps, framed mini-screens, or shaded tables that fill a rectangle cleanly.
- `"remove_bg"`: for floating icons, emblems (`VLM`, `Critic`), neural networks, waveforms, device illustrations, DAGs, document icons, or stacked cards sitting on a white or light tinted card background (transparent background removal will preserve their crisp lines cleanly!).

Return ONLY valid JSON (no markdown fences) with up to {max_assets} assets, ordered cleanly top-to-bottom, left-to-right:
{{
  "assets": [
    {{
      "asset_id": "AF01",
      "description": "brief description of the graphic",
      "bbox_1000": [x1, y1, x2, y2],
      "bg_mode": "opaque_rect | remove_bg",
      "rounded_rx": 4
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
        max_output_tokens=10000,
        temperature=0.1,
    )

    parsed = _parse_assets_json(raw)
    raw_assets = parsed.get("assets") or []

    extracted = _crop_assets_from_specs(
        raw_assets, draft_rgb, assets_dir, viewbox, max_assets=max_assets
    )

    if extracted and verify_masks:
        # Up to 2 visual verification passes: Pass 1 fixes coarse shifts; Pass 2 verifies the calibrated
        # overlay and catches any last clipped portrait edge, outer vector-field ring, or swallowed text!
        for cal_round in range(1, 3):
            verify_name = "stage1_draft_verify.png" if cal_round == 1 else "stage1_draft_verify_calibrated.png"
            verify_path = output_dir / verify_name
            render_verification_overlay(draft_path, extracted, viewbox, verify_path)
            adjusted, refined_specs = _verify_and_calibrate_asset_boxes(
                client, draft_path, ruler_path, verify_path, extracted, max_assets=max_assets
            )
            if not adjusted:
                break
            new_extracted = _crop_assets_from_specs(
                refined_specs, draft_rgb, assets_dir, viewbox, max_assets=max_assets
            )
            if new_extracted:
                extracted = new_extracted
                if cal_round == 2:
                    render_verification_overlay(
                        draft_path, extracted, viewbox, output_dir / "stage1_draft_verify_final.png"
                    )
            else:
                break

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
    """Create an authentic AutoFigure-Edit `samed.png` mask overlay: solid `#808080` grey rectangle
    with crisp `#000000` outline and centered white `AF01`, `AF02`, ... label fitting inside each box.
    """
    with Image.open(draft_path) as base_img:
        rgba = base_img.convert("RGBA")
        dw, dh = rgba.size

    overlay = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    vb_w, vb_h = viewbox
    for idx, a in enumerate(assets, 1):
        x1 = int(round((a.svg_x / float(vb_w)) * dw))
        y1 = int(round((a.svg_y / float(vb_h)) * dh))
        x2 = int(round(((a.svg_x + a.svg_w) / float(vb_w)) * dw))
        y2 = int(round(((a.svg_y + a.svg_h) / float(vb_h)) * dh))
        draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 240), outline=(0, 0, 0, 255), width=2)
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        tag = a.asset_id if len(a.asset_id) <= 6 else f"AF{idx:02d}"
        fsize = max(10, min(20, int(min(box_w * 0.28, box_h * 0.36))))
        font = _load_ui_font(fsize, bold=True)
        bbox = draw.textbbox((0, 0), tag, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw > box_w - 4:
            font = _load_ui_font(max(9, int(fsize * (box_w - 4) / max(1, tw))), bold=True)
            bbox = draw.textbbox((0, 0), tag, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.text((cx - tw // 2 - bbox[0], cy - th // 2 - bbox[1]), tag, fill=(255, 255, 255, 255), font=font)

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
