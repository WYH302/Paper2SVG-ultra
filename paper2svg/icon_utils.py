"""Pure-NumPy Icon Segmentation, Photo Protection & Border-FloodFill Matting Helpers (Zero-GPU).

Cleanly consolidated for Paper2SVG-ultra v3.0:
  - `detect_cc_frame_stripped`: strips ONLY long panel borders (>=150px, 1px dilation) before connected-component
    extraction + horizontal/vertical projection splitting so panels & text stay pure SVG while icons are extracted cleanly.
  - `strip_text_edges_on_box`: classifies rectangular opaque photos vs transparent icons and strips pure-black caption
    lines grazed along top/bottom box edges.
  - `looks_like_opaque_photo_tile`: protects real portraits/photos/heatmaps/DUT chips from neural or floodfill chewing.
  - `strip_edge_text_bleed` & `smart_remove_border_background`: zero-GPU edge cleanup + perimeter flood-fill transparency.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def looks_like_opaque_photo_tile(crop_img: Image.Image) -> bool:
    """Detect whether a cropped tile is a rectangular photograph/heatmap (keep opaque + optional rx=4)
    vs an icon/diagram on a uniform white or soft pastel card backdrop.
    """
    rgb = np.asarray(crop_img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    if h < 16 or w < 16:
        return False
    border = np.concatenate([rgb[0, :, :], rgb[h - 1, :, :], rgb[:, 0, :], rgb[:, w - 1, :]], axis=0)
    med = np.median(border, axis=0)
    lum = float(0.299 * med[0] + 0.587 * med[1] + 0.114 * med[2])
    dists = np.linalg.norm(border - med, axis=1)
    uniform_ratio = float(np.mean(dists < 22.0))
    if uniform_ratio >= 0.55 and lum >= 205.0:
        return False
    if uniform_ratio < 0.45:
        return True
    corners = np.array([rgb[2, 2], rgb[2, w - 3], rgb[h - 3, 2], rgb[h - 3, w - 3]])
    corner_spread = float(np.max(np.linalg.norm(corners - med, axis=1)))
    return corner_spread > 38.0


def _find_1d_bands(counts: np.ndarray) -> list[tuple[int, int, int]]:
    bands: list[tuple[int, int, int]] = []
    in_band = False
    b_start = 0
    for i in range(len(counts)):
        if counts[i] > 0 and not in_band:
            in_band = True
            b_start = i
        elif counts[i] == 0 and in_band:
            in_band = False
            bands.append((b_start, i - 1, int(np.sum(counts[b_start:i]))))
    if in_band:
        bands.append((b_start, len(counts) - 1, int(np.sum(counts[b_start:]))))
    return bands


def strip_edge_text_bleed(crop_img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Strip stray clipped text captions or open C-shaped card-border slivers touching outer crop edges."""
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

    # Peel open C-shaped grazed card border wrapping top+bottom+left (never peel closed 4-sided picture frame)
    if h >= 40 and w >= 40:
        if (
            float(np.mean(is_ink[:4, :])) > 0.45
            and float(np.mean(is_ink[5:9, 6 : w - 4])) < 0.03
            and float(np.mean(is_ink[h - 4 :, :])) > 0.45
            and float(np.mean(is_ink[h - 9 : h - 5, 6 : w - 4])) < 0.03
            and float(np.mean(is_ink[:, :4])) > 0.45
            and float(np.mean(is_ink[6 : h - 6, 5:9])) < 0.03
            and float(np.mean(is_ink[6 : h - 6, w - 4 :])) < 0.20
        ):
            inner = is_ink[7 : h - 7, 7:w]
            iys, ixs = np.where(inner)
            if len(ixs) >= 24 and len(iys) >= 24:
                new_left, new_right = 7 + int(ixs.min()), 7 + int(ixs.max()) + 1
                new_top, new_bottom = 7 + int(iys.min()), 7 + int(iys.max()) + 1

    # Strip disconnected top/bottom caption slivers & left/right stray letter slivers
    for _ in range(2):
        sub_ink = is_ink[new_top:new_bottom, new_left:new_right]
        sh, sw = sub_ink.shape
        if sh < 28 or sw < 24:
            break
        changed = False
        rbands = _find_1d_bands(np.sum(sub_ink, axis=1))
        if len(rbands) >= 2:
            main_b = max(rbands, key=lambda b: b[2])
            main_bh = main_b[1] - main_b[0] + 1
            max_sliver_h = max(24, int(round(sh * 0.28)))
            first = rbands[0]
            if first != main_b and first[0] <= 2:
                sliver_h = first[1] - first[0] + 1
                gap = rbands[1][0] - first[1] - 1
                if sliver_h <= max_sliver_h and sliver_h <= main_bh * 0.38 and gap >= 1 and first[2] < main_b[2] * 0.42:
                    new_top += rbands[1][0]
                    changed = True
            last = rbands[-1]
            if last != main_b and last[1] >= sh - 3:
                sliver_h = last[1] - last[0] + 1
                gap = last[0] - rbands[-2][1] - 1
                if sliver_h <= max_sliver_h and sliver_h <= main_bh * 0.38 and gap >= 1 and last[2] < main_b[2] * 0.42:
                    new_bottom -= sh - (rbands[-2][1] + 1)
                    changed = True

        sub_ink = is_ink[new_top:new_bottom, new_left:new_right]
        sh, sw = sub_ink.shape
        cbands = _find_1d_bands(np.sum(sub_ink, axis=0))
        if len(cbands) >= 2:
            main_c = max(cbands, key=lambda b: b[2])
            main_cw = main_c[1] - main_c[0] + 1
            max_sliver_w = max(14, int(round(sw * 0.16)))
            first_c = cbands[0]
            if first_c != main_c and first_c[0] <= 2:
                sliver_w = first_c[1] - first_c[0] + 1
                gap = cbands[1][0] - first_c[1] - 1
                if sliver_w <= max_sliver_w and sliver_w <= main_cw * 0.25 and gap >= 2 and first_c[2] < main_c[2] * 0.22:
                    new_left += cbands[1][0]
                    changed = True
            last_c = cbands[-1]
            if last_c != main_c and last_c[1] >= sw - 3:
                sliver_w = last_c[1] - last_c[0] + 1
                gap = last_c[0] - cbands[-2][1] - 1
                if sliver_w <= max_sliver_w and sliver_w <= main_cw * 0.25 and gap >= 2 and last_c[2] < main_c[2] * 0.22:
                    new_right -= sw - (cbands[-2][1] + 1)
                    changed = True
        if not changed:
            break

    if (new_left, new_top, new_right, new_bottom) == (0, 0, w, h):
        return rgba, (0, 0, 0, 0)
    if (new_right - new_left) < 16 or (new_bottom - new_top) < 16:
        return rgba, (0, 0, 0, 0)
    return rgba.crop((new_left, new_top, new_right, new_bottom)), (new_left, new_top, w - new_right, h - new_bottom)


def smart_remove_border_background(
    crop_img: Image.Image,
    tolerance: float = 18.0,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Remove uniform card/canvas backdrop around an icon via perimeter-seeded 4-neighbor flood fill.

    Interior white regions (e.g., white robot faces, white cubes, white clouds) remain 100% opaque.
    """
    rgba = crop_img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32)
    h, w, _ = arr.shape
    if h < 8 or w < 8:
        return rgba, (0, 0, 0, 0)

    rgb = arr[:, :, :3]
    border_pixels = np.concatenate([rgb[0, :, :], rgb[h - 1, :, :], rgb[:, 0, :], rgb[:, w - 1, :]], axis=0)
    bg_color = np.median(border_pixels, axis=0)
    border_dists = np.linalg.norm(border_pixels - bg_color, axis=1)
    if float(np.mean(border_dists < tolerance * 1.15)) < 0.45:
        return rgba, (0, 0, 0, 0)

    dist_map = np.linalg.norm(rgb - bg_color, axis=2)
    bg_mask = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

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

    # Soft 1px anti-alias feathering along silhouette edges
    padded = np.pad(bg_mask, 1, mode="constant", constant_values=False)
    touches_bg = (padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]) & (~bg_mask)
    soft_zone = touches_bg & (dist_map < tolerance * 1.8)
    if np.any(soft_zone):
        alpha_scale = np.clip((dist_map[soft_zone] - tolerance * 0.85) / (tolerance * 0.95), 0.15, 1.0)
        out[soft_zone, 3] = np.round(out[soft_zone, 3] * alpha_scale)

    alpha = out[:, :, 3]
    ys, xs = np.where(alpha > 15)
    if len(xs) < 16 or len(ys) < 16:
        return rgba, (0, 0, 0, 0)

    tx0, tx1 = int(xs.min()), int(xs.max()) + 1
    ty0, ty1 = int(ys.min()), int(ys.max()) + 1
    cleaned = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGBA")
    return cleaned.crop((tx0, ty0, tx1, ty1)), (tx0, ty0, w - tx1, h - ty1)


def _long_runs_h(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros_like(mask)
    for r in range(mask.shape[0]):
        row = mask[r]
        if int(np.sum(row)) < min_len:
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if e - s >= min_len:
                out[r, s:e] = True
    return out


def _split_cc_box_by_projection(
    fg_full: np.ndarray,
    arr_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    min_dim: int = 26,
) -> list[tuple[int, int, int, int]]:
    """Split adjacent icons at thin arrow/blank gaps and strip pure-black single-line headers/captions."""
    x0, y0, x1, y1 = box
    bh, bw = y1 - y0, x1 - x0
    if bw < min_dim or bh < min_dim:
        return []
    sub = fg_full[y0:y1, x0:x1]
    if sub.size == 0:
        return []

    col_counts = np.sum(sub, axis=0)
    sub_rgb = arr_rgb[y0:y1, x0:x1]
    sub_sat = np.max(sub_rgb, axis=2) - np.min(sub_rgb, axis=2)
    col_chromatic = np.sum(sub & (sub_sat > 24.0), axis=0)
    is_vgap = (col_counts == 0) | ((col_counts <= 24) & (col_chromatic <= 2))

    vsegs: list[tuple[int, int]] = []
    in_s, s0 = False, 0
    for c in range(bw):
        if not is_vgap[c] and not in_s:
            in_s, s0 = True, c
        elif is_vgap[c] and in_s:
            in_s = False
            if c - s0 >= min_dim:
                vsegs.append((s0, c))
    if in_s and bw - s0 >= min_dim:
        vsegs.append((s0, bw))

    merged_v: list[tuple[int, int]] = []
    for a, b in vsegs:
        if merged_v and (a - merged_v[-1][1]) < 10:
            merged_v[-1] = (merged_v[-1][0], b)
        else:
            merged_v.append((a, b))

    if len(merged_v) > 1 or (len(merged_v) == 1 and (merged_v[0][0] > 2 or merged_v[0][1] < bw - 2)):
        out_v: list[tuple[int, int, int, int]] = []
        for a, b in merged_v:
            ys, xs = np.where(fg_full[y0:y1, x0 + a : x0 + b])
            if len(xs) < 30:
                continue
            nb = (x0 + a + int(xs.min()), y0 + int(ys.min()), x0 + a + int(xs.max()) + 1, y0 + int(ys.max()) + 1)
            if len(merged_v) > 1:
                out_v.extend(_split_cc_box_by_projection(fg_full, arr_rgb, nb, min_dim=min_dim))
            else:
                out_v.append(nb)
        if out_v:
            if len(merged_v) > 1:
                return out_v
            x0, y0, x1, y1 = out_v[0]
            bh, bw = y1 - y0, x1 - x0
            sub = fg_full[y0:y1, x0:x1]

    row_counts = np.sum(sub, axis=1)
    is_hgap = row_counts <= max(1, int(bw * 0.015))
    hsegs: list[tuple[int, int]] = []
    in_s, s0 = False, 0
    for r in range(bh):
        if not is_hgap[r] and not in_s:
            in_s, s0 = True, r
        elif is_hgap[r] and in_s:
            in_s = False
            hsegs.append((s0, r))
    if in_s:
        hsegs.append((s0, bh))

    merged_h: list[tuple[int, int]] = []
    for a, b in hsegs:
        if merged_h and (a - merged_h[-1][1]) < 5:
            merged_h[-1] = (merged_h[-1][0], b)
        else:
            merged_h.append((a, b))

    if len(merged_h) >= 2:
        kept_h: list[tuple[int, int]] = []
        for a, b in merged_h:
            seg_h = b - a
            sub_fg_h = fg_full[y0 + a : y0 + b, x0:x1]
            fg_px = arr_rgb[y0 + a : y0 + b, x0:x1][sub_fg_h]
            cr = float(np.mean((np.max(fg_px, axis=1) - np.min(fg_px, axis=1)) > 26.0)) if len(fg_px) else 0.0
            if seg_h <= 46 and cr < 0.08 and bh >= 75:
                continue
            if seg_h >= min_dim:
                kept_h.append((a, b))
        if len(kept_h) == 1 and (kept_h[0][0] > 2 or kept_h[0][1] < bh - 2):
            a, b = kept_h[0]
            ys, xs = np.where(fg_full[y0 + a : y0 + b, x0:x1])
            if len(xs) >= 30:
                nb = (x0 + int(xs.min()), y0 + a + int(ys.min()), x0 + int(xs.max()) + 1, y0 + a + int(ys.max()) + 1)
                return _split_cc_box_by_projection(fg_full, arr_rgb, nb, min_dim=min_dim)
        elif len(kept_h) >= 2 and all((b - a) >= 44 for a, b in kept_h):
            out_h: list[tuple[int, int, int, int]] = []
            for a, b in kept_h:
                ys, xs = np.where(fg_full[y0 + a : y0 + b, x0:x1])
                if len(xs) < 30:
                    continue
                nb = (x0 + int(xs.min()), y0 + a + int(ys.min()), x0 + int(xs.max()) + 1, y0 + a + int(ys.max()) + 1)
                out_h.extend(_split_cc_box_by_projection(fg_full, arr_rgb, nb, min_dim=min_dim))
            if out_h:
                return out_h

    ys, xs = np.where(sub)
    if len(xs) < 30:
        return []
    return [(x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)]


def detect_cc_frame_stripped(img: Image.Image) -> list[dict[str, Any]]:
    """Frame-stripped zero-GPU CC icon detector: erases ONLY long panel borders (>=150px, 1px dilation)
    before connected-component extraction + projection splitting so icons inside bordered panels are never merged.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    sat = np.max(arr, axis=2) - np.min(arr, axis=2)

    gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
    gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1]))
    grad = np.max(gx, axis=2) + np.max(gy, axis=2)
    flat_bright = (grad < 8.0) & (lum > 180.0)
    q = (arr // 12.0).astype(np.int32)
    codes = q[:, :, 0] * 1024 + q[:, :, 1] * 32 + q[:, :, 2]
    vals, cnts = np.unique(codes[flat_bright], return_counts=True)
    bg_cols = [np.median(arr[flat_bright & (codes == c)], axis=0) for c in vals[cnts > w * h * 0.008]]
    bg_cols.append(np.array([255.0, 255.0, 255.0], dtype=np.float32))

    bg = np.zeros((h, w), dtype=bool)
    for col in bg_cols:
        bg |= np.linalg.norm(arr - col, axis=2) < 18.0
    bg |= (lum > 243.0) & (sat < 14.0)
    fg = ~bg

    v_thick = np.zeros_like(fg, dtype=np.int16)
    h_thick = np.zeros_like(fg, dtype=np.int16)
    for d in range(-5, 6):
        v_thick += np.roll(fg, d, axis=0)
        h_thick += np.roll(fg, d, axis=1)
    thin_h = _long_runs_h(fg & (v_thick <= 7), min_len=150)
    thin_v = _long_runs_h((fg & (h_thick <= 7)).T, min_len=150).T
    border_mask = (
        np.asarray(Image.fromarray(((thin_h | thin_v) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))) > 128
    )
    fg_strip = fg & (~border_mask)

    sh, sw = h // 2, w // 2
    fg_s = fg_strip[::2, ::2]
    sat_s = sat[::2, ::2]
    lum_s = lum[::2, ::2]
    dark_neutral = fg_s & (sat_s < 24.0) & (lum_s < 160.0)
    rich_fg = fg_s & (~dark_neutral)
    rich_dil = np.asarray(Image.fromarray((rich_fg * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))) > 128
    dn_u = dark_neutral.astype(np.uint8)
    vs = sum(np.roll(dn_u, d, axis=0) for d in range(-4, 5))
    hs = sum(np.roll(dn_u, d, axis=1) for d in range(-4, 5))
    dn_thick = dark_neutral & (vs >= 5) & (hs >= 4)
    comb = rich_dil | (np.asarray(Image.fromarray((dn_thick * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))) > 128)

    vis = np.zeros((sh, sw), dtype=bool)
    out: list[dict[str, Any]] = []
    for y in range(sh):
        for x in range(sw):
            if comb[y, x] and not vis[y, x]:
                dq = deque([(y, x)])
                vis[y, x] = True
                x0 = x1 = x
                y0 = y1 = y
                while dq:
                    cy, cx = dq.popleft()
                    if cx < x0:
                        x0 = cx
                    if cx > x1:
                        x1 = cx
                    if cy < y0:
                        y0 = cy
                    if cy > y1:
                        y1 = cy
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < sh and 0 <= nx < sw and comb[ny, nx] and not vis[ny, nx]:
                            vis[ny, nx] = True
                            dq.append((ny, nx))
                ox0, oy0, ox1, oy1 = max(0, x0 * 2), max(0, y0 * 2), min(w, (x1 + 1) * 2), min(h, (y1 + 1) * 2)
                if (ox1 - ox0) < 30 or (oy1 - oy0) < 30 or (ox1 - ox0) > w * 0.35 or (oy1 - oy0) > h * 0.55:
                    continue
                for bx0, by0, bx1, by1 in _split_cc_box_by_projection(fg_strip, arr, (ox0, oy0, ox1, oy1), min_dim=26):
                    bw, bh = bx1 - bx0, by1 - by0
                    if bw < 28 or bh < 28 or bw > w * 0.35 or bh > h * 0.55:
                        continue
                    sub_fg = fg_strip[by0:by1, bx0:bx1]
                    fg_px = arr[by0:by1, bx0:bx1][sub_fg]
                    if len(fg_px) < 100:
                        continue
                    cr = float(np.mean((np.max(fg_px, axis=1) - np.min(fg_px, axis=1)) > 24.0))
                    if cr < 0.08 and bh < 46:
                        continue
                    out.append(
                        {
                            "x1": max(0, bx0 - 2),
                            "y1": max(0, by0 - 2),
                            "x2": min(w, bx1 + 2),
                            "y2": min(h, by1 + 2),
                            "score": 0.85,
                            "prompt": "cc_icon",
                        }
                    )
    return out


def strip_text_edges_on_box(arr: np.ndarray, box: dict[str, Any]) -> dict[str, Any]:
    """Classify opaque photo tile vs transparent icon and strip pure-black caption lines along box edges."""
    h, w, _ = arr.shape
    x1, y1, x2, y2 = max(0, int(box["x1"])), max(0, int(box["y1"])), min(w, int(box["x2"])), min(h, int(box["y2"]))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return {**box, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "bg_mode": "remove_bg", "rounded_rx": 0}
    patch = arr[y1:y2, x1:x2]
    border = np.concatenate([patch[0, :, :], patch[-1, :, :], patch[:, 0, :], patch[:, -1, :]], axis=0)
    bg_col = np.median(border, axis=0)
    b_lum = float(0.299 * bg_col[0] + 0.587 * bg_col[1] + 0.114 * bg_col[2])
    b_uni = float(np.mean(np.linalg.norm(border - bg_col, axis=1) < 22.0))
    if b_uni < 0.48 or b_lum < 195.0:
        return {**box, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "bg_mode": "opaque_rect", "rounded_rx": 4}

    fg_local = np.linalg.norm(patch - bg_col, axis=2) > 22.0
    bh, bw = fg_local.shape
    row_cnt = np.sum(fg_local, axis=1)
    active = row_cnt > max(1, int(bw * 0.015))
    d = np.diff(np.concatenate(([0], active.astype(np.int8), [0])))
    segs = [(int(s), int(e)) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])]
    m_segs: list[tuple[int, int]] = []
    for a, b in segs:
        if m_segs and a - m_segs[-1][1] < 4:
            m_segs[-1] = (m_segs[-1][0], b)
        else:
            m_segs.append((a, b))
    if len(m_segs) >= 2:
        while len(m_segs) >= 2:
            a, b = m_segs[0]
            px = patch[a:b][fg_local[a:b]]
            cr = float(np.mean((np.max(px, axis=1) - np.min(px, axis=1)) > 24.0)) if len(px) else 0.0
            if (b - a) <= 34 and cr < 0.08:
                m_segs.pop(0)
            else:
                break
        while len(m_segs) >= 2:
            a, b = m_segs[-1]
            px = patch[a:b][fg_local[a:b]]
            cr = float(np.mean((np.max(px, axis=1) - np.min(px, axis=1)) > 24.0)) if len(px) else 0.0
            if (b - a) <= 34 and cr < 0.08:
                m_segs.pop()
            else:
                break
        if m_segs:
            sub = fg_local[m_segs[0][0] : m_segs[-1][1]]
            ys, xs = np.where(sub)
            if len(xs) > 40:
                ny1 = y1 + m_segs[0][0] + int(ys.min())
                ny2 = y1 + m_segs[0][0] + int(ys.max()) + 1
                nx1 = x1 + int(xs.min())
                nx2 = x1 + int(xs.max()) + 1
                x1, y1, x2, y2 = max(0, nx1 - 1), max(0, ny1 - 1), min(w, nx2 + 1), min(h, ny2 + 1)
    return {**box, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "bg_mode": "remove_bg", "rounded_rx": 0}
