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
    """Create a temporary inspection copy of stage1_draft.png with a subtle 0..1000 coordinate grid
    so the multimodal model can read bounding boxes with ~3-5px precision.
    """
    with Image.open(draft_path) as base_img:
        rgb = base_img.convert("RGBA")
        w, h = rgb.size

    margin = 36
    end_pad = 24
    canvas = Image.new("RGBA", (w + margin + end_pad, h + margin + end_pad), (255, 255, 255, 255))
    canvas.paste(rgb, (margin, margin))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_ui_font(13, bold=False)

    # Draw subtle dashed/translucent grid every 100 units in 0..1000 space
    for tick in range(0, 1001, 50):
        is_major = (tick % 100 == 0)
        alpha = 95 if is_major else 45
        color = (220, 38, 38, alpha) if is_major else (59, 130, 246, alpha)

        gx = margin + int(round(w * tick / 1000.0))
        gy = margin + int(round(h * tick / 1000.0))

        draw.line([(gx, margin), (gx, margin + h)], fill=color, width=1)
        draw.line([(margin, gy), (margin + w, gy)], fill=color, width=1)

        if is_major:
            draw.text((max(margin, gx - 12), 8), str(tick), fill=(185, 28, 28, 255), font=font)
            draw.text((4, max(margin, gy - 7)), str(tick), fill=(185, 28, 28, 255), font=font)

    combined = Image.alpha_composite(canvas, overlay).convert("RGB")
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

    cleaned = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGBA")

    # Trim empty transparent edges so SVG placement aligns tightly to visible artwork
    alpha = out[:, :, 3]
    ys, xs = np.where(alpha > 15)
    if len(xs) < 16 or len(ys) < 16:
        return rgba, (0, 0, 0, 0)

    tx0, tx1 = int(xs.min()), int(xs.max()) + 1
    ty0, ty1 = int(ys.min()), int(ys.max()) + 1
    trimmed = cleaned.crop((tx0, ty0, tx1, ty1))
    return trimmed, (tx0, ty0, w - tx1, h - ty1)


def _encode_pil_png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def detect_and_crop_visual_assets(
    client: Any,
    draft_path: Path,
    output_dir: Path,
    viewbox: tuple[int, int],
    *,
    max_assets: int = 8,
) -> list[ExtractedAsset]:
    """Detect rich visual assets in stage1_draft.png using the multimodal model + coordinate ruler,
    then crop them losslessly at full resolution for hybrid editable SVG assembly.
    """
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    ruler_path = output_dir / "stage1_draft_ruler.png"
    render_coordinate_ruler_overlay(draft_path, ruler_path)

    with Image.open(draft_path) as d_img:
        draft_rgb = d_img.convert("RGB")
        dw, dh = draft_rgb.size

    vb_w, vb_h = viewbox

    prompt = f"""You are a Precision Visual Asset Segmenter converting a scientific overview figure into a **Hybrid Editable SVG** (like AutoFigure-Edit).

Look at BOTH attached images:
- **Image 1**: Clean `stage1_draft.png` ({dw} x {dh} px).
- **Image 2**: The exact same figure overlaid with a **normalized 0..1000 coordinate ruler grid** (red major lines every 100 units, blue minor lines every 50 units).
  **IMPORTANT RULER READING RULE**: Read `[x1, y1, x2, y2]` strictly from the **RED `0..1000` numbers printed in the white ruler margin** (`0` = artwork top/left edge, `1000` = artwork bottom/right edge). Do NOT include the white ruler margin in your coordinate math!

YOUR JOB:
Locate EVERY rich visual artwork region inside the figure that should be cropped losslessly from Image 1 and embedded as a draggable `<image>` inside the editable SVG — **specifically**:
- Realistic photos or image samples (e.g., human faces, natural photos, medical scans, stone/marble slabs, satellite/scene crops)
- 3D/isometric rendered objects (e.g., 3D meshes, voxels, optical setups, molecules, shaded geometric volumes, layered 3D tensors)
- Tactile material textures, spatial fields, complex plots/heatmaps/spectra with dense detail
- Expressive bespoke illustrated characters/icons (e.g., robots, researchers, microscopes, trophies, rich scientific symbols)

CRITICAL EXCLUSION RULES (DO NOT CROP THESE!):
1. **NEVER include text labels, titles, captions, or equations inside a crop bbox!** Keep every bbox TIGHT around ONLY the visual graphic itself so all text remains 100% natively editable vector `<text>`!
2. **NEVER crop an entire module box or container frame!** Only crop the rich inner artwork/photo/3D object/illustration!
3. **NEVER crop plain simple geometric arrows, plain empty rectangles, or simple 1-color dots** — those are cleaner as native SVG vectors!
4. If the figure has NO complex photos/3D/rich illustrations worth cropping (purely geometric boxes/arrows), return `{{"assets": []}}`.

For each asset, choose `bg_mode`:
- `"opaque_rect"`: for rectangular framed photos, heatmaps, or full-bleed square image crops (e.g. a square portrait or rectangular marble texture photo).
- `"remove_bg"`: for irregular/floating illustrations, 3D objects, cut-out icons, or organic stone slabs sitting on a uniform white or pale tinted card background.

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
            "Image 2 — 0..1000 Normalized Coordinate Ruler Overlay:",
            ruler_path,
        ],
        max_output_tokens=8000,
        temperature=0.1,
    )

    parsed = _parse_assets_json(raw)
    raw_assets = parsed.get("assets") or []

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

        if px2 <= px1 or py2 <= py1:
            continue

        crop_file = assets_dir / f"{slug}.png"
        final_crop.save(crop_file, format="PNG", optimize=True)

        svg_x = (px1 / float(dw)) * vb_w
        svg_y = (py1 / float(dh)) * vb_h
        svg_w = ((px2 - px1) / float(dw)) * vb_w
        svg_h = ((py2 - py1) / float(dh)) * vb_h

        extracted.append(
            ExtractedAsset(
                asset_id=slug,
                description=str(item.get("description") or slug),
                bg_mode=bg_mode,
                bbox_1000=[x1_n, y1_n, x2_n, y2_n],
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
    for a in assets:
        x1 = int(round((a.svg_x / float(vb_w)) * dw))
        y1 = int(round((a.svg_y / float(vb_h)) * dh))
        x2 = int(round(((a.svg_x + a.svg_w) / float(vb_w)) * dw))
        y2 = int(round(((a.svg_y + a.svg_h) / float(vb_h)) * dh))
        draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 215), outline=(15, 23, 42, 255), width=2)
        tag = f"ASSET:{a.asset_id}"
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bbox = draw.textbbox((0, 0), tag, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((max(x1 + 2, cx - tw // 2), max(y1 + 2, cy - th // 2)), tag, fill=(255, 255, 255, 255), font=font)

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
