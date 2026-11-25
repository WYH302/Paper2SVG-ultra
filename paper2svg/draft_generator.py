"""Stage 4: High-Craft Reference-Faithful Visual Anchor Generation + Smart Banner Trim."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .archetypes import ArchetypeSpec
from .architect import extract_exact_text_inventory, extract_prompt_body
from .config import resolve_density_mode


def smart_trim_white_margins(
    image: Image.Image,
    *,
    white_threshold: int = 247,
    pad_px: int = 28,
) -> Image.Image:
    """Trim excess pure-white outer borders cleanly while preserving crisp symmetric breathing margins."""
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    h, w, _ = arr.shape

    non_white = np.any(arr < white_threshold, axis=2)
    ys, xs = np.where(non_white)
    if len(xs) < 100 or len(ys) < 100:
        return rgb

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    crop_x0 = max(0, x0 - pad_px)
    crop_y0 = max(0, y0 - pad_px)
    crop_x1 = min(w, x1 + pad_px)
    crop_y1 = min(h, y1 + pad_px)

    cropped = rgb.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    cw, ch = cropped.size

    need_left = max(0, pad_px - x0)
    need_top = max(0, pad_px - y0)
    need_right = max(0, pad_px - (w - x1))
    need_bottom = max(0, pad_px - (h - y1))
    if any((need_left, need_top, need_right, need_bottom)):
        canvas = Image.new("RGB", (cw + need_left + need_right, ch + need_top + need_bottom), (255, 255, 255))
        canvas.paste(cropped, (need_left, need_top))
        cropped = canvas

    return cropped


def compute_svg_viewbox(
    draft_img: Image.Image,
    archetype: ArchetypeSpec,
    canonical_width: int = 1600,
) -> tuple[int, int]:
    """Use the EXACT 1:1 native pixel dimensions of stage1_draft.png (matching AutoFigure-Edit's
    zero-scaling coordinate system) so every cropped icon/graphic, panel border, arrow thickness,
    and font proportion maps 1:1 with ZERO fractional coordinate distortion.
    """
    dw, dh = draft_img.size
    if dw < 64 or dh < 64:
        return archetype.canvas_width, archetype.canvas_height
    return int(dw), int(dh)


def generate_stage1_draft(
    client: Any,
    schema_md: str,
    style_dna: dict[str, Any],
    archetype: ArchetypeSpec,
    output_dir: Path,
    *,
    density: str = "auto",
) -> tuple[Path, tuple[int, int], dict[str, Any]]:
    """Generate a publication-grade visual anchor faithfully honoring the reference folder's artistry."""
    prompt_body = extract_prompt_body(schema_md)
    inventory = extract_exact_text_inventory(schema_md)
    resolved_density = resolve_density_mode(density, len(inventory))
    anchor_paths = [Path(p) for p in style_dna.get("anchor_files", []) if Path(p).is_file()]

    aspect_directive = (
        "CANVAS PROPORTION: Create a sleek **WIDE PANORAMIC (~2.35:1–2.55:1) HORIZONTAL BANNER** "
        "designed for the top of a two-column CVPR / SIGGRAPH / NeurIPS / IEEE S&P / Nature paper, on a clean pure `#FFFFFF` background with balanced white margins."
        if archetype.is_wide_banner
        else "CANVAS PROPORTION: Create a balanced ~1.6:1 worked-example scientific teaser diagram on a pure `#FFFFFF` background with generous margins."
    )

    ref_clause = (
        "ART-DIRECTION REFERENCE IMAGE(S) ATTACHED — FOLLOW THEIR VISUAL CRAFT CLOSELY!\n"
        "- Study and match their EXACT visual sophistication: rendering richness (3D shaded objects, photorealistic specimens, flat 2D editorial panels, tactile textures, bespoke scientific icons, or crisp mini-plots), "
        "their border/borderless container philosophy, their arrow craft (e.g., wide colored arrows, chunky white-filled block arrows, soft ribbons, or clean connectors), color harmony, and whitespace rhythm.\n"
        "- DO NOT invent conflicting visual styles (e.g., NEVER add 3D plastic shading or drop shadows if the reference is crisp flat 2D editorial, and NEVER degrade a rich 3D reference into a dry wireframe flowchart)!\n"
        "- STRICT CONTENT ISOLATION ('借骨架与画风，不借画中内容'): NEVER copy ANY text, model names, titles, or unrelated modules from the reference images!"
        if anchor_paths
        else "Create a vibrant, visually captivating flagship NeurIPS / CVPR / Nature scientific illustration (NOT a dry plain Visio box-and-arrow flowchart!)."
    )

    if resolved_density == "detailed":
        density_mandate = """1. **CRISP MULTI-PANEL SYSTEMS ARCHITECTURE CLARITY (`detailed` tier)**:
   - Follow the exact multi-panel grid and white gutter spacing specified in the Visual Schema, keeping all panels cleanly aligned, well-proportioned, and visually uncluttered.
   - **STRICTLY FORBIDDEN**: DO NOT invent extra numbers, extra labels, or reference-image words not listed in `[EXACT TEXT INVENTORY]`!
2. **SHARP DOMAIN MICRO-VISUALS (MATCHING REFERENCE CRAFT)**:
   - Render every micro-visual in `[HERO VISUAL ASSETS]` crisp, domain-authentic, and well-separated from text titles and arrows."""
    else:
        density_mandate = """1. **AIRY EDITORIAL MINIMALISM (35%–45% PURE WHITE WHITESPACE!)**:
   - Keep the canvas remarkably **spacious, uncluttered, and breathable** — like a Nature / CVPR Oral flagship Figure 1!
   - Wide outer white margins, wide open gaps between stages, and generous spacing around every graphic and text label.
   - **STRICTLY FORBIDDEN**: DO NOT invent or add extra tiny numbers, extra hyperparameters, extra axis ticks, extra mini-charts, or extra labels that are NOT listed in `[EXACT TEXT INVENTORY]`!
2. **BIG, CRISP VISUAL HEROES (3–6 HERO GRAPHICS, NOT 16 POSTAGE STAMPS)**:
   - Make the 3–6 hero domain visuals in `[HERO VISUAL ASSETS]` LARGE, SHARP, and PROMINENT with clean open space around them. Let visuals breathe instead of trapping every sub-element inside nested boxes!"""

    render_prompt = f"""Design a **flagship, award-winning top-conference scientific overview illustration** (SIGGRAPH / CVPR / NeurIPS / IEEE S&P / Nature publication craft).

{aspect_directive}

{ref_clause}

KEY VISUAL EXCELLENCE & FIDELITY MANDATES:
{density_mandate}
3. **TRUE REFERENCE AESTHETIC**: Follow the container style (`{style_dna.get('container_philosophy', {}).get('style', '')}`), arrow craft (`{style_dna.get('arrow_craft', {}).get('style', '')}`), and color palette distilled from the reference.
4. **CRISP ACADEMIC TYPOGRAPHY**: Clean unified sans-serif (Inter / Helvetica / Arial), crisp and legible when shrunk to 7-inch two-column print width. Keep text cleanly separated from illustrations and arrows.
5. **ZERO META-LABEL LEAKAGE**: Render ONLY the concise labels in `[EXACT TEXT INVENTORY]`. NEVER write structural tags like "SECTION 1", "HERO VISUAL ASSETS", "BASELINE", "NOVEL", or "Container".

=== VISUAL SCHEMA TO RENDER ===
{prompt_body}
"""

    (output_dir / "stage4_render_prompt.txt").write_text(render_prompt, encoding="utf-8")

    raw_img, meta = client.generate_image(
        render_prompt,
        reference_images=anchor_paths,
        quality=client.settings.image_quality,
        size=client.settings.image_size,
    )

    raw_path = output_dir / "stage1_draft_raw.png"
    raw_img.save(raw_path, format="PNG")

    trimmed_img = smart_trim_white_margins(raw_img)
    draft_path = output_dir / "stage1_draft.png"
    trimmed_img.save(draft_path, format="PNG")

    viewbox = compute_svg_viewbox(trimmed_img, archetype)
    meta["raw_size"] = list(raw_img.size)
    meta["trimmed_size"] = list(trimmed_img.size)
    meta["svg_viewbox"] = list(viewbox)
    return draft_path, viewbox, meta
