"""Stage 5: Hybrid & Pure-Vector Editable SVG Reconstructor + Strict XML Auto-Repair."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .archetypes import ArchetypeSpec
from .architect import extract_exact_text_inventory, extract_prompt_body
from .asset_extractor import (
    ExtractedAsset,
    detect_and_crop_visual_assets,
    hydrate_svg_assets,
    inject_missing_assets,
)
from .config import resolve_density_mode
from .svg_linter import TWO_COLUMN_WIDTH_PT, validate_svg_xml


def extract_svg_block(text: str) -> str:
    """Extract clean root `<svg ...>...</svg>` code from model response text (supports nested `<svg>`)."""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:svg|xml|html)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Greedy match to the final </svg> so nested <svg> viewports never truncate the root SVG
    m = re.search(r"(<svg\b[^>]*>.*</svg>)", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def normalize_svg_root(svg_code: str, width: int, height: int) -> str:
    """Ensure `<svg>` has standard xmlns and exact width/height/viewBox attributes."""
    if not svg_code:
        return svg_code
    if 'xmlns="http://www.w3.org/2000/svg"' not in svg_code and "xmlns='http://www.w3.org/2000/svg'" not in svg_code:
        svg_code = re.sub(r"<svg\b", '<svg xmlns="http://www.w3.org/2000/svg"', svg_code, count=1)
    if "viewBox=" not in svg_code:
        svg_code = re.sub(
            r"<svg\b",
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}"',
            svg_code,
            count=1,
        )
    return svg_code


def fix_svg_xml_with_llm(
    client: Any,
    svg_code: str,
    errors: list[str],
    *,
    width: int,
    height: int,
    max_retries: int = 3,
) -> str:
    """Iteratively repair broken XML syntax using the main LLM until lxml validates cleanly."""
    current = svg_code
    current_errors = errors
    for _ in range(max_retries):
        prompt = f"""The following SVG code has XML syntax errors that break SVG parsers/editors.
Fix ALL XML syntax errors (such as unescaped `&` -> `&amp;`, unescaped `<` inside text -> `&lt;`, unclosed tags, duplicate attributes, or truncated ending) WITHOUT changing the visual layout, text labels, or `__ASSET_...__` image placeholders.

XML ERRORS:
{chr(10).join('- ' + e for e in current_errors)}

BROKEN SVG:
{current}

Output ONLY the repaired, valid SVG code starting with `<svg` and ending with `</svg>`. NO markdown fences, NO commentary."""
        repaired_raw = client.generate_text(prompt, max_output_tokens=64000, temperature=0.1)
        candidate = normalize_svg_root(extract_svg_block(repaired_raw), width, height)
        if not candidate:
            continue
        ok, new_errs = validate_svg_xml(candidate)
        if ok:
            return candidate
        current = candidate
        current_errors = new_errs
    return current


def build_editable_svg(
    client: Any,
    draft_path: Path,
    schema_md: str,
    style_dna: dict[str, Any],
    archetype: ArchetypeSpec,
    viewbox: tuple[int, int],
    output_dir: Path,
    *,
    font_family: str,
    svg_mode: str = "hybrid_rich",
    density: str = "auto",
    cached_assets: Optional[list[ExtractedAsset]] = None,
) -> tuple[str, list[ExtractedAsset], bool, list[str]]:
    """Reconstruct a publication-grade editable SVG faithfully preserving BOTH visual artistry & editability.

    - In `hybrid_rich` mode (default, AutoFigure-Edit style without SAM3/GPU):
      Automatically detects and crops photorealistic samples, 3D renders, material textures,
      rich plots, and bespoke scientific icons from `stage1_draft.png` and embeds them as
      individually draggable `<image>` assets inside a 100% natively editable vector framework
      (cards/zones, colored header bands, wide arrows, equations, and crisp `<text>` labels).
    - In `pure_vector` mode:
      Draws everything purely using SVG vector primitives.
    - Adaptive density (`concise` vs `detailed`): calibrates typography floors & box padding cleanly
      so both airy CVPR/Nature single-strip figures and dense IEEE S&P / USENIX multi-panel system
      architectures render with ZERO text overflow and ZERO canvas clipping.
    """
    w, h = viewbox
    prompt_body = extract_prompt_body(schema_md)
    inventory = extract_exact_text_inventory(schema_md)
    resolved_density = resolve_density_mode(density, len(inventory))

    pt_scale = TWO_COLUMN_WIDTH_PT / float(w)
    if resolved_density == "detailed":
        # Calibrated to exact 1:1 native pixel canvas (e.g. 2016x756 -> pt_scale=0.25)
        px_title = int(round(8.2 / pt_scale))
        px_panel = int(round(7.4 / pt_scale))
        px_module = int(round(5.8 / pt_scale))
        px_sub = int(round(5.0 / pt_scale))
        px_floor = int(round(4.3 / pt_scale))
        box_pad_px = max(10, int(round(w * 0.006)))
    else:
        px_title = int(round(10.5 / pt_scale))
        px_panel = int(round(9.5 / pt_scale))
        px_module = int(round(8.2 / pt_scale))
        px_sub = int(round(6.8 / pt_scale))
        px_floor = int(round(6.4 / pt_scale))
        box_pad_px = max(14, int(round(w * 0.009)))

    assets: list[ExtractedAsset] = []
    if svg_mode == "hybrid_rich":
        if cached_assets is not None:
            assets = cached_assets
        else:
            assets = detect_and_crop_visual_assets(client, draft_path, output_dir, viewbox, max_assets=28)

    pal = style_dna.get("palette", {})
    cont = style_dna.get("container_philosophy", {})
    arr = style_dna.get("arrow_craft", {})

    if assets:
        asset_manifest_json = json.dumps([a.to_prompt_dict() for a in assets], ensure_ascii=False, indent=2)
        asset_instructions = f"""=== EXACT EMBEDDED VISUAL ASSETS ({len(assets)} HD GRAPHICS — ZERO HAND-DRAWN ICONS!) ===
Image 2 (`stage1_draft_masked.png`) covers EVERY non-text visual graphic (`AF01`, `AF02`, ..., `AF{len(assets):02d}`) with a grey `#808080` rectangle.
We ALREADY cropped all {len(assets)} graphics losslessly at native `{w}x{h}` resolution with transparent backgrounds!
- **STRICT RULE**: **NEVER hand-draw ANY icon, emblem, waveform, neural network, DAG, hardware device, spectrogram, matrix, or mini-screen with `<path>`!**
- Instead, place EVERY SINGLE asset verbatim inside `<g id="layer-visual-assets">` at its EXACT `(x, y, width, height)` from the manifest below:
```xml
<g id="asset-AF01">
  <image x="..." y="..." width="..." height="..." href="__ASSET_AF01__" preserveAspectRatio="xMidYMid meet"/>
</g>
```
- Only add a `<clipPath>` for `"bg_mode": "opaque_rect"` items when `rounded_rx > 0`.
- Keep the exact `(x, y, width, height)` from the manifest (these are ALREADY 1:1 native pixels aligned to Image 1!).
- EVERY SINGLE `href_placeholder` (`__ASSET_AF01__` .. `__ASSET_AF{len(assets):02d}__`) MUST appear verbatim in your SVG!

ASSET PLACEMENT MANIFEST (1:1 NATIVE PIXEL COORDINATES):
```json
{asset_manifest_json}
```"""
    else:
        asset_instructions = (
            "No cropped raster assets were extracted — render all visual motifs cleanly matching `stage1_draft.png`."
        )

    prompt = f"""You are a Master Scientific SVG Engineer performing **AutoFigure-Edit Style 1:1 Pixel-Exact SVG Reconstruction** of a publication-grade academic figure.

YOUR SINGLE GOAL — **1:1 PIXEL-LEVEL VISUAL FIDELITY TO IMAGE 1 (`stage1_draft.png`)**:
Replicate Image 1 (`stage1_draft.png`, exact dimensions `{w} x {h}` px) so faithfully that when rendered, our editable SVG looks **VIRTUALLY IDENTICAL IN BEAUTY, PROPORTION, BOLDNESS, COLOR, AND CRAFTSMANSHIP** to Image 1 — with EVERY icon/illustration preserved via the crisp embedded `<image href="__ASSET_AF..__"/>` assets, and EVERY panel, white inset card, dashed frame, text pill, arrow, and text label rebuilt as razor-sharp native editable SVG!

1. **EXACT 1:1 NATIVE PIXEL COORDINATE SYSTEM (DO NOT RESCALE!)**:
   - Root tag MUST be:
     `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">`
   - First child: `<rect x="0" y="0" width="{w}" height="{h}" fill="{pal.get('canvas_bg', '#FFFFFF')}"/>`
   - Keep EVERY element strictly inside `[10, {w - 10}] x [10, {h - 10}]`.

2. **EXACT CONTAINER, COLOR & BORDER REPLICATION**:
   - Match Image 1's EXACT panel bounding boxes, row alignment (identical top/bottom Y across panels in the same row!), white gutters, corner radii, flat solid fill hexes (NO unsolicited muddy gradients or drop shadows if Image 1 is crisp flat 2D!), crisp border stroke widths (e.g. `3px`–`3.5px` on `{w}x{h}`), pure `#FFFFFF` inset cards, dashed focus frames (`stroke-dasharray`), and soft tinted text pills!

3. **EXACT ARROW CRAFT, BOLDNESS & DIRECTION (DO NOT MISS, SHRINK, OR INVERT ANY ARROW!)**:
   - Inspect Image 1 carefully and replicate **EVERY SINGLE ARROW** (both chunky inter-panel block arrows AND every short black intra-panel connector arrow!).
   - Match Image 1's EXACT arrow thickness, broad symmetric triangular heads, fill color (e.g., chunky `#FFFFFF` block arrows with crisp `#171717` outlines; chunky `#9DBBE5` blue combine arrows), and clean centering inside gutters.
   - **PER-ARROW DIRECTION CHECK (MANDATORY!)**: NEVER copy-paste an upward vertical arrow polygon for a downward connector (or vice versa)! Check EVERY inter-panel arrow individually against Image 1 AND `[FLOW CONNECTIONS]`: verify its arrowhead tip (`y_tip < y_base` for **UP**, `y_tip > y_base` for **DOWN**, `x_tip < x_base` for **LEFT**, `x_tip > x_base` for **RIGHT**) matches BOTH Image 1 and `[FLOW CONNECTIONS]` 100%!

4. **BOLD, HIGH-CONTRAST ACADEMIC TYPOGRAPHY MATCHING IMAGE 1**:
   - Unified clean sans-serif font: `font-family="{font_family}"` (NO Serif, NO Comic Sans).
   - **MATCH IMAGE 1's VISUAL WEIGHT & INK DARKNESS**: Use crisp bold `#111111` (NOT timid washed-out grey!) for panel titles, card sub-headers, pill labels, and figure annotations wherever Image 1 uses dark bold black text!
   - Reference scale on this `{w}x{h}` canvas: panel titles `~{px_panel}px` bold, card headers `~{px_module}px` bold, pill/module labels `~{px_sub}px` bold/semibold, minimum floor `>= {px_floor}px`.
   - **ZERO `textLength` / `lengthAdjust` SQUASHING!** Wrap multi-line headers cleanly onto 2 `<tspan>` lines matching Image 1's exact line breaks, centered with `text-anchor="middle"` and `dominant-baseline="central"`, with `>= {box_pad_px}px` horizontal padding inside every pill/box and ZERO collisions with icons, borders, or arrows!
   - Follow `[EXACT TEXT INVENTORY]` strictly for exact spelling (fixing any minor typo in Image 1, zero meta-labels).

{asset_instructions}

CLEAN 5-LAYER SEMANTIC GROUPING (FOR EASY EDITING IN ILLUSTRATOR / FIGMA / SVG-EDIT):
1. `<defs>`: reusable arrowheads, `<clipPath>` for rectangular framed assets, and clean `<style>` classes.
2. `<g id="layer-backdrop">`: outer colored/pastel macro panels & dividers matching Image 1 exact positions.
3. `<g id="layer-modules">`: `#FFFFFF` inset cards, dashed focal frames, and colored text pills (`<g id="mod-...">`).
4. `<g id="layer-visual-assets">`: ALL {len(assets)} embedded `<image href="__ASSET_AF..__"/>` assets at exact 1:1 coordinates (NO duplicate hand-drawn shapes behind/over them!).
5. `<g id="layer-connectors">`: ALL chunky inter-panel block arrows and intra-panel connector arrows matching Image 1 exact boldness & direction.
6. `<g id="layer-labels-and-legend">`: ALL crisp bold editable `<text>` labels and legend keys positioned with exact parity to Image 1.

XML VALIDITY: Escape `&` as `&amp;`, `<` as `&lt;`, `>` as `&gt;`. NEVER render meta-labels ("SECTION 1", "AF01", "<AF>01", "BASELINE", "NOVEL").

=== AUTHORITATIVE EXACT TEXT INVENTORY ===
{json.dumps(inventory, ensure_ascii=False, indent=2)}

=== DISTILLED COLOR PALETTE ===
{json.dumps(pal, ensure_ascii=False, indent=2)}

=== FLOW & STRUCTURE REFERENCE ===
{prompt_body}

OUTPUT ONLY THE COMPLETE, VALID SVG CODE starting with `<svg` and ending with `</svg>`. NO markdown fences, NO explanation.
"""

    masked_path = output_dir / "stage1_draft_masked.png"
    multimodal_inputs: list[Any] = [
        prompt,
        f"Image 1 — Target Figure to Replicate 1:1 (`stage1_draft.png`, {w}x{h} px):",
        draft_path,
    ]
    if assets and masked_path.is_file():
        multimodal_inputs.extend(
            [
                f"Image 2 — Cropped Visual Asset Mask Overlay (`stage1_draft_masked.png`, {w}x{h} px): "
                "Every grey box (`AF01`, `AF02`, ...) marks an exact graphic ALREADY cropped for you — "
                "DO NOT redraw anything inside these grey boxes; place the corresponding `<image href=\"__ASSET_AF..__\"/>` "
                "at its exact manifest coordinates and keep all text/arrows cleanly outside their boundaries!",
                masked_path,
            ]
        )

    raw = client.generate_multimodal(
        multimodal_inputs,
        max_output_tokens=64000,
        temperature=0.15,
    )
    svg_template = normalize_svg_root(extract_svg_block(raw), w, h)
    if not svg_template:
        raise RuntimeError(f"Model failed to return an <svg> block. Preview: {raw[:500]}")

    ok, errs = validate_svg_xml(svg_template)
    if not ok:
        svg_template = fix_svg_xml_with_llm(client, svg_template, errs, width=w, height=h)
        ok, errs = validate_svg_xml(svg_template)

    if assets:
        svg_template = inject_missing_assets(svg_template, assets)
        svg_hydrated = hydrate_svg_assets(svg_template, assets)
    else:
        svg_hydrated = svg_template

    return svg_hydrated, assets, ok, errs
