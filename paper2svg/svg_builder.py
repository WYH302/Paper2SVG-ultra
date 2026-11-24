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
    """
    w, h = viewbox
    pt_scale = TWO_COLUMN_WIDTH_PT / float(w)
    px_title = int(round(10.5 / pt_scale))
    px_panel = int(round(9.5 / pt_scale))
    px_module = int(round(8.2 / pt_scale))
    px_sub = int(round(6.8 / pt_scale))
    px_floor = int(round(6.4 / pt_scale))

    assets: list[ExtractedAsset] = []
    if svg_mode == "hybrid_rich":
        if cached_assets is not None:
            assets = cached_assets
        else:
            assets = detect_and_crop_visual_assets(client, draft_path, output_dir, viewbox)

    prompt_body = extract_prompt_body(schema_md)
    inventory = extract_exact_text_inventory(schema_md)
    pal = style_dna.get("palette", {})
    cont = style_dna.get("container_philosophy", {})
    arr = style_dna.get("arrow_craft", {})

    if assets:
        asset_manifest_json = json.dumps([a.to_prompt_dict() for a in assets], ensure_ascii=False, indent=2)
        asset_instructions = f"""=== HYBRID EDITABLE VISUAL ASSETS (MUST PLACE ALL OF THESE!) ===
We have already cropped {len(assets)} crisp high-resolution visual assets directly from `stage1_draft.png`
(photos, 3D visuals, material textures, rich illustrations/plots).
**DO NOT crudely redraw these objects with simple cartoon polygons!**
Instead, place EVERY SINGLE asset cleanly inside `<g id="layer-visual-assets">` using its exact `href_placeholder`:
```xml
<g id="asset-<asset_id>">
  <image x="..." y="..." width="..." height="..." href="__ASSET_<asset_id>__" preserveAspectRatio="xMidYMid meet"/>
</g>
```
- For assets with `rounded_rx > 0` (`opaque_rect` photos/crops), define a crisp `<clipPath id="clip-<asset_id>"><rect x="..." y="..." width="..." height="..." rx="..."/></clipPath>` in `<defs>` and apply `clip-path="url(#clip-<asset_id>)"` to the `<image>`!
- You may fine-tune `(x, y, width, height)` slightly by a few pixels so each asset sits harmoniously centered inside its card/section with ZERO text overlap.
- EVERY asset in the JSON list below MUST appear verbatim via its `href_placeholder` string (`__ASSET_...__`)!

ASSET PLACEMENT MANIFEST:
```json
{asset_manifest_json}
```"""
    else:
        asset_instructions = (
            "No cropped raster assets were required for this diagram — render all visual motifs cleanly using refined SVG primitives "
            "and cohesive gradients/filters matching `stage1_draft.png`."
        )

    prompt = f"""You are a Master Scientific SVG Engineer & Editorial Graphic Designer (SIGGRAPH / CVPR / NeurIPS / Nature).

YOUR TASK:
Inspect the attached **Stage-1 visual composition (`stage1_draft.png`)** and reconstruct it faithfully as a **stunning, publication-grade EDITABLE SVG** (`viewBox="0 0 {w} {h}"`).

CRITICAL DUAL FIDELITY MANDATE:
1. **PRESERVE THE ARTISTRY OF `stage1_draft.png` — DO NOT DOWNGRADE INTO A DRY FLOWCHART!**
   - Faithfully match the visual aesthetic of `stage1_draft.png`: its exact colors, borderless pills / soft tinted zones / colored header bars / subtle shadows (`container_philosophy`: `{cont.get('style')}`), its arrow style (`arrow_craft`: `{arr.get('style')}` — wide colored block arrows, smooth curves, or clean connectors), and its open breathing room!
   - NEVER add ugly heavy dark wireframe borders around elements that are open or borderless in `stage1_draft.png`!
2. **100% AUTHORITATIVE TEXT ACCURACY**:
   - Every single word, symbol, and formula MUST be pure editable `<text>` (never baked into images) and MUST follow `[EXACT TEXT INVENTORY]` strictly — fixing any tiny typo or garbled glyph present in `stage1_draft.png`!

{asset_instructions}

EXACT CANVAS & VIEWBOX:
- Root tag MUST be:
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">`
- First child: `<rect x="0" y="0" width="{w}" height="{h}" fill="{pal.get('canvas_bg', '#FFFFFF')}"/>`

TWO-COLUMN PRINT TYPOGRAPHY CALIBRATION (MANDATORY FOR 7-INCH PRINT LEGIBILITY):
- Single unified sans-serif font across ALL text: `font-family="{font_family}"` (NO Serif mixing, NO Comic Sans MS).
- Define clean `<style>` classes calibrated to this {w}x{h} canvas:
  * `.title-text`   -> `font-size: {px_title}px; font-weight: 700; fill: {pal.get('text_primary', '#111827')};`
  * `.panel-title`  -> `font-size: {px_panel}px; font-weight: 700; fill: {pal.get('text_primary', '#111827')};`
  * `.module-title` -> `font-size: {px_module}px; font-weight: 600; fill: {pal.get('text_primary', '#111827')};`
  * `.sub-label`    -> `font-size: {px_sub}px; font-weight: 500; fill: {pal.get('text_secondary', '#4B5563')};`
  * `.arrow-label`  -> `font-size: {px_floor}px; font-weight: 600; fill: {pal.get('text_secondary', '#4B5563')};`
- **HARD FLOOR**: NEVER use `font-size` smaller than `{px_floor}px` on any readable label!
- Center labels cleanly with `text-anchor="middle"` and `dominant-baseline="central"`, giving >= 14px horizontal padding inside cards; wrap multi-word labels onto 2 `<tspan>` lines rather than letting text overflow borders.

CLEAN 5-LAYER SEMANTIC GROUPING (FOR EASY EDITING IN ILLUSTRATOR / FIGMA / SVG-EDIT):
1. `<defs>`: reusable arrowheads / custom wide arrow markers / gradients / soft shadow filter (`#soft-shadow`) / clipPaths for framed assets.
2. `<g id="layer-backdrop">`: soft tinted zones, colored section header bands, subtle dividers (strictly matching `stage1_draft.png` — NO unnecessary heavy outlines!).
3. `<g id="layer-modules">`: semantic cards / borderless pastel pills / foundation bars, each wrapped in `<g id="mod-<slug>">`.
4. `<g id="layer-visual-assets">`: all embedded high-res visual assets (`<image href="__ASSET_...__"/>`) + any crisp vector mini-charts/icons.
5. `<g id="layer-connectors">`: wide colored block arrows / smooth curves / crisp orthogonal arrows matching `stage1_draft.png` (never slicing through text!).
6. `<g id="layer-labels-and-legend">`: all crisp editable `<text>` labels, equations, and compact legend.

XML VALIDITY RULES:
- Escape `&` as `&amp;`, `<` as `&lt;`, `>` as `&gt;` inside text nodes.
- NEVER include meta-labels ("SECTION 1", "ZONE 1", "BASELINE", "NOVEL", "HERO VISUAL ASSETS", "<AF>01").

=== AUTHORITATIVE EXACT TEXT INVENTORY ===
{json.dumps(inventory, ensure_ascii=False, indent=2)}

=== DISTILLED ART DIRECTION PALETTE ===
{json.dumps(pal, ensure_ascii=False, indent=2)}

=== STRUCTURED VISUAL SCHEMA ===
{prompt_body}

OUTPUT ONLY THE COMPLETE, VALID SVG CODE starting with `<svg` and ending with `</svg>`. NO markdown fences, NO explanation.
"""

    masked_path = output_dir / "stage1_draft_masked.png"
    multimodal_inputs: list[Any] = [prompt, "Image 1 — Stage-1 High-Craft Visual Anchor (`stage1_draft.png`):", draft_path]
    if assets and masked_path.is_file():
        multimodal_inputs.extend(
            [
                "Image 2 — Cropped Visual Asset Mask Overlay (`stage1_draft_masked.png`): "
                "Every grey `ASSET:...` rectangle marks the EXACT footprint where the corresponding "
                "`<image href=\"__ASSET_...__\"/>` will appear. DO NOT redraw duplicate cartoon shapes inside these grey boxes "
                "and keep all text/arrows cleanly outside their boundaries!",
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
