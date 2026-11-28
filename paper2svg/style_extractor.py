"""Stage 2: Visual Art-Director Style DNA extractor — faithfully mirrors reference aesthetic without flowchart lock-in."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Neutral fallback ONLY used when the user provides ZERO reference images
DEFAULT_STYLE_DNA: dict[str, Any] = {
    "visual_paradigm": "illustrated_scientific_overview",
    "rendering_richness": (
        "Rich top-venue publication visual illustration combining concrete domain visuals "
        "(photorealistic specimens, 3D/isometric schematic objects, mini-charts, and expressive cohesive icons) "
        "with clean borderless or softly elevated pastel cards and wide directional flow arrows."
    ),
    "palette": {
        "canvas_bg": "#FFFFFF",
        "neutral_surface": "#F3F5F8",
        "neutral_stroke": "#CBD5E1",
        "accent_1_fill": "#DCEEFF",
        "accent_1_stroke": "#2B6CB0",
        "accent_2_fill": "#E6F7EE",
        "accent_2_stroke": "#2F855A",
        "accent_3_fill": "#FFF1DE",
        "accent_3_stroke": "#DD6B20",
        "accent_4_fill": "#EFE8FF",
        "accent_4_stroke": "#6B46C1",
        "header_band_bg": "#E2E8F0",
        "text_primary": "#111827",
        "text_secondary": "#4B5563",
        "arrow_primary": "#4A6FA5",
        "arrow_secondary": "#94A3B8",
    },
    "container_philosophy": {
        "style": "soft_borderless_cards_with_colored_headers",
        "use_heavy_dark_outlines": False,
        "use_dashed_stage_frames": False,
        "module_rx": 12,
        "panel_rx": 16,
        "stroke_width_standard": 1.2,
        "stroke_width_focal": 2.0,
        "shadow_style": "soft_elevation",
        "description": (
            "Prefer open visual breathing room, soft tinted zones without heavy dark outlines, "
            "or subtle solid header bars. Never wrap every sub-item in nested dark boxes."
        ),
    },
    "arrow_craft": {
        "style": "wide_colored_block_or_smooth_flow_arrows",
        "stroke_width": 3.0,
        "description": "Use confident, smooth colored directional arrows or clean shaded block arrows matching top-venue visual polish.",
    },
    "visual_hero_guidance": (
        "Make concrete domain imagery (sample inputs/outputs, 3D/isometric visuals, physical fields, plots, or bespoke icons) "
        "prominent hero elements (~55-65% visual weight) rather than shrinking them into tiny box corners."
    ),
    "layout_rhythm_from_refs": "Balanced horizontal scientific narrative with generous margins, clear visual focal points, and airy grouping.",
    "anti_flowchart_notes": "Avoid rigid Visio/draw.io box-in-box-in-box wireframes and thin black connector spaghetti.",
    "top_anchor_indices": [],
}


def discover_reference_images(
    refs_dir: Optional[Path],
    max_images: int = 6,
    primary_ref: Optional[Path] = None,
) -> list[Path]:
    """Discover image files inside a reference folder, prioritizing largest/sharpest figures.

    If `primary_ref` is explicitly provided, returns `[primary_ref]` directly as the sole
    authoritative visual style anchor so other folder images never dilute its aesthetic.
    """
    if primary_ref is not None:
        if not primary_ref.is_file():
            raise FileNotFoundError(f"Primary reference image not found: {primary_ref}")
        if primary_ref.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported primary reference image format: {primary_ref.suffix}")
        return [primary_ref]

    if refs_dir is None:
        return []
    if not refs_dir.exists():
        raise FileNotFoundError(f"Reference folder not found: {refs_dir}")
    if refs_dir.is_file() and refs_dir.suffix.lower() in IMAGE_EXTENSIONS:
        return [refs_dir]

    found: list[tuple[float, Path]] = []
    for path in sorted(refs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.name.startswith("."):
            continue
        try:
            with Image.open(path) as img:
                w, h = img.size
            if w < 120 or h < 80:
                continue
            ratio = w / max(1, h)
            wide_bonus = 1.25 if ratio >= 1.35 else 1.0
            score = (w * h) ** 0.5 * wide_bonus
            found.append((score, path))
        except Exception:
            continue

    found.sort(key=lambda pair: pair[0], reverse=True)
    return [p for _, p in found[:max_images]]


def extract_style_dna(
    client: Any,
    ref_images: list[Path],
    target_archetype: str,
) -> dict[str, Any]:
    """Act as a Principal Scientific Art Director to extract the TRUE visual artistry of the reference folder.

    Unlike rigid templates, this NEVER forces a pre-baked blue/copper/grey wireframe palette.
    It faithfully observes the references' actual rendering richness (3D, photorealistic specimens,
    bespoke icons, borderless pills, wide colored arrows, header bands, mini-plots) while strictly
    isolating content ('借骨架与画风，不借画中文字和模块').
    """
    if not ref_images:
        dna = json.loads(json.dumps(DEFAULT_STYLE_DNA))
        dna["reference_files"] = []
        dna["anchor_files"] = []
        return dna

    contents: list[Any] = [
        f"""You are a Principal Scientific Art Director (Nature / SIGGRAPH / CVPR / NeurIPS) analyzing {len(ref_images)} reference paper figures supplied by the user.

YOUR CORE MISSION — TRUE ART-DIRECTION TRANSFER ("借骨架与画风，绝不降维成素颜流程图"):
Look closely at what actually makes these reference figures visually stunning, modern, and distinctly publication-grade!
DO NOT flatten them into a generic Visio flowchart with blue/copper boxes and dashed outlines unless the references themselves truly look like that!

OBSERVE CAREFULLY ACROSS THE REFERENCE IMAGES:
1. **Visual Paradigm & Rendering Richness**:
   - Do they showcase rich 3D/isometric rendered objects, realistic sample photos/textures, floating visual sequences, expressive custom scientific illustrations/icons, or rich mini-charts/heatmaps?
   - How prominent are visual artifacts vs text boxes? (e.g., open floating visuals with captions beneath, borderless pastel pills, bold colored header bands, or soft cards?)
2. **True Color Palette (EXTRACT DIRECTLY FROM THE IMAGES!)**:
   - Sample the ACTUAL dominant harmonious hex palette observed in the best reference image(s) (could be soft cyan/violet/mint, vibrant orange/blue/teal, crisp monochrome-grey + soft sky pills, warm editorial pastel, etc.).
   - NEVER substitute a canned default palette if the references clearly have their own aesthetic!
3. **Container & Border Philosophy**:
   - Are modules **borderless soft filled pills/rectangles** (like CVPR JiT)? Are visuals **floating openly without boxes** (like SIGGRAPH Chapper)? Do they use **solid colored header strips** (like AutoFigure-Edit)? Or subtle soft elevation cards?
   - Explicitly note whether heavy dark outlines or dashed frames should be AVOIDED!
4. **Arrow & Connector Craft**:
   - Are arrows **wide filled colored block arrows** (orange/blue/gradient), **soft curved ribbons**, **thick pastel connectors**, or crisp minimal vectors?
5. **PRIMARY STYLE WINNER RULE (CRITICAL — ZERO STYLE DILUTION & PREFER AIRY PUBLICATION CRAFT!)**:
   - When multiple reference images are provided, **DO NOT average or blend conflicting aesthetics** across different images (e.g., never mix 3D glossy shading from one image with flat 2D black-bordered pastel cards from another)!
   - Instead, pick the **SINGLE** visually cleanest, sharpest, most modern, open-canvas reference image (`top_anchor_indices`: `[winner_idx]`, strictly 1 index!) as the **Primary Style Winner** — strongly preferring airy, open-white, visually illustrative CVPR/NeurIPS/Nature frameworks with sleek physical/neural blocks and generous whitespace over cramped gray multi-card grids!
   - Extract the palette, container borders, shadow policy, arrow craft, and rendering richness strictly from that single winner!
6. **IGNORE REVIEW ANNOTATIONS & PAGE ARTIFACTS**:
   - Completely IGNORE any hand-drawn red/blue pen marks, review circles, highlighter scribbles, PDF figure captions (`Figure 1: ...`), page headers/footers, or watermarks overlaid on reference screenshots. Extract ONLY the clean underlying figure design!

STRICT CONTENT & SUBJECT ISOLATION:
- NEVER copy, transcribe, or mention ANY paper-specific text labels, model names, datasets, faces/portraits/identities, or method modules from the reference images. Describe ONLY visual style, composition grammar, color hexes, typography feel, container style, arrow style, and illustration craft!

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown fences) matching this schema (replace EVERY angle-bracket value with your ACTUAL visual observations from the Primary Style Winner image!):
{{
  "visual_paradigm": "<concise slug, e.g. 3d_visual_progression | borderless_pastel_architecture | icon_rich_scientific_framework | multi_band_infographic | soft_card_editorial>",
  "primary_anchor_reason": "<1 crisp sentence explaining why this single reference image was selected as the Primary Style Winner>",
  "rendering_richness": "<2-3 specific sentences describing the exact illustration/3D/photo/icon craft seen in the Primary Style Winner that MUST be recreated in our figure>",
  "palette": {{
    "canvas_bg": "#FFFFFF",
    "neutral_surface": "<exact observed light neutral hex>",
    "neutral_stroke": "<exact observed subtle neutral border hex or 'none'>",
    "accent_1_fill": "<exact observed primary accent fill hex>",
    "accent_1_stroke": "<exact observed primary accent stroke/darker hex>",
    "accent_2_fill": "<exact observed secondary accent fill hex>",
    "accent_2_stroke": "<exact observed secondary accent stroke/darker hex>",
    "accent_3_fill": "<exact observed tertiary accent fill hex (or repeat harmonious shade)>",
    "accent_3_stroke": "<exact observed tertiary accent stroke hex>",
    "accent_4_fill": "<optional 4th soft tint hex>",
    "accent_4_stroke": "<optional 4th stroke hex>",
    "header_band_bg": "<observed section header/band tint hex>",
    "text_primary": "<observed main dark text hex>",
    "text_secondary": "<observed muted caption text hex>",
    "arrow_primary": "<observed main arrow color hex>",
    "arrow_secondary": "<observed secondary arrow color hex>"
  }},
  "container_philosophy": {{
    "style": "<e.g. open_floating_visuals | borderless_soft_pills | header_band_cards | soft_shadow_cards | minimal_hairline_zones>",
    "use_heavy_dark_outlines": false,
    "use_dashed_stage_frames": false,
    "module_rx": 12,
    "panel_rx": 16,
    "stroke_width_standard": 1.2,
    "stroke_width_focal": 2.0,
    "shadow_style": "<none | subtle_soft_shadow>",
    "description": "<concrete instructions on how containers, cards, or open spaces are styled in the references>"
  }},
  "arrow_craft": {{
    "style": "<e.g. wide_colored_block_arrows | soft_curved_flow_ribbons | thick_pastel_arrows | clean_minimal_vectors>",
    "stroke_width": 3.0,
    "description": "<concrete description of arrow shape, color, thickness, and routing in the references>"
  }},
  "visual_hero_guidance": "<how visual objects (photos, 3D meshes, textures, charts, icons) dominate the canvas relative to text in the references>",
  "layout_rhythm_from_refs": "<spatial composition rhythm, grouping cues, label placement, and whitespace balance learned from the references>",
  "anti_flowchart_notes": "<specific visual traits to AVOID so our diagram matches the references instead of turning into a dry box-and-arrow flowchart>",
  "top_anchor_indices": [1]
}}
"""
    ]

    for idx, img_path in enumerate(ref_images, 1):
        contents.append(f"Reference Image #{idx} ({img_path.name}):")
        contents.append(img_path)

    raw = client.generate_multimodal(contents, max_output_tokens=6000, temperature=0.25)
    parsed = _parse_style_json(raw)

    # Merge safely onto fallback only for missing keys
    merged = json.loads(json.dumps(DEFAULT_STYLE_DNA))
    for top_key in (
        "visual_paradigm",
        "primary_anchor_reason",
        "rendering_richness",
        "visual_hero_guidance",
        "layout_rhythm_from_refs",
        "anti_flowchart_notes",
    ):
        if isinstance(parsed.get(top_key), str) and parsed[top_key].strip():
            merged[top_key] = parsed[top_key].strip()

    for dict_key in ("palette", "container_philosophy", "arrow_craft"):
        if isinstance(parsed.get(dict_key), dict):
            for k, v in parsed[dict_key].items():
                if v is not None and not (isinstance(v, str) and "<" in v and ">" in v):
                    merged[dict_key][k] = v

    raw_indices = parsed.get("top_anchor_indices") or [1]
    # Primary Style Winner: pin strictly to the SINGLE best reference anchor (length 1) to prevent style dilution
    valid_indices = [i for i in raw_indices if isinstance(i, int) and 1 <= i <= len(ref_images)][:1]
    if not valid_indices:
        valid_indices = [1]
    merged["top_anchor_indices"] = valid_indices
    merged["reference_files"] = [str(p) for p in ref_images]
    merged["anchor_files"] = [str(ref_images[i - 1]) for i in valid_indices]
    return merged


def render_style_guide_markdown(style_dna: dict[str, Any]) -> str:
    """Render a clean human-readable Art-Direction Style Guide Markdown."""
    pal = style_dna.get("palette", {})
    cont = style_dna.get("container_philosophy", {})
    arr = style_dna.get("arrow_craft", {})
    lines = [
        "# Distilled Visual Art Direction & Style DNA (借骨架与画风，不借画中内容)",
        "",
        f"- **Reference Images Scanned**: {len(style_dna.get('reference_files', []))}",
        f"- **Primary Style Winner**: {', '.join(Path(p).name for p in style_dna.get('anchor_files', [])) or 'Default Editorial Style'}",
        *(
            [f"- **Selection Rationale**: {style_dna['primary_anchor_reason']}"]
            if style_dna.get("primary_anchor_reason")
            else []
        ),
        f"- **Visual Paradigm**: `{style_dna.get('visual_paradigm', '')}`",
        f"- **Rendering Richness**: {style_dna.get('rendering_richness', '')}",
        f"- **Visual Hero Guidance**: {style_dna.get('visual_hero_guidance', '')}",
        f"- **Layout Rhythm**: {style_dna.get('layout_rhythm_from_refs', '')}",
        f"- **Anti-Flowchart Guardrail**: {style_dna.get('anti_flowchart_notes', '')}",
        "",
        "## Container & Border Philosophy",
        f"- **Style**: `{cont.get('style')}` | **Heavy dark outlines**: `{cont.get('use_heavy_dark_outlines')}` | **Dashed frames**: `{cont.get('use_dashed_stage_frames')}` | **Shadow**: `{cont.get('shadow_style')}`",
        f"- **Notes**: {cont.get('description', '')}",
        "",
        "## Arrow & Connector Craft",
        f"- **Style**: `{arr.get('style')}` | **Stroke**: `{arr.get('stroke_width')}`",
        f"- **Notes**: {arr.get('description', '')}",
        "",
        "## Extracted Color Palette",
        "| Role | Hex |",
        "| :--- | :--- |",
    ]
    for k, v in pal.items():
        lines.append(f"| `{k}` | `{v}` |")
    return "\n".join(lines) + "\n"


def _parse_style_json(raw: str) -> dict[str, Any]:
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
        return {}
