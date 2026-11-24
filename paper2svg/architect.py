"""Stage 3: Visual-First Story Architect (fusing reference Art Direction + AIA + happy-figure + 6 Archetypes)."""

from __future__ import annotations

import json
import re
from typing import Any

from .archetypes import GOLDEN_8_PRINCIPLES_BLOCK, ArchetypeSpec, get_archetype


def generate_visual_schema(
    client: Any,
    paper_analysis: dict[str, Any],
    style_dna: dict[str, Any],
    *,
    font_family: str,
    target_lang: str,
    archetype_override: str | None = None,
) -> tuple[str, ArchetypeSpec]:
    """Generate a rich, visual-first Visual Schema (schema.md) that honors the reference Style DNA."""
    chosen_key = archetype_override or paper_analysis.get("recommended_archetype", "pipeline")
    archetype = get_archetype(chosen_key)

    lang_rule = (
        "All labels in [EXACT TEXT INVENTORY] MUST be crisp academic English (1-4 words each; NO full sentences)."
        if target_lang == "en"
        else "所有 [EXACT TEXT INVENTORY] 中的标签必须为精炼学术中文并保留通用英文缩写（每条 2-6 字，绝不写完整长句子）。"
    )

    banner_rule = (
        f"Wide double-column panoramic banner (~{archetype.canvas_width}:{archetype.canvas_height}, ~2.25:1–2.35:1 aspect ratio). "
        "Keep vertical height sleek and well-proportioned for top-of-page two-column placement."
        if archetype.is_wide_banner
        else f"Worked-example layout ({archetype.canvas_width}x{archetype.canvas_height}, ~1.6:1) with generous vertical space."
    )

    style_clean = {k: v for k, v in style_dna.items() if k not in ("reference_files", "anchor_files")}

    prompt = f"""You are a World-Class Scientific Illustrator & Visual Architect (SIGGRAPH / CVPR / NeurIPS / Nature).
Your job is to transform the grounded paper analysis + the user's distilled Reference Art Direction (`style_dna`)
into a breathtaking, publication-grade **[VISUAL SCHEMA]** (`schema.md`) that looks like a flagship top-conference visual —
**NOT a rigid Visio box-and-arrow flowchart!**

{GOLDEN_8_PRINCIPLES_BLOCK}

=== SELECTED STORY ARCHETYPE ===
- Key: `{archetype.key}` ({archetype.zh_name} / {archetype.en_name})
- Canvas Goal: {banner_rule}
- Flexible Archetype Guidance:
{archetype.layout_idioms}

=== DISTILLED ART DIRECTION FROM USER'S REFERENCE FOLDER (FOLLOW THIS STYLE FAITHFULLY!) ===
```json
{json.dumps(style_clean, ensure_ascii=False, indent=2)}
```

=== GROUNDED PAPER ANALYSIS (100% SCIENTIFIC TRUTH SOURCE) ===
```json
{json.dumps(paper_analysis, ensure_ascii=False, indent=2)}
```

=== ANTI-FLOWCHART, AIRY MINIMALISM & HARD INFORMATION BUDGETS (NON-NEGOTIABLE!) ===
1. **LESS IS MORE — HARD TOP-VENUE DENSITY CAPS**:
   - **`[EXACT TEXT INVENTORY]` HARD CAP: 12 to 24 concise items MAX!** (NEVER output 40–70 tiny labels! Ruthlessly prune hyperparameters, tensor slices `[224:240]`, FPS/attribute counts, axis numbers, and micro-ops like `RMSNorm`/`Zero-init`/`clip`.)
   - **`[HERO VISUAL ASSETS]` HARD CAP: 3 to 6 LARGE, CRISP visual heroes MAX!** (NEVER cram 12–16 postage-stamp thumbnails onto one figure! Give each hero visual generous size and open white breathing space.)
   - **WHITESPACE MANDATE: At least 35% to 45% of the canvas MUST be pure `#FFFFFF` open breathing space!** Prefer a clean **single-tier left-to-right visual flow** (3–4 spacious macro stages with wide gaps) rather than stacking two crowded multi-system rows!
2. **MAKE VISUALS THE HERO (~55-65% OF INK WEIGHT)**:
   - Give the 3–6 hero visuals (e.g., realistic input/output images, physical sensing/3D visual, key intermediate representation, clean schematic) generous size so a reviewer understands the paper's core story in **3 seconds**.
3. **RESPECT THE REFERENCE CONTAINER & ARROW PHILOSOPHY**:
   - Follow `style_dna.container_philosophy` faithfully (e.g., open floating visuals, borderless soft pastel pills, solid colored header bars, or soft shadow cards).
   - DO NOT wrap every element in heavy dark borders or dashed boxes unless references truly do!
   - Use the reference's actual arrow style (e.g., wide colored arrows, soft curved ribbons, or crisp clean connectors) — keep connectors minimal (<= 10 clean arrows total, ZERO spaghetti crossings!).
   - DO NOT hardcode pixel coordinates like `x=32-244` in the schema — describe airy spatial proportions so the Stage-1 image renderer can compose naturally with wide margins!
4. **USE THE REFERENCE PALETTE**:
   - Use the exact harmonious colors distilled in `style_dna.palette`, making the paper's 2–3 core novel innovations pop effortlessly against neutral/standard parts.
5. {lang_rule}
6. **SINGLE SOURCE OF TRUTH FOR TEXT**:
   - List EVERY single visible text string permitted in `[EXACT TEXT INVENTORY]` using straight ASCII double quotes `"..."` (strictly 12–24 items total!). Zero prose sentences, zero meta-label leaks!

OUTPUT FORMAT (follow this clean Markdown structure directly, NO outer markdown fences):

```markdown
# Visual Schema: <Paper Short Title>

## Art Direction & Grounding (NOT RENDERED)
- **One-Sentence Story**: <...>
- **Selected Archetype**: `{archetype.key}`
- **Visual Paradigm (from References)**: `{style_dna.get('visual_paradigm', '')}`
- **Hero Visual Strategy**: <how concrete domain visuals anchor the reader's eye instead of plain text boxes>
- **Novelty Highlight Strategy**: <how core innovations visually pop against standard/baseline parts>

---BEGIN PROMPT---

[ART DIRECTION & VISUAL CRAFT]
- Paradigm: {style_dna.get('visual_paradigm', 'illustrated_scientific_overview')}
- Canvas: {banner_rule}
- Rendering Style: {style_dna.get('rendering_richness', '')}
- Container Philosophy: {style_dna.get('container_philosophy', {}).get('description', '')}
- Arrow Craft: {style_dna.get('arrow_craft', {}).get('description', '')}
- Typography: Clean unified sans-serif (`{font_family}`); bold section titles, semi-bold module labels, crisp readable annotations; strictly NO Serif mixing, NO Comic Sans, NO tiny cramped print text.
- Color Palette (strictly from distilled Style DNA):
  <list exact hex colors and their 1-to-1 semantic meanings concisely>

[HERO VISUAL ASSETS]
(Rich concrete visual illustrations / photorealistic specimens / 3D objects / tactile textures / plots / bespoke icons that MUST be rendered large, crisp, and high-craft!)
1. "<Asset Name>" — <exact visual depiction, e.g., photorealistic input face crop / textured natural marble slab with vein field / 3D voxelized bunny inside transparent bounding cube / crisp loss curve or heatmap> | placement: <which region>
2. ...

[SPATIAL COMPOSITION & FLOW]
- Reading Order: <natural visual flow matching both the paper logic and reference rhythm>
- Visual Balance: <relative spatial proportions, open breathing room, header banners / soft tint zones / floating hero visuals>
- Legend / Key (if needed): <compact, unobtrusive legend showing semantic colors or trainable/frozen cues>

[SECTION 1: <POSITION - SEMANTIC ROLE>]
- Container / Framing: <follow reference philosophy — e.g., open floating visual group with subtle bottom caption, OR soft borderless pastel card, OR colored header banner>
- Visual Content & Modules:
  * <Rich visual description + `[BASELINE]` or `[NOVEL]` role + exact short label>
  * ...
- Labels in this Section: "<Label 1>", "<Label 2>"

[SECTION 2: <POSITION - SEMANTIC ROLE>]
...

(Use 2-5 well-paced visual sections — airy, uncluttered, visually captivating!)

[FLOW CONNECTIONS]
1. "<Source>" -> "<Target>" | <arrow style matching reference craft, e.g. wide warm-orange block arrow / smooth curved blue arrow / dashed feedback return> | label: "<optional 1-2 word symbol or NONE>"
2. ...

[EXACT TEXT INVENTORY]
- "<Exact string 1>"
- "<Exact string 2>"
- (List EVERY SINGLE visible text string allowed in the figure — concise 1-4 word phrases/symbols only!)

[HARD CONSTRAINTS]
- Render ONLY the strings in `[EXACT TEXT INVENTORY]`. NEVER render structural words like "SECTION 1", "HERO VISUAL ASSETS", "BASELINE", "NOVEL", "Container", or "EXACT TEXT INVENTORY".
- NEVER copy any text, logos, or method names from reference images.
- Make concrete visuals large and crisp; avoid unnecessary nested black boxes; ensure every text label is effortlessly readable when shrunk to 7-inch two-column print width.

---END PROMPT---
```
"""

    schema_md = client.generate_text(prompt, max_output_tokens=14000, temperature=0.3)
    cleaned = schema_md.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip() + "\n", archetype


def extract_prompt_body(schema_md: str) -> str:
    """Extract the text between ---BEGIN PROMPT--- and ---END PROMPT---."""
    m = re.search(r"---BEGIN PROMPT---(.*?)---END PROMPT---", schema_md, re.DOTALL)
    if m:
        return m.group(1).strip()
    return schema_md.strip()


def extract_exact_text_inventory(schema_md: str) -> list[str]:
    """Extract all quoted labels from the [EXACT TEXT INVENTORY] section."""
    body = extract_prompt_body(schema_md)
    m = re.search(
        r"\[EXACT TEXT INVENTORY\](.*?)(?=\n\[[A-Z0-9 _-]+\]|\Z)",
        body,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return []
    section = m.group(1)
    items: list[str] = []
    seen: set[str] = set()
    for line in section.splitlines():
        line = line.strip()
        if not line or line.startswith("("):
            continue
        quoted = re.findall(r'["“]([^"”]+)["”]', line)
        if quoted:
            for q in quoted:
                q_clean = q.strip().strip('“”"\'`')
                if q_clean and q_clean not in seen:
                    seen.add(q_clean)
                    items.append(q_clean)
        elif line.startswith(("-", "*")):
            raw = line.lstrip("-* ").strip().strip('“”"\'`')
            if raw and not raw.startswith("(") and raw not in seen:
                seen.add(raw)
                items.append(raw)
    return items
