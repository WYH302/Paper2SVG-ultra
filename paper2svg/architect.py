"""Stage 3 (Cut 1A): True AIA-Academic-Illustrator Visual Architect + Strict Anti-PPT-Form Rules."""

from __future__ import annotations

import json
import re
from typing import Any

from .archetypes import ArchetypeSpec, get_archetype

# Preserved core layout decision table from AIA-Academic-Illustrator (prompts_config.py / visual-schema.md)
AIA_LAYOUT_SELECTOR_BLOCK = """=== AIA PHASE 1: LAYOUT STRATEGY SELECTOR (PICK THE SINGLE CLEANEST TOPOLOGY!) ===
Choose ONE primary topology (or minimal natural combination) strictly driven by the paper's scientific logic:
1. `Linear Pipeline` (Left -> Right): sequential signal/data transformation, encoding-decoding, sensing-to-reconstruction.
2. `Cyclic / Iterative`: optimization, proximal projection, RL, or alternating solver loops with explicit curved feedback arrows.
3. `Hierarchical Stack` (Top -> Down / Bottom -> Up): multi-scale features, coarse-to-fine pyramids, or tree abstractions.
4. `Parallel / Dual-Stream`: multi-modal fusion, contrastive alignment, dual-domain conditioning branches merging cleanly.
5. `Central Hub`: central foundation model / agent / orchestrator interacting radially with surrounding tools or environments.
"""

ANTI_PPT_FORM_RULES_CONCISE = """=== STRICT ANTI-PPT-FORM & PUBLICATION AESTHETIC RULES (NON-NEGOTIABLE!) ===
You are illustrating a flagship CVPR / NeurIPS / SIGGRAPH / Nature figure — NEVER a 2012 corporate PowerPoint form!
1. **OPEN PURE-WHITE CANVAS (NO BOX-IN-BOX-IN-BOX CAGES!)**:
   - Let concrete scientific visuals, portraits/specimens, 3D/isometric optical or neural blocks, and clean mathematical flow arrows **float openly on pure `#FFFFFF` white space** (like CVPR 2025 LSI / NeurIPS OSEDiff / CVPR JiT)!
   - **STRICTLY FORBIDDEN** to wrap every stage in uniform tall gray rounded boxes lined up like a spreadsheet/tax form!
   - **STRICTLY FORBIDDEN** to draw colored top header strips (red/yellow banners) inside gray cards, or stack rows of oval UI "pill buttons" (`55.9 FPS`, `0.36s Fast`, etc.)!
   - Use AT MOST 1–2 subtle soft borderless tint zones (`#F5F7FA` / `#FDF6EC`, zero dark outline) ONLY if a specific iterative loop or shared module truly benefits from grouping!
2. **CONCRETE PHYSICAL / OPTICAL / 3D-LAYERED OBJECTS (NEVER CRUDE EXCEL BAR CHARTS OR EMPTY TEXT BOXES!)**:
   - Every major stage MUST be anchored by a crisp, authentic **concrete scientific visual object**:
     * Sensing / optics -> realistic target subject + crisp 3D/isometric binary mask / optical lens / photodetector + sleek 1D continuous signal waveform (NEVER ugly jagged Excel bar histograms!).
     * Neural backbones -> sleek **trapezoid / 3D stacked translucent transformer blocks** with a crisp snowflake (❄) for Frozen weights and tiny flame (🔥) for Trainable adapters — NEVER a boring gray box with 3 lines of centered text!
     * Intermediate representations -> real progressive visual frames (blurry -> noisy -> refined -> razor-sharp output), compact clean color-coded token strips, or crisp geometric wireframe/mesh overlays!
3. **ZERO WASTED TOP "TRAINING" BAR**:
   - **NEVER** draw a giant full-width rounded bar across the top of the canvas just to hold 2 loss pills!
   - If training vs. inference losses are needed, show them cleanly as compact, unobtrusive dashed supervision arrows (`L_code`, `L_opt`) tucked tightly above the exact trainable modules — costing near-zero vertical height!
4. **RUTHLESS LABEL DISCIPLINE (`concise` CVPR/NeurIPS/Nature budget)**:
   - **`[EXACT TEXT INVENTORY]` HARD CAP: 10 to 18 crisp labels MAX!** (1–3 words or compact math symbol each!)
   - Drop benchmark speed numbers (`55.9 FPS`, `0.36s / 0.92s`), verbose multi-word jargon stacks, and redundant subtitles — those belong in the paper caption/tables, NOT crammed into pill badges!
   - Keep **40%–50% of the canvas pure `#FFFFFF` open breathing space**!
5. **100% ORIGINAL SUBJECTS (ZERO REFERENCE CONTENT LEAK!)**:
   - If the paper involves human faces, animals, scenes, or medical samples, explicitly specify a **completely fresh, ordinary, natural-looking subject** (e.g., neutral warm daylight studio portrait of a young woman or man with natural dark hair — NEVER copying any distinctive face, blue hair, clothing, or artwork from any reference image!).
"""

DETAILED_SYSTEMS_RULES = """=== DENSE SYSTEMS ARCHITECTURE BUDGET (`detailed` mode — IEEE S&P / USENIX / OSDI) ===
1. **`[EXACT TEXT INVENTORY]` TARGET CAP: 24 to 50 crisp items MAX!** Preserve core subsystems, resource/knowledge plane modules, and closed-loop verification labels cleanly (each 1–4 words; NEVER full sentences).
2. **CLEAN 2-ROW MULTI-PANEL DISCIPLINE**: When matching a multi-panel systems reference (e.g. IEEE S&P / USENIX), keep flat pastel panels crisp with uniform pure `#FFFFFF` gutters, zero 3D plastic clutter, bold uniform borders, and crisp domain micro-visuals (hardware schematic, real multi-frequency waveforms, spectrograms, compact workflow DAG, single-color neural wireframes).
3. **ZERO META-LABELS OR REFERENCE TEXT LEAKS**: Only render paper-grounded labels from `[EXACT TEXT INVENTORY]`.
"""


def generate_visual_schema(
    client: Any,
    paper_analysis: dict[str, Any],
    style_dna: dict[str, Any],
    *,
    font_family: str,
    target_lang: str,
    archetype_override: str | None = None,
    density: str = "auto",
) -> tuple[str, ArchetypeSpec]:
    """Generate a lean, physical-object-first AIA Visual Schema (`schema.md`)."""
    chosen_key = archetype_override or paper_analysis.get("recommended_archetype", "pipeline")
    archetype = get_archetype(chosen_key)
    is_detailed = (density or "auto").strip().lower() == "detailed"

    lang_rule = (
        "All labels in [EXACT TEXT INVENTORY] MUST be crisp academic English (1–3 words or math symbol each; ZERO sentences)."
        if target_lang == "en"
        else "所有 [EXACT TEXT INVENTORY] 中的标签必须为极简学术中文并保留通用英文缩写（每条 2–6 字，绝不写长句）。"
    )

    banner_rule = (
        f"Sleek wide two-column panoramic canvas (~2.3:1 aspect ratio, ~{archetype.canvas_width}x{archetype.canvas_height}) "
        "on pure `#FFFFFF` background with generous white breathing margins."
        if archetype.is_wide_banner
        else f"Balanced worked-example canvas (~1.6:1, ~{archetype.canvas_width}x{archetype.canvas_height}) on pure `#FFFFFF` background."
    )

    aesthetic_block = DETAILED_SYSTEMS_RULES if is_detailed else ANTI_PPT_FORM_RULES_CONCISE
    inv_cap_hint = "24–50 concise items" if is_detailed else "strictly 10–18 ultra-crisp items MAX"

    style_clean = {k: v for k, v in style_dna.items() if k not in ("reference_files", "anchor_files")}

    prompt = f"""You are a Principal Scientific Illustrator & Visual Architect (AIA-Academic-Illustrator × CVPR / NeurIPS / Nature).
Transform the grounded paper analysis + distilled Style DNA into a razor-clean, physical-object-first **[VISUAL SCHEMA]** (`schema.md`)
that produces a breathtaking flagship publication illustration in ONE shot.

{AIA_LAYOUT_SELECTOR_BLOCK}

{aesthetic_block}

=== SELECTED STORY ARCHETYPE ===
- Key: `{archetype.key}` ({archetype.zh_name} / {archetype.en_name})
- Canvas Goal: {banner_rule}

=== DISTILLED STYLE DNA FROM USER'S REFERENCE FOLDER (STYLE ONLY — ZERO CONTENT COPY!) ===
```json
{json.dumps(style_clean, ensure_ascii=False, indent=2)}
```

=== GROUNDED PAPER ANALYSIS (100% SCIENTIFIC TRUTH SOURCE) ===
```json
{json.dumps(paper_analysis, ensure_ascii=False, indent=2)}
```

=== ADDITIONAL MANDATES ===
- {lang_rule}
- Define concrete physical/visual objects inside every zone (e.g., 3D optical setup, portrait frames, sleek trapezoid neural network blocks with ❄/🔥 icon, smooth waveform curve, clean 3-frame progression) — NEVER vague boxes!
- Explicitly specify arrow DIRECTION (`pointing RIGHT`, `pointing DOWN`, `curved loop UP`) for every connection in `[CONNECTIONS]`.
- Keep `[EXACT TEXT INVENTORY]` as the single source of truth ({inv_cap_hint}) using straight ASCII double quotes `"..."`.

OUTPUT FORMAT (clean Markdown directly, NO outer fences):

```markdown
# Visual Schema: <Paper Short Title>

## Scientific Grounding & Art Direction (NOT RENDERED)
- **One-Sentence Story**: <...>
- **Selected AIA Layout**: <Linear Pipeline / Cyclic Iterative / Dual-Stream / etc.>
- **Selected Archetype**: `{archetype.key}`
- **Visual Hero Strategy**: <how physical/optical/3D-block/photographic visuals tell the story in 3 seconds on an open white canvas>

---BEGIN PROMPT---

[STYLE & CANVAS]
- Canvas: {banner_rule}
- Aesthetic Craft: <concrete publication illustration style honoring Style DNA — open pure-white canvas, sleek 3D/isometric or clean flat geometric blocks, crisp portraits/specimens, refined restrained palette; ZERO boxy gray PPT forms, ZERO red/yellow header strips, ZERO oval pill clutter>
- Typography: Clean unified sans-serif (`{font_family}`), bold crisp hierarchy, effortless 2-column print legibility.
- Palette: <3–5 exact harmonious hex colors from Style DNA with 1-to-1 semantic meanings>

[LAYOUT CONFIGURATION]
- Selected Layout: <chosen AIA topology>
- Reading Flow & Whitespace: <left-to-right open horizontal visual progression with ~40–50% open `#FFFFFF` space and ZERO nested box cages>
- Compact Legend (ONLY if frozen/trainable icons are used): <unobtrusive tiny corner cue, e.g. ❄ Frozen  🔥 Trainable>

[ZONE 1: <POSITION - SEMANTIC ROLE>]
- Framing: <open floating on pure white canvas, OR minimal soft borderless tint zone>
- Concrete Visual Objects:
  * <Exact physical/visual object description — e.g., fresh studio portrait of an ordinary young woman with dark hair, 3D binary DMD checkerboard plate + converging lens + photodetector icon, smooth 1D signal waveform>
- Exact Labels in Zone: "<Label 1>", "<Label 2>"

[ZONE 2: <POSITION - SEMANTIC ROLE>]
...

(Use 3–4 airy, well-paced zones — visual heroes dominate ~60% of the canvas!)

[CONNECTIONS]
1. "<Source>" -> "<Target>" | <clean arrow style & color, explicit direction: pointing RIGHT / curved loop UP / thin dashed DOWN> | label: "<optional short math symbol or NONE>"
2. ...

[EXACT TEXT INVENTORY]
- "<Exact string 1>"
- "<Exact string 2>"
(List EVERY SINGLE visible string allowed — {inv_cap_hint}!)

[HARD CONSTRAINTS]
- Render ONLY the strings in `[EXACT TEXT INVENTORY]`. NEVER render structural tags ("ZONE 1", "STYLE", "Framing", "EXACT TEXT INVENTORY").
- NEVER copy any face, person, hair color, object, logo, or text from reference images.
- NO nested gray PPT boxes, NO colored card header strips, NO rows of oval pill buttons, NO giant top Training bar, NO crude Excel bar charts.
- Keep all arrows crisp, unambiguous, and cleanly separated from labels and visuals.

---END PROMPT---
```
"""

    schema_md = client.generate_text(prompt, max_output_tokens=12000, temperature=0.3)
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
