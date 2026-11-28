"""Stage 3 (Cut 1A): Clean AIA-Academic-Illustrator Visual Architect — Zero Spaghetti Arrows, Zero Overlap."""

from __future__ import annotations

import json
import re
from typing import Any

from .archetypes import ArchetypeSpec, get_archetype
from .config import resolve_stage_effort

AIA_LAYOUT_SELECTOR_BLOCK = """=== AIA PHASE 1: LAYOUT STRATEGY SELECTOR ===
Choose ONE clean primary topology driven by the paper's core story:
1. `Linear Pipeline` (Left -> Right): sequential signal/data transformation, sensing-to-reconstruction, encoding-decoding.
2. `Cyclic / Iterative`: optimization, proximal solver, or RL loop with ONE clean outer return arc.
3. `Hierarchical Stack` (Top -> Down / Bottom -> Up): multi-scale features or tree abstractions.
4. `Parallel / Dual-Stream`: two parallel branches merging cleanly into one unified output.
5. `Central Hub`: central coordinator interacting radially with clean surrounding tools.
"""

CLEAN_TOPOLOGY_RULES_CONCISE = """=== STRICT ZERO-COLLISION, ZERO-SPAGHETTI & PUBLICATION CRAFT RULES (NON-NEGOTIABLE!) ===
Look at why flagship CVPR / NeurIPS / AutoFigure-Edit figures look breathtaking at a glance:
EVERY visual object is a clean standalone island, EVERY text label sits in open white space, and EVERY arrow is crystal clear!

1. **STRICT ARROW DISCIPLINE (HARD CAP: 5 TO 7 CLEAN ARROWS TOTAL — ZERO SPAGHETTI!)**:
   - **MAX 5 to 7 arrows** across the ENTIRE figure!
   - Use clean **straight horizontal/vertical orthogonal arrows** connecting the right edge of Stage k -> left edge of Stage k+1!
   - Allow **AT MOST 1–2** clean outer bypass/feedback arrows routed cleanly around the outer bottom/top white margin (NEVER cutting diagonally through the middle!).
   - **STRICTLY FORBIDDEN**:
     * NEVER draw arrows pointing to floating title text (e.g., NEVER draw a stub arrow pointing up at `MiniT2I-B/16`!).
     * NEVER draw two parallel/overlapping arrows between the same modules!
     * NEVER start or end an arrow in empty white space — every arrow MUST connect two named visible objects!
2. **ISLAND SEPARATION DISCIPLINE (HARD CAP: 6 TO 10 STANDALONE VISUAL ISLANDS — ZERO FUSED SEAMS!)**:
   - **MAX 6 to 10 crisp, self-contained visual objects** across the entire figure (NEVER 20–26 crowded micro-widgets!).
   - **EVERY visual object (portrait, DMD optical setup, neural block, waveform, face strip) MUST be an independent island surrounded by >= 20px of open white/uniform margin on ALL 4 sides!**
   - **STRICTLY FORBIDDEN** to fuse a 3D neural block directly onto a sub-card with text across the seam (e.g., NEVER merge 3D transformer slabs onto a card containing `LP-Flow` text!). Put the neural block and the flow progression strip cleanly SIDE-BY-SIDE or neatly stacked with generous white separation!
   - **ZERO TEXT INSIDE OR TOUCHING ICONS**: Place every label cleanly ABOVE or BELOW its visual island with clear empty margin — NEVER overlapping any graphic, border, or arrow!
3. **OPEN PURE-WHITE CANVAS (NO BOXY PPT FORMS!)**:
   - Let standalone visual islands float cleanly on pure `#FFFFFF` white space (45–55% open pure white!).
   - **NO** uniform tall gray rounded boxes wrapping every stage, **NO** red/yellow header strips inside cards, **NO** giant top `Training` bar, **NO** oval UI pill buttons, and **NO** crude Excel bar charts!
   - Use AT MOST 1–2 subtle soft pale-tint grouping zones (`#F4F7FB` / `#FFF7ED`, hairline or borderless) ONLY where functionally helpful, with generous padding so nothing touches the edge!
4. **CRISP TYPOGRAPHY, NO GIANT FIGURE TITLE & ZERO LITERAL UNDERSCORES (`_`)**:
   - **NEVER put a giant paper-title banner across the top of the diagram** (e.g., DO NOT put `"PhysFlow"` as a huge standalone top header — the paper's LaTeX `Figure 1:` caption already gives the title, and removing it gives 25% more vertical scale to the actual architecture!).
   - **`[EXACT TEXT INVENTORY]` HARD CAP: 10 to 16 ultra-crisp labels MAX!** (1–3 words each; drop FPS/latency numbers like `55.9 FPS` / `0.36s`!)
   - **STRICTLY FORBIDDEN to use raw programming underscores `_` in visible labels!** Write `x₀ → x₁` (Unicode subscripts) or clean readable short phrases (`Semantic`, `Identity`, `Geometry`, `y`) instead of ugly `c_sem`, `c_id`, `c_geo`, `x_0 -> x_1`!
   - When placing 3 sibling sub-labels horizontally (e.g., `Semantic`, `Identity`, `Geometry`), leave generous horizontal spacing (`>= 30px` gap between adjacent labels) so words NEVER touch each other!
5. **100% ORIGINAL SUBJECTS (ZERO REFERENCE CONTENT LEAK!)**:
   - If the paper shows a human face/scene/object, specify a fresh, ordinary, natural studio subject with ZERO resemblance to any reference image portrait/face/hair/artwork.
"""

DETAILED_SYSTEMS_RULES = """=== DENSE SYSTEMS ARCHITECTURE BUDGET (`detailed` mode — IEEE S&P / USENIX / OSDI) ===
1. **`[EXACT TEXT INVENTORY]` TARGET CAP: 24 to 48 crisp items MAX!** Preserve core subsystems cleanly (1–4 words each; zero sentences; zero raw `_` underscores).
2. **CLEAN 2-ROW MULTI-PANEL DISCIPLINE**: Uniform crisp flat pastel panels with wide `#FFFFFF` gutters, zero overlapping shapes, zero stray arrows, every icon cleanly separated from every text label!
3. **EXACT ORTHOGONAL CONNECTORS**: Clean inter-panel block arrows centered in white gutters with explicit direction (`pointing RIGHT/DOWN/UP/LEFT`).
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
    """Generate a lean, clean-topology, zero-overlap AIA Visual Schema (`schema.md`)."""
    chosen_key = archetype_override or paper_analysis.get("recommended_archetype", "pipeline")
    archetype = get_archetype(chosen_key)
    is_detailed = (density or "auto").strip().lower() == "detailed"

    lang_rule = (
        "All labels in [EXACT TEXT INVENTORY] MUST be crisp academic English (1–3 words each; ZERO raw underscores `_`; ZERO sentences)."
        if target_lang == "en"
        else "所有 [EXACT TEXT INVENTORY] 中的标签必须为极简学术中文并保留通用英文缩写（每条 2–6 字，严禁出现下划线 `_` 或长句）。"
    )

    banner_rule = (
        f"Clean wide two-column panoramic canvas (~2.3:1 ratio, ~{archetype.canvas_width}x{archetype.canvas_height}) "
        "on pure `#FFFFFF` background with wide symmetric outer margins."
        if archetype.is_wide_banner
        else f"Balanced worked-example canvas (~1.6:1, ~{archetype.canvas_width}x{archetype.canvas_height}) on pure `#FFFFFF` background."
    )

    aesthetic_block = DETAILED_SYSTEMS_RULES if is_detailed else CLEAN_TOPOLOGY_RULES_CONCISE
    inv_cap_hint = "24–48 concise items" if is_detailed else "strictly 10–16 crisp items MAX (ZERO raw `_` underscores!)"
    conn_cap_hint = "clean orthogonal arrows only" if is_detailed else "STRICTLY 5 to 7 clean non-crossing orthogonal arrows MAX!"

    style_clean = {k: v for k, v in style_dna.items() if k not in ("reference_files", "anchor_files")}

    prompt = f"""You are a Principal Scientific Illustrator (AIA-Academic-Illustrator × CVPR / NeurIPS / Nature).
Convert the grounded paper analysis + Style DNA into a **CRYSTAL-CLEAN, ZERO-OVERLAP, ZERO-SPAGHETTI** `[VISUAL SCHEMA]` (`schema.md`).

{AIA_LAYOUT_SELECTOR_BLOCK}

{aesthetic_block}

=== SELECTED ARCHETYPE ===
- `{archetype.key}` ({archetype.zh_name} / {archetype.en_name}) | {banner_rule}

=== DISTILLED STYLE DNA (STYLE ONLY — ZERO CONTENT COPY!) ===
```json
{json.dumps(style_clean, ensure_ascii=False, indent=2)}
```

=== GROUNDED PAPER ANALYSIS ===
```json
{json.dumps(paper_analysis, ensure_ascii=False, indent=2)}
```

=== ADDITIONAL MANDATES ===
- {lang_rule}
- Keep `[CONNECTIONS]` minimal and crystal-clear ({conn_cap_hint}). Every arrow MUST connect two named visible objects cleanly with ZERO diagonal spiderwebs and ZERO arrows pointing at floating title text!
- Keep `[EXACT TEXT INVENTORY]` as the single source of truth ({inv_cap_hint}) using straight ASCII double quotes `"..."`.

OUTPUT FORMAT (clean Markdown directly, NO outer fences):

```markdown
# Visual Schema: <Paper Short Title>

## Scientific Grounding & Art Direction (NOT RENDERED)
- **One-Sentence Story**: <...>
- **Selected AIA Layout**: <Linear Pipeline / Cyclic Iterative / Dual-Stream>
- **Visual Clarity Strategy**: <how 6–10 crisp standalone visual islands + 5–7 clean orthogonal arrows tell the entire story in 3 seconds with zero overlap>

---BEGIN PROMPT---

[STYLE & CANVAS]
- Canvas: {banner_rule}
- Aesthetic Craft: Clean open-white CVPR/Nature scientific illustration with flat/subtle-isometric vector-like clarity, 45–55% pure `#FFFFFF` breathing space, crisp standalone visual islands separated by wide white gaps; ZERO overlapping shapes, ZERO fused 3D-on-card seams, ZERO nested gray PPT boxes, ZERO red/yellow header strips, ZERO oval pills, ZERO crude bar charts.
- Typography: Clean unified sans-serif (`{font_family}`), bold crisp hierarchy, every text label placed cleanly ABOVE or BELOW its graphic with wide open margin (NEVER touching graphics or arrows; ZERO raw `_` underscores!).
- Palette: <3–4 exact harmonious hex colors from Style DNA>

[LAYOUT CONFIGURATION]
- Selected Layout: <chosen AIA topology>
- Spatial Grid: <3–4 spacious horizontal columns left-to-right aligned on one clean centerline, with wide white gutters and zero clutter>
- Legend (ONLY if ❄/🔥 icons are used): <tiny clean corner legend `"❄ Frozen  🔥 Trainable"`>

[ZONE 1: LEFT — <SEMANTIC ROLE>]
- Framing: Open on pure `#FFFFFF` canvas (no box border)
- Standalone Visual Islands (2–3 max, each separated by >=25px white gap):
  * <Clean standalone visual object>
- Clean Labels (placed strictly above/below objects): "<Label 1>", "<Label 2>"

[ZONE 2: CENTER-LEFT — <SEMANTIC ROLE>]
...

(STRICTLY 3–4 airy zones total; MAX 6–10 visual islands across the ENTIRE figure!)

[CONNECTIONS]
({conn_cap_hint})
1. "<Source Object>" -> "<Target Object>" | straight horizontal black 2.5px arrow pointing RIGHT | label: NONE
2. ...

[EXACT TEXT INVENTORY]
- "<Exact string 1>"
- "<Exact string 2>"
({inv_cap_hint})

[HARD CONSTRAINTS]
- Render ONLY the strings in `[EXACT TEXT INVENTORY]`. NEVER render structural tags ("ZONE 1", "STYLE", "Framing", "CONNECTIONS", "EXACT TEXT INVENTORY").
- ZERO stray arrows, ZERO diagonal crossing spiderwebs, ZERO arrows pointing into empty space or floating title text.
- ZERO overlapping shapes and ZERO text touching or clipped by any graphic/icon/border.
- NEVER copy any face, person, hair color, object, logo, or text from reference images.

---END PROMPT---
```
"""

    base_eff = getattr(getattr(client, "settings", None), "reasoning_effort", "medium")
    eff = resolve_stage_effort(base_eff, "medium")
    schema_md = client.generate_text(prompt, max_output_tokens=10000, temperature=0.25, reasoning_effort=eff)
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
