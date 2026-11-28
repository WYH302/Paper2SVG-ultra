"""Stage 4 (Cut 1B): AIA-Style Visual Anchor Renderer + Aesthetic & Anti-Copy Gate."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

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
    """Use the EXACT 1:1 native pixel dimensions of the Stage-1 anchor figure (AutoFigure-Edit
    zero-scaling coordinate system) so every cropped icon/graphic, arrow, and text label maps 1:1.
    """
    dw, dh = draft_img.size
    if dw < 64 or dh < 64:
        return archetype.canvas_width, archetype.canvas_height
    return int(dw), int(dh)


def _parse_gate_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except Exception:
            pass
    return {}


def audit_stage1_draft_aesthetics(
    client: Any,
    draft_path: Path,
    schema_md: str,
    *,
    reference_paths: list[Path],
    resolved_density: str,
) -> dict[str, Any]:
    """AIA Step 4 Visual Gate: brutally inspect stage1_draft.png BEFORE entering SVG conversion.

    Detects:
      1. Reference content/portrait/text contamination (e.g., copying a face from a reference paper)
      2. Ugly boxy PPT-form degradation (nested gray boxes + colored header strips + oval pills + crude bars)
      3. Spelling typos, garbled text, clipped borders, or broken/contradictory arrows
    """
    inventory = extract_exact_text_inventory(schema_md)
    prompt_body = extract_prompt_body(schema_md)

    ppt_rule = (
        "In `detailed` systems-architecture mode, clean flat pastel multi-panel grids (like IEEE S&P / USENIX) ARE acceptable "
        "if crisp, aligned, and uncluttered; set `is_ppt_box_form=true` ONLY if panels look messy, dated, gradient-heavy, or overcrowded."
        if resolved_density == "detailed"
        else "In `concise` CVPR/NeurIPS/Nature overview mode, set `is_ppt_box_form=true` IMMEDIATELY if the figure looks like an ugly "
        "corporate PowerPoint form: e.g., uniform tall gray rounded boxes wrapping every stage, colored red/yellow header strips inside gray cards, "
        "a giant wasted top `Training` bar, clusters of oval pill buttons, or ugly crude Excel bar charts instead of clean open-white scientific/physical visuals!"
    )

    audit_prompt = f"""You are a Brutally Honest Principal Art Director for CVPR / NeurIPS / Nature reviewing a freshly generated raster academic figure (`Image 1`) BEFORE it gets vectorized into editable SVG.

DO NOT BE POLITE. If this figure looks mediocre, boxy, cluttered, or copied ANY person/subject/text from the reference image, FAIL IT decisively so we regenerate a stunning 9.5/10 figure!

CHECK ALL 4 GATES STRICTLY:
1. **REFERENCE CONTENT LEAK (`reference_content_leaked`)**:
   - Compare `Image 1` against the Style Reference Image(s) (if attached below).
   - Did `Image 1` copy ANY human face/identity, distinctive hair color, animal, artwork, logo, or text label from a reference image?
   - Set `reference_content_leaked: true` if there is ANY recognizable subject/identity/text copy!
2. **ANTI-PPT-FORM & PUBLICATION BEAUTY (`is_ppt_box_form`, `aesthetic_score_10`)**:
   - {ppt_rule}
   - Score `aesthetic_score_10` honestly from 1.0 to 10.0:
     * `9.0–10.0`: Truly looks like a CVPR Best Paper / Nature flagship figure — clean open white breathing space, gorgeous physical/optical/3D-block/photographic visuals, effortless 3-second comprehension, publication typography.
     * `7.5–8.9`: Decent layout but has noticeable visual flaws (minor clutter, weak visual metaphor, uneven alignment, or a couple of label/arrow issues).
     * `< 7.5`: Ugly boxy PPT form, cramped text, crude bar charts, wasted gray headers, or amateur aesthetic.
3. **TEXT & SPELLING ACCURACY (`text_or_arrow_errors`)**:
   - Check EVERY visible word in `Image 1` against the Expected Text Inventory below: report ANY misspelling, garbled letters, duplicate/unwanted meta-labels ("ZONE 1", "SECTION"), cut-off edges, or confusing/reversed arrows.
4. **ACTIONABLE FIX PLAN (`revision_mode`, `fix_instructions`)**:
   - `revision_mode`:
     * `"pass"`: ONLY if `reference_content_leaked == false` AND `is_ppt_box_form == false` AND `aesthetic_score_10 >= 8.5` AND `text_or_arrow_errors` is empty!
     * `"regenerate_clean"`: if `reference_content_leaked == true` OR `is_ppt_box_form == true` OR `aesthetic_score_10 < 7.8` (needs fresh cleaner composition — drop contaminating reference portrait!).
     * `"targeted_edit"`: if the overall art direction & composition are already gorgeous (`>= 7.8`, no leak, not PPT-boxy) and ONLY specific text typos, arrow tweaks, or small element fixes are needed!

=== EXPECTED TEXT INVENTORY ===
{json.dumps(inventory, ensure_ascii=False, indent=2)}

=== VISUAL SCHEMA SUMMARY ===
{prompt_body[:3500]}

Return ONLY valid JSON (no markdown fences):
{{
  "aesthetic_score_10": 7.2,
  "reference_content_leaked": false,
  "is_ppt_box_form": true,
  "text_or_arrow_errors": ["..."],
  "aesthetic_critique": "<2 sharp sentences explaining what looks great vs what looks ugly>",
  "revision_mode": "pass | targeted_edit | regenerate_clean",
  "fix_instructions": "<concrete surgical instructions for the next image generation/edit round>"
}}
"""

    contents: list[Any] = [
        audit_prompt,
        "Image 1 — Newly Generated Stage-1 Draft Figure (`stage1_draft.png`):",
        draft_path,
    ]
    for idx, rpath in enumerate(reference_paths[:2], 2):
        contents.extend(
            [
                f"Image {idx} — Style-Only Reference ({rpath.name}) [CHECK FOR SUBJECT/PORTRAIT/TEXT LEAK!]:",
                rpath,
            ]
        )

    raw = client.generate_multimodal(contents, max_output_tokens=4000, temperature=0.15)
    parsed = _parse_gate_json(raw)
    score = float(parsed.get("aesthetic_score_10") or 7.0)
    leaked = bool(parsed.get("reference_content_leaked", False))
    ppt_boxy = bool(parsed.get("is_ppt_box_form", False))
    errs = [str(x) for x in (parsed.get("text_or_arrow_errors") or []) if str(x).strip()]
    mode = str(parsed.get("revision_mode") or "").strip().lower()

    # Deterministic gate enforcement (never let a leaked or PPT-boxy draft silently pass!)
    if leaked or ppt_boxy or score < 7.8:
        mode = "regenerate_clean"
    elif errs or score < 8.5:
        if mode != "regenerate_clean":
            mode = "targeted_edit"
    else:
        mode = "pass"

    return {
        "aesthetic_score_10": round(score, 2),
        "reference_content_leaked": leaked,
        "is_ppt_box_form": ppt_boxy,
        "text_or_arrow_errors": errs,
        "aesthetic_critique": str(parsed.get("aesthetic_critique") or "").strip(),
        "revision_mode": mode,
        "fix_instructions": str(parsed.get("fix_instructions") or "").strip(),
        "passed": mode == "pass",
    }


def _build_stage4_render_prompt(
    *,
    prompt_body: str,
    style_dna: dict[str, Any],
    archetype: ArchetypeSpec,
    resolved_density: str,
    has_references: bool,
    extra_revision_notes: str = "",
) -> str:
    aspect_directive = (
        "CANVAS PROPORTION: Sleek **WIDE PANORAMIC (~2.3:1) HORIZONTAL BANNER** "
        "for top-of-page two-column publication placement on a crisp pure `#FFFFFF` white background with generous outer margins."
        if archetype.is_wide_banner
        else "CANVAS PROPORTION: Balanced ~1.6:1 worked-example scientific illustration on a crisp pure `#FFFFFF` background."
    )

    ref_clause = (
        "STYLE-ONLY REFERENCE IMAGE ATTACHED:\n"
        "- Study ONLY its visual sophistication: open whitespace rhythm, clean geometric/3D/illustrative polish, color harmony, line weight, and arrow elegance.\n"
        "- **STRICT IDENTITY & CONTENT ISOLATION**: NEVER copy ANY human face, hair color, pose, character, photo subject, text label, or module from the reference image! Invent a 100% original, natural-looking subject for our paper."
        if has_references
        else "Create a breathtaking, open-canvas flagship CVPR / NeurIPS / Nature scientific illustration on pure `#FFFFFF` white background."
    )

    if resolved_density == "detailed":
        craft_mandate = (
            "1. **CRISP SYSTEMS ARCHITECTURE CLARITY (`detailed` tier)**: Follow the exact flat pastel multi-panel grid and uniform pure white gutters in the Visual Schema; crisp dark borders, bold legible sans-serif typography, clean domain micro-visuals.\n"
            "2. **ZERO EXTRA/INVENTED TEXT**: Render ONLY the strings listed in `[EXACT TEXT INVENTORY]`."
        )
    else:
        craft_mandate = (
            "1. **OPEN-CANVAS PUBLICATION MINIMALISM (40%–50% PURE `#FFFFFF` WHITESPACE!)**:\n"
            "   - Let concrete scientific visuals (portraits/specimens, 3D/isometric optical or hardware setups, sleek trapezoid/layered transformer blocks with ❄/🔥 icons, smooth signal curves, crisp directional arrows) breathe openly on pure white canvas!\n"
            "   - **STRICTLY FORBIDDEN**: NO uniform row of tall gray rounded boxes wrapping every stage, NO red/yellow colored card header strips, NO giant wasted top `Training` bar, NO clusters of oval UI pill buttons, and NO ugly crude Excel bar charts!\n"
            "2. **BIG, CRISP VISUAL HEROES & SHARP TYPOGRAPHY**:\n"
            "   - Visual objects carry ~60% of the visual weight; render ONLY the concise strings in `[EXACT TEXT INVENTORY]` in clean bold academic sans-serif (Inter / Helvetica / Arial), effortlessly readable at 7-inch two-column print width."
        )

    revision_block = (
        f"\n=== MANDATORY REVISION DIRECTIVES FROM ART-DIRECTOR GATE ===\n{extra_revision_notes}\n"
        if extra_revision_notes
        else ""
    )

    return f"""Create a **flagship, publication-grade top-conference scientific overview illustration** (CVPR / NeurIPS / SIGGRAPH / Nature craft — raster figure with crisp vector-like clarity).

{aspect_directive}

{ref_clause}

KEY VISUAL EXCELLENCE MANDATES:
{craft_mandate}
3. **ZERO META-LABEL LEAKAGE**: Render ONLY the exact labels listed in `[EXACT TEXT INVENTORY]`. NEVER render blueprint headings ("ZONE 1", "STYLE", "Framing", "EXACT TEXT INVENTORY").
{revision_block}
=== VISUAL SCHEMA TO RENDER ===
{prompt_body}
"""


def generate_stage1_draft(
    client: Any,
    schema_md: str,
    style_dna: dict[str, Any],
    archetype: ArchetypeSpec,
    output_dir: Path,
    *,
    density: str = "auto",
    review_rounds: int = 2,
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[Path, tuple[int, int], dict[str, Any]]:
    """Generate a high-craft Stage-1 visual anchor guarded by the AIA Stage-4 Aesthetic & Anti-Copy Gate."""
    prompt_body = extract_prompt_body(schema_md)
    inventory = extract_exact_text_inventory(schema_md)
    resolved_density = resolve_density_mode(density, len(inventory))
    anchor_paths = [Path(p) for p in style_dna.get("anchor_files", []) if Path(p).is_file()]

    best_score = -1.0
    best_clean_score = -1.0
    best_img: Optional[Image.Image] = None
    best_raw_img: Optional[Image.Image] = None
    best_meta: dict[str, Any] = {}
    gate_history: list[dict[str, Any]] = []

    current_refs: list[Path] = list(anchor_paths)
    edit_target_path: Optional[Path] = None
    extra_notes = ""
    total_attempts = 1 + max(0, int(review_rounds))

    for attempt in range(1, total_attempts + 1):
        if attempt == 1:
            render_prompt = _build_stage4_render_prompt(
                prompt_body=prompt_body,
                style_dna=style_dna,
                archetype=archetype,
                resolved_density=resolved_density,
                has_references=bool(current_refs),
            )
            (output_dir / "stage4_render_prompt.txt").write_text(render_prompt, encoding="utf-8")
        elif edit_target_path is not None:
            render_prompt = (
                "Surgically refine the attached `[EDIT TARGET FIGURE]` to fix ONLY the specific flaws listed below.\n"
                "Keep the clean open-white composition, visual style, color palette, and all already-correct artwork & labels intact.\n\n"
                f"=== SURGICAL FIXES REQUIRED ===\n{extra_notes}\n\n"
                f"=== AUTHORITATIVE EXACT TEXT INVENTORY ===\n{json.dumps(inventory, ensure_ascii=False, indent=2)}\n"
            )
            (output_dir / f"stage4_edit_prompt_{attempt - 1}.txt").write_text(render_prompt, encoding="utf-8")
        else:
            render_prompt = _build_stage4_render_prompt(
                prompt_body=prompt_body,
                style_dna=style_dna,
                archetype=archetype,
                resolved_density=resolved_density,
                has_references=bool(current_refs),
                extra_revision_notes=extra_notes,
            )
            (output_dir / f"stage4_regen_prompt_{attempt - 1}.txt").write_text(render_prompt, encoding="utf-8")

        if on_progress:
            mode_tag = "initial render" if attempt == 1 else ("targeted edit" if edit_target_path else "clean redesign")
            on_progress(f"Stage 4/6 (Attempt {attempt}/{total_attempts} · {mode_tag}): Rendering visual anchor...")

        raw_img, meta = client.generate_image(
            render_prompt,
            reference_images=current_refs if edit_target_path is None else None,
            edit_target_image=edit_target_path,
            quality=client.settings.image_quality,
            size=client.settings.image_size,
        )
        trimmed_img = smart_trim_white_margins(raw_img)

        ver_path = output_dir / f"stage1_draft_v{attempt}.png"
        trimmed_img.save(ver_path, format="PNG")

        if review_rounds <= 0:
            best_img = trimmed_img
            best_raw_img = raw_img
            best_meta = meta
            break

        if on_progress:
            on_progress(f"Stage 4/6 (Attempt {attempt}/{total_attempts}): Running AIA Aesthetic & Anti-Copy Gate...")

        audit = audit_stage1_draft_aesthetics(
            client,
            ver_path,
            schema_md,
            reference_paths=anchor_paths,
            resolved_density=resolved_density,
        )
        audit["attempt"] = attempt
        audit["file"] = ver_path.name
        gate_history.append(audit)

        score = float(audit.get("aesthetic_score_10") or 0.0)
        is_clean = (not audit["reference_content_leaked"]) and (not audit["is_ppt_box_form"])
        # Strongly prefer clean (non-leaked, non-PPT-boxy) candidates over higher-scoring flawed ones
        if is_clean and score >= best_clean_score:
            best_clean_score = score
            best_score = score
            best_img = trimmed_img
            best_raw_img = raw_img
            best_meta = meta
        elif best_clean_score < 0 and score >= best_score:
            best_score = score
            best_img = trimmed_img
            best_raw_img = raw_img
            best_meta = meta

        if on_progress:
            on_progress(
                f"Stage 4 Gate v{attempt}: score={score}/10 | leaked={audit['reference_content_leaked']} | "
                f"ppt_boxy={audit['is_ppt_box_form']} | errors={len(audit['text_or_arrow_errors'])} -> {audit['revision_mode']}"
            )

        if audit["passed"] or attempt >= total_attempts:
            break

        # Prepare next surgical attempt
        fix_lines = []
        if audit["reference_content_leaked"]:
            # Drop the contaminating reference image completely on clean redesign!
            current_refs = []
            fix_lines.append(
                "- CRITICAL: The previous attempt copied a face/subject/element from the reference image! "
                "Generate a 100% ORIGINAL ordinary natural human subject/visual with ZERO resemblance to the reference!"
            )
        if audit["is_ppt_box_form"]:
            # If the reference itself nudged boxy cards, drop reference and rely on clean open-white Style DNA
            if resolved_density != "detailed":
                current_refs = []
            fix_lines.append(
                "- CRITICAL ANTI-PPT REDESIGN: Eliminate ALL uniform tall gray outer boxes, colored top header strips, "
                "wasted full-width top Training bars, oval pill buttons, and crude Excel bar charts! "
                "Compose an AIRY OPEN-WHITE CVPR/Nature illustration with sleek 3D/isometric optical blocks, "
                "trapezoid/layered neural blocks (with ❄/🔥 badges), clean progressive visual frames, and 45% pure `#FFFFFF` breathing space!"
            )
        if audit["text_or_arrow_errors"]:
            fix_lines.append("- Fix these exact text/arrow errors: " + "; ".join(audit["text_or_arrow_errors"]))
        if audit["fix_instructions"]:
            fix_lines.append(f"- Principal Art Director directive: {audit['fix_instructions']}")

        extra_notes = "\n".join(fix_lines)
        if audit["revision_mode"] == "targeted_edit" and is_clean:
            edit_target_path = ver_path
        else:
            edit_target_path = None

    assert best_img is not None and best_raw_img is not None

    raw_path = output_dir / "stage1_draft_raw.png"
    best_raw_img.save(raw_path, format="PNG")

    draft_path = output_dir / "stage1_draft.png"
    best_img.save(draft_path, format="PNG")

    (output_dir / "stage4_gate_report.json").write_text(
        json.dumps(gate_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    viewbox = compute_svg_viewbox(best_img, archetype)
    best_meta["raw_size"] = list(best_raw_img.size)
    best_meta["trimmed_size"] = list(best_img.size)
    best_meta["svg_viewbox"] = list(viewbox)
    best_meta["gate_attempts"] = len(gate_history)
    best_meta["gate_final_score"] = gate_history[-1]["aesthetic_score_10"] if gate_history else None
    best_meta["gate_passed"] = gate_history[-1]["passed"] if gate_history else True
    return draft_path, viewbox, best_meta
