"""Stage 6: Dual-Resolution Vision Critic + Deterministic Linter Multi-Round Refinement (Hybrid-Asset Safe)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from .architect import extract_exact_text_inventory, extract_prompt_body
from .asset_extractor import (
    ExtractedAsset,
    dehydrate_svg_assets,
    hydrate_svg_assets,
    verify_assets_preserved,
)
from .config import resolve_density_mode
from .svg_builder import extract_svg_block, fix_svg_xml_with_llm, normalize_svg_root
from .svg_linter import LintReport, lint_svg, validate_svg_xml
from .svg_renderer import render_dual_preview


def check_critic_content_retention(
    lint_before: LintReport,
    lint_after: LintReport,
    inventory: list[str],
) -> tuple[bool, str]:
    """Verify Critic refinement did not catastrophically delete diagram text content.

    Uses total rendered characters + authoritative `[EXACT TEXT INVENTORY]` coverage ratio
    instead of raw `<text>` tag count, so consolidating multi-line `<text>` elements into
    `<tspan>` groups NEVER triggers a false rollback!
    """
    prev_chars = int(lint_before.stats.get("total_text_chars", 0) or 0)
    new_chars = int(lint_after.stats.get("total_text_chars", 0) or 0)
    prev_cov = float(lint_before.stats.get("inventory_coverage_ratio", 1.0) or 0.0)
    new_cov = float(lint_after.stats.get("inventory_coverage_ratio", 1.0) or 0.0)
    inv_total = int(lint_after.stats.get("inventory_total", 0) or 0)

    if inventory and inv_total >= 4 and prev_cov >= 0.45:
        min_allowed_cov = max(0.50, prev_cov - 0.25)
        if new_cov < min_allowed_cov:
            return (
                False,
                f"Patched SVG dropped too many required inventory labels "
                f"(coverage {prev_cov * 100:.0f}% -> {new_cov * 100:.0f}%); keeping previous SVG.",
            )

    if prev_chars >= 40 and new_chars < prev_chars * 0.55:
        return (
            False,
            f"Patched SVG dropped too much rendered text content "
            f"({prev_chars} -> {new_chars} chars); keeping previous SVG.",
        )

    return True, ""


def _parse_critique_json(raw_text: str) -> dict[str, Any]:
    """Extract optional JSON critique header placed before `<svg`."""
    svg_idx = raw_text.lower().find("<svg")
    prefix = raw_text[:svg_idx] if svg_idx != -1 else raw_text
    start, end = prefix.find("{"), prefix.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(prefix[start : end + 1])
        except Exception:
            pass
    return {}


def refine_svg_with_critic(
    client: Any,
    initial_svg: str,
    draft_path: Path,
    schema_md: str,
    viewbox: tuple[int, int],
    output_dir: Path,
    *,
    assets: Optional[list[ExtractedAsset]] = None,
    max_rounds: int = 2,
    density: str = "auto",
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[str, Path, Path, str, list[dict[str, Any]]]:
    """Run iterative Deterministic Linter + Dual-Resolution Vision Critic refinement loop.

    Protects embedded high-res visual assets via automatic placeholder dehydration/re-hydration
    so Critic never wastes tokens on base64 strings or accidentally deletes cropped visuals.
    """
    w, h = viewbox
    active_assets = assets or []
    inventory = extract_exact_text_inventory(schema_md)
    prompt_body = extract_prompt_body(schema_md)
    resolved_density = resolve_density_mode(density, len(inventory))

    current_svg = initial_svg
    round_logs: list[dict[str, Any]] = []

    asset_guard_clause = (
        f"- **PROTECT ALL {len(active_assets)} EMBEDDED VISUAL ASSETS**: Keep EVERY `href=\"__ASSET_...__\"` `<image>` "
        "element intact! You may fine-tune their `(x, y, width, height)` or `<clipPath>` alignment so they fit cleanly "
        "without colliding with text or arrows, but NEVER delete or replace them with crude hand-drawn polygons."
        if active_assets
        else "- Keep all visual motifs sharp, cohesive, and publication-grade."
    )

    for r in range(1, max_rounds + 1):
        if on_progress:
            on_progress(f"Stage 6 Critic Round {r}/{max_rounds}: Running deterministic lint + dual render...")

        iter_svg_path = output_dir / f"iter_{r - 1}.svg"
        iter_svg_path.write_text(current_svg, encoding="utf-8")

        full_png = output_dir / f"iter_{r - 1}_full.png"
        twocol_png = output_dir / f"iter_{r - 1}_2col.png"
        renderer_used = render_dual_preview(current_svg, full_png, twocol_png, viewbox=viewbox)
        lint_before: LintReport = lint_svg(
            current_svg,
            exact_text_inventory=inventory,
            expected_viewbox=viewbox,
            rendered_png_path=full_png,
            density=resolved_density,
        )

        # Dehydrate base64 strings -> compact __ASSET_<id>__ placeholders before sending SVG code to LLM
        compact_svg = dehydrate_svg_assets(current_svg, active_assets) if active_assets else current_svg

        critic_prompt = f"""You are a Brutally Honest Principal Art Director (IEEE S&P / SIGGRAPH / CVPR / Nature) performing **Visual Fidelity Refinement Round {r}/{max_rounds}** on an editable academic SVG (`viewBox="0 0 {w} {h}"`, `density={resolved_density}`).

YOUR NORTH STAR — **MAKE IMAGE 2 LOOK JUST AS GORGEOUS AS IMAGE 1 (`stage1_draft.png`)!**
DO NOT rubber-stamp an ugly figure with a fake 9.5/10 score! Compare **Image 2 (`iter_{r - 1}_full.png`, our current SVG with all {len(active_assets)} embedded HD icons)** side-by-side against **Image 1 (`stage1_draft.png`, the visual gold standard)** and **Image 3 (700px 2-column print preview)**!

=== PROGRAMMATIC AUDIT REPORT (FIX EVERY HARD ERROR FIRST!) ===
{lint_before.to_prompt_feedback()}

=== STRICT SIDE-BY-SIDE VISUAL FIDELITY AUDIT (CHECK EVERY SINGLE PANEL!) ===
Scan Image 2 vs Image 1 left-to-right, top-to-bottom and fix EVERY visual flaw:
1. **PER-ARROW DIRECTION, COMPLETENESS & BOLDNESS AUDIT (CRITICAL!)**:
   - Trace EVERY SINGLE inter-panel block arrow in Image 2 ONE BY ONE against Image 1 AND `[FLOW CONNECTIONS]`:
     * For EACH vertical arrow (left vertical arrow, right vertical arrow, combine arrow) and horizontal arrow (top row arrows, bottom row return/output arrows), verify the exact arrowhead tip direction (`UP` vs `DOWN`, `LEFT` vs `RIGHT`)! If ANY arrow in Image 2 points the opposite direction of Image 1 (e.g., pointing UP when Image 1 points DOWN!), **FLIP IT IMMEDIATELY**!
   - Is ANY arrow present in Image 1 MISSING in Image 2, skinnier/crooked, or disconnected from panel borders? Redraw it bold, symmetric, cleanly spanning the gutter!
   - Does ANY intra-panel arrow pierce into an icon, collide with text, or float awkwardly? Fix its exact start/end coordinates!
2. **TYPOGRAPHY BOLDNESS, CONTRAST & ZERO COLLISIONS**:
   - Match Image 1's **bold, crisp dark `#111111` ink weight** — NEVER leave headings or sub-labels pale grey or timid/tiny when Image 1 makes them bold black!
   - Check EVERY text label in Image 2: does ANY label touch/overlap an icon, waveform, border, arrow, or neighboring card? Move or wrap onto 2 clean `<tspan>` lines!
   - NEVER use `textLength` or `lengthAdjust="spacingAndGlyphs"` horizontal squashing, and NEVER delete required `[EXACT TEXT INVENTORY]` labels!
3. **EMBEDDED HD ASSET ALIGNMENT & ZERO DUPLICATE DOODLES**:
   - Inspect all {len(active_assets)} embedded `<image href="__ASSET_...__"/>` icons in Image 2: ensure ZERO ugly hand-drawn SVG lines sit behind/over them, and fine-tune `(x, y, width, height)` by a few pixels ONLY if needed to align cleanly with arrows and captions.
4. **PANEL & CARD GEOMETRY / FLAT COLOR FIDELITY**:
   - Verify all top-row panels share identical top & bottom Y, all bottom-row panels share identical top & bottom Y, uniform white gutters, crisp uniform borders, and pure flat fills matching Image 1 (remove any unwanted gradients/shadows if Image 1 is flat 2D!).
5. **CANVAS SAFE-ZONE & TWO-COLUMN LEGIBILITY (Image 3)**:
   - Keep EVERY element strictly inside `[10, {w - 10}] x [10, {h - 10}]` and effortlessly readable in Image 3 (700px wide).
6. **HONEST SCORING RULE**:
   - Give `< 8.0` if ANY arrow is missing/skinny/misaligned, ANY label collides with an icon/border, or typography looks noticeably weaker than Image 1. Only give `>= 9.3` when Image 2 truly rivals Image 1 in visual polish!

=== SURGICAL PATCHING & ASSET SAFETY RULES ===
{asset_guard_clause}
- Keep exact `viewBox="0 0 {w} {h}" width="{w}" height="{h}"` and clean 5-layer `<g>` grouping.

=== AUTHORITATIVE VISUAL SCHEMA & TEXT INVENTORY ===
{prompt_body}

=== CURRENT SVG SOURCE ===
```xml
{compact_svg}
```

OUTPUT FORMAT:
First output a brief, brutally honest JSON critique block (between `{{` and `}}`) that EXPLICITLY verifies every inter-panel arrow's direction in Image 2 vs Image 1:
`{{"score_10": 8.2, "two_col_readable": true, "arrow_direction_audit": [{{"connector": "Panel A -> Panel B", "expected": "DOWN", "seen_in_iter": "UP", "fixed": true}}], "issues_found": ["..."], "fixes_applied": ["..."]}}`
Immediately followed by the COMPLETE improved SVG starting with `<svg` and ending with `</svg>` (NO markdown fences around the SVG).
"""

        if on_progress:
            on_progress(f"Stage 6 Critic Round {r}/{max_rounds}: Consulting {client.settings.main_model} vision critic...")

        try:
            raw_reply = client.generate_multimodal(
                [
                    critic_prompt,
                    "Image 1 — Stage-1 Art-Direction Anchor (`stage1_draft.png`):",
                    draft_path,
                    f"Image 2 — Current SVG Full-Res Render (`iter_{r - 1}_full.png`):",
                    full_png,
                    f"Image 3 — Current SVG Simulated 2-Column Print Preview (`iter_{r - 1}_2col.png`):",
                    twocol_png,
                ],
                max_output_tokens=64000,
                temperature=0.15,
            )
        except Exception as exc:
            err_msg = f"Critic Round {r} interrupted by upstream API error ({exc}); preserving best verified SVG."
            if on_progress:
                on_progress(f"Stage 6 Critic Round {r}/{max_rounds} warning: {err_msg}")
            round_logs.append(
                {
                    "round": r,
                    "renderer": renderer_used,
                    "lint_errors_before": len(lint_before.errors),
                    "lint_warnings_before": len(lint_before.warnings),
                    "critique": {},
                    "accepted": False,
                    "reject_reason": err_msg,
                }
            )
            break

        critique_meta = _parse_critique_json(raw_reply)
        candidate_template = normalize_svg_root(extract_svg_block(raw_reply), w, h)

        accepted = False
        reject_reason = ""
        if not candidate_template:
            reject_reason = "No <svg> block found in critic reply; keeping previous SVG."
        else:
            ok, errs = validate_svg_xml(candidate_template)
            if not ok:
                candidate_template = fix_svg_xml_with_llm(client, candidate_template, errs, width=w, height=h)
                ok, errs = validate_svg_xml(candidate_template)

            if not ok:
                reject_reason = f"Patched SVG still invalid XML ({errs[:2]}); keeping previous SVG."
            else:
                assets_ok, missing_assets = verify_assets_preserved(candidate_template, active_assets)
                if active_assets and not assets_ok:
                    reject_reason = f"Critic accidentally dropped embedded visual asset(s) {missing_assets}; keeping previous SVG."
                else:
                    hydrated_candidate = (
                        hydrate_svg_assets(candidate_template, active_assets) if active_assets else candidate_template
                    )
                    lint_after = lint_svg(
                        hydrated_candidate,
                        exact_text_inventory=inventory,
                        expected_viewbox=viewbox,
                        density=resolved_density,
                    )
                    retained_ok, retained_msg = check_critic_content_retention(lint_before, lint_after, inventory)
                    if not retained_ok:
                        reject_reason = retained_msg
                    else:
                        current_svg = hydrated_candidate
                        accepted = True

        round_logs.append(
            {
                "round": r,
                "renderer": renderer_used,
                "lint_errors_before": len(lint_before.errors),
                "lint_warnings_before": len(lint_before.warnings),
                "critique": critique_meta,
                "accepted": accepted,
                "reject_reason": reject_reason,
            }
        )

        lint_now = lint_svg(
            current_svg,
            exact_text_inventory=inventory,
            expected_viewbox=viewbox,
            density=resolved_density,
        )
        score = float(critique_meta.get("score_10") or 0.0)
        if accepted and lint_now.passed_hard_checks and score >= 9.5:
            if on_progress:
                on_progress(f"Stage 6 Critic converged early at Round {r} (score={score}/10, 0 lint errors)!")
            break

    final_svg_path = output_dir / "final.svg"
    final_svg_path.write_text(current_svg, encoding="utf-8")

    final_png_path = output_dir / "final.png"
    final_2col_path = output_dir / "print_preview_2col.png"
    final_renderer = render_dual_preview(current_svg, final_png_path, final_2col_path, viewbox=viewbox, full_zoom=2.0)

    final_lint = lint_svg(
        current_svg,
        exact_text_inventory=inventory,
        expected_viewbox=viewbox,
        rendered_png_path=final_png_path,
        density=resolved_density,
    )
    qa_md = _build_qa_report(final_lint, round_logs, final_renderer, viewbox, len(active_assets))
    (output_dir / "qa.md").write_text(qa_md, encoding="utf-8")

    return current_svg, final_png_path, final_2col_path, qa_md, round_logs


def _build_qa_report(
    final_lint: LintReport,
    round_logs: list[dict[str, Any]],
    renderer: str,
    viewbox: tuple[int, int],
    num_assets: int,
) -> str:
    w, h = viewbox
    lines = [
        "# Final SVG Quality Audit Report (`qa.md`)",
        "",
        f"- **Final ViewBox**: `{w} x {h}` (aspect ratio `{w / h:.2f}:1`)",
        f"- **SVG Renderer**: `{renderer}`",
        f"- **Embedded High-Res Visual Assets**: `{num_assets}`",
        f"- **XML Valid**: `{final_lint.valid_xml}`",
        f"- **Hard Linter Errors Remaining**: `{len(final_lint.errors)}`",
        f"- **Linter Warnings Remaining**: `{len(final_lint.warnings)}`",
        f"- **Stats**: `{json.dumps(final_lint.stats, ensure_ascii=False)}`",
        "",
        "## Refinement Rounds",
    ]
    for entry in round_logs:
        c = entry.get("critique") or {}
        lines.append(
            f"### Round {entry['round']} — Accepted: `{entry['accepted']}` | Score: `{c.get('score_10', 'N/A')}/10` "
            f"| Pre-lint errors: `{entry['lint_errors_before']}`"
        )
        if entry.get("reject_reason"):
            lines.append(f"- **Note**: {entry['reject_reason']}")
        if c.get("issues_found"):
            lines.append("- **Issues Found**:")
            for iss in c["issues_found"]:
                lines.append(f"  - {iss}")
        if c.get("fixes_applied"):
            lines.append("- **Fixes Applied**:")
            for fx in c["fixes_applied"]:
                lines.append(f"  - {fx}")
        lines.append("")

    if final_lint.errors:
        lines.append("## Remaining Hard Linter Errors")
        for e in final_lint.errors:
            lines.append(f"- ❌ {e}")
        lines.append("")
    if final_lint.warnings:
        lines.append("## Remaining Linter Warnings")
        for wn in final_lint.warnings:
            lines.append(f"- ⚠️ {wn}")
        lines.append("")

    lines.append("## Deliverables")
    lines.append("- `final.svg` — Editable SVG (vector text/boxes/arrows + lossless draggable visual assets)")
    lines.append("- `final.png` — High-resolution 2x raster preview")
    lines.append("- `print_preview_2col.png` — Simulated 7-inch two-column paper print preview")
    return "\n".join(lines) + "\n"
