"""End-to-end Two-Stage Academic Paper -> Editable Hybrid SVG Pipeline with Stage Resume Support."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from .archetypes import ArchetypeSpec, get_archetype
from .architect import generate_visual_schema
from .asset_extractor import ExtractedAsset, load_cached_assets
from .client import ResponsesAPIClient
from .config import Settings, sanitize_font_family
from .draft_generator import compute_svg_viewbox, generate_stage1_draft
from .pdf_parser import (
    analyze_paper_structure,
    extract_pdf_sections,
    render_paper_method_markdown,
)
from .style_extractor import (
    discover_reference_images,
    extract_style_dna,
    render_style_guide_markdown,
)
from .svg_builder import build_editable_svg
from .svg_critic import refine_svg_with_critic


@dataclass
class PipelineResult:
    output_dir: Path
    final_svg: Path
    final_png: Path
    print_preview_2col: Path
    qa_report: Path
    stage1_draft: Path
    schema_md: Path
    style_guide_md: Path
    paper_method_md: Path
    manifest_path: Path
    archetype: str
    viewbox: tuple[int, int]
    elapsed_seconds: float
    embedded_assets_count: int = 0
    round_logs: list[dict[str, Any]] = field(default_factory=list)


def run_pipeline(
    *,
    paper_pdf: Path,
    refs_dir: Optional[Path],
    output_dir: Path,
    settings: Settings,
    lang: str = "auto",
    archetype_override: Optional[str] = None,
    font_override: Optional[str] = None,
    svg_mode_override: Optional[str] = None,
    critic_rounds: Optional[int] = None,
    from_stage: int = 1,
    stop_after_stage: Optional[int] = None,
    on_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> PipelineResult:
    """Execute the full quality-first Two-Stage Academic Overview Editable SVG pipeline.

    Stages:
      1: Section-aware PDF parsing & contribution/novelty extraction
      2: Visual Art-Director Style DNA extraction (true reference fidelity, anti-flowchart)
      3: Visual-First Story Schema Architect (6 Flexible Archetypes + Hero Visual Assets)
      4: Stage-1 High-Craft Visual Anchor (Responses image_generation) + Smart Wide-Banner Trim
      5: Zero-GPU Visual Asset Cropping + Hybrid/Pure Editable SVG Reconstruction + XML Auto-Repair
      6: Deterministic Linter + Dual-Resolution Vision Critic Multi-Round Refinement
    """
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    def emit(stage_msg: str, extra: Optional[dict[str, Any]] = None) -> None:
        if on_event:
            payload = {"message": stage_msg, "ts": datetime.now().isoformat(timespec="seconds")}
            if extra:
                payload.update(extra)
            on_event("progress", payload)

    client = ResponsesAPIClient(settings)
    rounds = critic_rounds if critic_rounds is not None else settings.critic_rounds
    svg_mode = (svg_mode_override or settings.svg_mode or "hybrid_rich").strip().lower()

    paper_analysis_path = output_dir / "paper_analysis.json"
    paper_method_path = output_dir / "paper_method.md"
    style_dna_path = output_dir / "style_dna.json"
    style_guide_path = output_dir / "style_guide.md"
    schema_path = output_dir / "schema.md"
    draft_path = output_dir / "stage1_draft.png"
    raw_svg_path = output_dir / "raw.svg"
    final_svg_path = output_dir / "final.svg"
    final_png_path = output_dir / "final.png"
    twocol_png_path = output_dir / "print_preview_2col.png"
    qa_path = output_dir / "qa.md"
    manifest_path = output_dir / "run_manifest.json"

    # ---- STAGE 1: PDF Parsing & Grounded Contribution Extraction ----
    if from_stage <= 1 or not paper_analysis_path.is_file():
        emit(
            f"Stage 1/6: Parsing full paper PDF via {settings.main_model} (effort={settings.reasoning_effort})...",
            {"stage": 1},
        )
        pdf_info = extract_pdf_sections(paper_pdf)
        resolved_lang = pdf_info["detected_lang"] if (lang or settings.lang) == "auto" else (lang or settings.lang)
        paper_analysis = analyze_paper_structure(client, pdf_info, target_lang=resolved_lang)
        paper_analysis["_resolved_lang"] = resolved_lang
        paper_analysis["_pdf_pages"] = pdf_info["num_pages"]
        paper_analysis_path.write_text(json.dumps(paper_analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        paper_method_path.write_text(render_paper_method_markdown(paper_analysis, pdf_info), encoding="utf-8")
        emit(
            f"Stage 1/6 complete: Archetype=`{paper_analysis.get('recommended_archetype')}`, "
            f"Modules={len(paper_analysis.get('modules', []))}, Lang=`{resolved_lang}`",
            {"stage": 1, "artifact": "paper_method.md"},
        )
    else:
        paper_analysis = json.loads(paper_analysis_path.read_text(encoding="utf-8"))
        resolved_lang = (
            paper_analysis.get("_resolved_lang", "en") if (lang or settings.lang) == "auto" else (lang or settings.lang)
        )
        emit(f"Stage 1/6 skipped (reusing cached paper_analysis.json, lang=`{resolved_lang}`)", {"stage": 1})

    chosen_archetype_key = archetype_override or paper_analysis.get("recommended_archetype", "pipeline")
    archetype: ArchetypeSpec = get_archetype(chosen_archetype_key)
    font_family = sanitize_font_family(font_override or settings.font_family, lang=resolved_lang)

    # ---- STAGE 2: Visual Art-Director Style DNA Extraction ----
    if from_stage <= 2 or not style_dna_path.is_file():
        ref_images = discover_reference_images(refs_dir) if refs_dir else []
        emit(
            f"Stage 2/6: Extracting true Visual Art Direction from {len(ref_images)} reference image(s)...",
            {"stage": 2},
        )
        style_dna = extract_style_dna(client, ref_images, target_archetype=archetype.key)
        style_dna_path.write_text(json.dumps(style_dna, ensure_ascii=False, indent=2), encoding="utf-8")
        style_guide_path.write_text(render_style_guide_markdown(style_dna), encoding="utf-8")
        emit(
            f"Stage 2/6 complete: Paradigm=`{style_dna.get('visual_paradigm')}`, "
            f"Containers=`{style_dna.get('container_philosophy', {}).get('style')}`, "
            f"Anchors={len(style_dna.get('anchor_files', []))}",
            {"stage": 2, "artifact": "style_guide.md"},
        )
    else:
        style_dna = json.loads(style_dna_path.read_text(encoding="utf-8"))
        emit("Stage 2/6 skipped (reusing cached style_dna.json)", {"stage": 2})

    # ---- STAGE 3: Visual-First Story Schema Architect ----
    if from_stage <= 3 or not schema_path.is_file():
        emit(
            f"Stage 3/6: Designing Visual-First Schema ({archetype.key} / {style_dna.get('visual_paradigm')})...",
            {"stage": 3},
        )
        schema_md, archetype = generate_visual_schema(
            client,
            paper_analysis,
            style_dna,
            font_family=font_family,
            target_lang=resolved_lang,
            archetype_override=chosen_archetype_key,
        )
        schema_path.write_text(schema_md, encoding="utf-8")
        emit("Stage 3/6 complete: Visual Schema blueprint saved to schema.md.", {"stage": 3, "artifact": "schema.md"})
    else:
        schema_md = schema_path.read_text(encoding="utf-8")
        emit("Stage 3/6 skipped (reusing edited/cached schema.md)", {"stage": 3})

    if stop_after_stage == 3:
        elapsed = round(time.time() - t0, 2)
        _write_manifest(
            manifest_path,
            paper_pdf=paper_pdf,
            refs_dir=refs_dir,
            settings=settings,
            archetype=archetype,
            viewbox=(archetype.canvas_width, archetype.canvas_height),
            lang=resolved_lang,
            font_family=font_family,
            svg_mode=svg_mode,
            embedded_assets_count=0,
            status="paused_after_schema",
            elapsed=elapsed,
        )
        return PipelineResult(
            output_dir=output_dir,
            final_svg=final_svg_path,
            final_png=final_png_path,
            print_preview_2col=twocol_png_path,
            qa_report=qa_path,
            stage1_draft=draft_path,
            schema_md=schema_path,
            style_guide_md=style_guide_path,
            paper_method_md=paper_method_path,
            manifest_path=manifest_path,
            archetype=archetype.key,
            viewbox=(archetype.canvas_width, archetype.canvas_height),
            elapsed_seconds=elapsed,
        )

    # ---- STAGE 4: Stage-1 High-Craft Visual Anchor ----
    if from_stage <= 4 or not draft_path.is_file():
        emit(
            f"Stage 4/6: Generating Stage-1 visual anchor via {settings.image_tool_model or 'built-in image_generation'} "
            f"(quality={settings.image_quality})...",
            {"stage": 4},
        )
        draft_path, viewbox, draft_meta = generate_stage1_draft(
            client,
            schema_md,
            style_dna,
            archetype,
            output_dir,
        )
        (output_dir / "stage4_meta.json").write_text(json.dumps(draft_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        emit(
            f"Stage 4/6 complete: Anchor saved ({draft_meta.get('trimmed_size')}) -> SVG viewBox {viewbox[0]}x{viewbox[1]}.",
            {"stage": 4, "artifact": "stage1_draft.png"},
        )
    else:
        with Image.open(draft_path) as d_img:
            viewbox = compute_svg_viewbox(d_img, archetype)
        emit(f"Stage 4/6 skipped (reusing stage1_draft.png -> viewBox {viewbox[0]}x{viewbox[1]})", {"stage": 4})

    # ---- STAGE 5: Visual Asset Cropping + Hybrid/Pure Editable SVG Reconstruction ----
    assets: list[ExtractedAsset] = []
    if from_stage <= 5 or not raw_svg_path.is_file():
        mode_desc = "Hybrid Editable SVG (vectors + cropped HD visuals)" if svg_mode == "hybrid_rich" else "Pure Vector SVG"
        emit(
            f"Stage 5/6: Building {mode_desc} ({viewbox[0]}x{viewbox[1]}) via {settings.main_model}...",
            {"stage": 5},
        )
        raw_svg, assets, xml_ok, xml_errs = build_editable_svg(
            client,
            draft_path,
            schema_md,
            style_dna,
            archetype,
            viewbox,
            output_dir,
            font_family=font_family,
            svg_mode=svg_mode,
        )
        raw_svg_path.write_text(raw_svg, encoding="utf-8")
        emit(
            f"Stage 5/6 complete: raw.svg generated ({len(raw_svg):,} chars, assets={len(assets)}, xml_valid={xml_ok}).",
            {"stage": 5, "artifact": "raw.svg", "assets_count": len(assets), "xml_errors": xml_errs},
        )
    else:
        raw_svg = raw_svg_path.read_text(encoding="utf-8")
        assets = load_cached_assets(output_dir) if svg_mode == "hybrid_rich" else []
        emit(f"Stage 5/6 skipped (reusing raw.svg + {len(assets)} cached visual asset(s))", {"stage": 5})

    # ---- STAGE 6: Deterministic Linter + Dual-Resolution Vision Critic Refinement ----
    emit(f"Stage 6/6: Starting {rounds}-round Linter + Dual-Resolution Vision Critic refinement...", {"stage": 6})
    final_svg, final_png_path, twocol_png_path, qa_md, round_logs = refine_svg_with_critic(
        client,
        raw_svg,
        draft_path,
        schema_md,
        viewbox,
        output_dir,
        assets=assets,
        max_rounds=rounds,
        on_progress=lambda msg: emit(msg, {"stage": 6}),
    )

    elapsed = round(time.time() - t0, 2)
    _write_manifest(
        manifest_path,
        paper_pdf=paper_pdf,
        refs_dir=refs_dir,
        settings=settings,
        archetype=archetype,
        viewbox=viewbox,
        lang=resolved_lang,
        font_family=font_family,
        svg_mode=svg_mode,
        embedded_assets_count=len(assets),
        status="completed",
        elapsed=elapsed,
        round_logs=round_logs,
    )
    emit(
        f"All 6 stages complete in {elapsed:.1f}s! Final editable SVG ready ({len(assets)} HD assets embedded).",
        {"stage": 6, "done": True, "artifact": "final.svg"},
    )

    return PipelineResult(
        output_dir=output_dir,
        final_svg=final_svg_path,
        final_png=final_png_path,
        print_preview_2col=twocol_png_path,
        qa_report=qa_path,
        stage1_draft=draft_path,
        schema_md=schema_path,
        style_guide_md=style_guide_path,
        paper_method_md=paper_method_path,
        manifest_path=manifest_path,
        archetype=archetype.key,
        viewbox=viewbox,
        elapsed_seconds=elapsed,
        embedded_assets_count=len(assets),
        round_logs=round_logs,
    )


def _write_manifest(
    manifest_path: Path,
    *,
    paper_pdf: Path,
    refs_dir: Optional[Path],
    settings: Settings,
    archetype: ArchetypeSpec,
    viewbox: tuple[int, int],
    lang: str,
    font_family: str,
    svg_mode: str,
    embedded_assets_count: int,
    status: str,
    elapsed: float,
    round_logs: Optional[list[dict[str, Any]]] = None,
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "paper_pdf": str(paper_pdf),
        "refs_dir": str(refs_dir) if refs_dir else None,
        "main_model": settings.main_model,
        "reasoning_effort": settings.reasoning_effort,
        "image_tool_model": settings.image_tool_model,
        "image_quality": settings.image_quality,
        "svg_mode": svg_mode,
        "embedded_assets_count": embedded_assets_count,
        "archetype": archetype.key,
        "archetype_name": f"{archetype.zh_name} / {archetype.en_name}",
        "viewbox": list(viewbox),
        "aspect_ratio": round(viewbox[0] / float(viewbox[1]), 2),
        "lang": lang,
        "font_family": font_family,
        "elapsed_seconds": elapsed,
        "round_logs": round_logs or [],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
