"""End-to-end Surgical Giant Pipeline (v2.0): AIA-Academic-Illustrator Front-End + AutoFigure-Edit Back-End."""

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
from .autofigure_engine import prepare_imported_figure, run_autofigure_svg_engine
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
    svg_engine: str = "autofigure"
    seg_backend: str = "auto"
    matting_backend: str = "auto"
    round_logs: list[dict[str, Any]] = field(default_factory=list)


def run_pipeline(
    *,
    paper_pdf: Optional[Path] = None,
    refs_dir: Optional[Path] = None,
    input_figure: Optional[Path] = None,
    output_dir: Path,
    settings: Settings,
    lang: str = "auto",
    archetype_override: Optional[str] = None,
    font_override: Optional[str] = None,
    svg_mode_override: Optional[str] = None,
    svg_engine_override: Optional[str] = None,
    density_override: Optional[str] = None,
    primary_ref: Optional[Path] = None,
    seg_backend_override: Optional[str] = None,
    matting_backend_override: Optional[str] = None,
    draft_review_rounds: Optional[int] = None,
    enable_upscale_4k: Optional[bool] = None,
    critic_rounds: Optional[int] = None,
    from_stage: int = 1,
    stop_after_stage: Optional[int] = None,
    on_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> PipelineResult:
    """Execute the v2.0 3-Cut Surgical Giant pipeline:

    - Front-Half (Stages 1–4, AIA-Academic-Illustrator):
        1. PyMuPDF full-paper section filtering & grounded novelty extraction
        2. Visual Art-Director Style DNA extraction (single Primary Style Winner, zero content copy)
        3. AIA Physical-Object-First Visual Schema (5 layouts, anti-PPT-form guardrails)
        4. High-Craft Visual Anchor Generation + AIA Aesthetic & Anti-Copy Gate (auto-revises ugly/leaked drafts)
           (Optional pause via `stop_after_stage=4`, or bypass Stages 1–4 entirely via `input_figure`!)
    - Back-Half (Stages 5–6, AutoFigure-Edit Native Engine):
        5. Optional 4K upscale + Pluggable SAM3/Zero-GPU `<AF>01` masking + Pluggable RMBG-2.0/BiRefNet/FloodFill matting + clean `<AF>01` Template SVG
        6. Visual Template Alignment Optimization (via Windows zero-DLL `resvg-py`) + Deterministic Icon Replacement -> `final.svg`
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
    review_rounds = draft_review_rounds if draft_review_rounds is not None else settings.draft_review_rounds
    svg_mode = (svg_mode_override or settings.svg_mode or "hybrid_rich").strip().lower()
    svg_engine = (svg_engine_override or settings.svg_engine or "autofigure").strip().lower()
    density = (density_override or settings.density or "auto").strip().lower()
    seg_backend = (seg_backend_override or settings.seg_backend or "auto").strip().lower()
    matting_backend = (matting_backend_override or settings.matting_backend or "auto").strip().lower()
    upscale_4k = settings.enable_upscale_4k if enable_upscale_4k is None else bool(enable_upscale_4k)

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

    # ---- DIRECT IMPORTED FIGURE FAST-PATH (AutoFigure-Edit `--input-figure` mode) ----
    if input_figure is not None:
        emit(
            f"Direct Figure Mode: Importing `{input_figure.name}` straight into AutoFigure-Edit SVG engine...",
            {"stage": 4},
        )
        draft_path, viewbox = prepare_imported_figure(input_figure, draft_path, enable_upscale=False)
        archetype = get_archetype(archetype_override or "pipeline")
        resolved_lang = "en" if (lang or settings.lang) == "auto" else (lang or settings.lang)
        font_family = sanitize_font_family(font_override or settings.font_family, lang=resolved_lang)
        schema_md = schema_path.read_text(encoding="utf-8") if schema_path.is_file() else ""
        af_res = run_autofigure_svg_engine(
            client,
            draft_path,
            output_dir,
            schema_md=schema_md,
            font_family=font_family,
            svg_mode=svg_mode,
            density=density,
            seg_backend=seg_backend,
            matting_backend=matting_backend,
            sam_prompts=settings.sam_prompts,
            sam_min_score=settings.sam_min_score,
            sam_merge_threshold=settings.sam_merge_threshold,
            sam_api_key=settings.sam_api_key,
            sam_max_masks=settings.sam_max_masks,
            rmbg_model_path=settings.rmbg_model_path,
            enable_upscale_4k=upscale_4k,
            optimize_iterations=rounds,
            on_progress=lambda msg: emit(msg, {"stage": 5}),
        )
        elapsed = round(time.time() - t0, 2)
        _write_manifest(
            manifest_path,
            paper_pdf=paper_pdf,
            input_figure=input_figure,
            refs_dir=refs_dir,
            settings=settings,
            archetype=archetype,
            viewbox=af_res["viewbox"],
            lang=resolved_lang,
            font_family=font_family,
            svg_mode=svg_mode,
            svg_engine="autofigure",
            seg_backend=af_res["seg_backend"],
            matting_backend=af_res["matting_backend"],
            density=density,
            primary_ref=primary_ref,
            embedded_assets_count=af_res["embedded_assets_count"],
            status="completed",
            elapsed=elapsed,
            round_logs=af_res["round_logs"],
        )
        emit(
            f"Direct AutoFigure-Edit conversion complete in {elapsed:.1f}s ({af_res['embedded_assets_count']} assets)!",
            {"stage": 6, "done": True, "artifact": "final.svg"},
        )
        return PipelineResult(
            output_dir=output_dir,
            final_svg=af_res["final_svg"],
            final_png=af_res["final_png"],
            print_preview_2col=af_res["print_preview_2col"],
            qa_report=af_res["qa_report"],
            stage1_draft=draft_path,
            schema_md=schema_path,
            style_guide_md=style_guide_path,
            paper_method_md=paper_method_path,
            manifest_path=manifest_path,
            archetype=archetype.key,
            viewbox=af_res["viewbox"],
            elapsed_seconds=elapsed,
            embedded_assets_count=af_res["embedded_assets_count"],
            svg_engine="autofigure",
            seg_backend=af_res["seg_backend"],
            matting_backend=af_res["matting_backend"],
            round_logs=af_res["round_logs"],
        )

    if paper_pdf is None and from_stage <= 1:
        raise ValueError("Either `paper_pdf` or `input_figure` must be provided.")

    # ---- STAGE 1: PDF Parsing & Grounded Contribution Extraction ----
    if from_stage <= 1 or not paper_analysis_path.is_file():
        assert paper_pdf is not None
        emit(
            f"Stage 1/6: Parsing full paper PDF via {settings.main_model} (effort={settings.reasoning_effort})...",
            {"stage": 1},
        )
        pdf_info = extract_pdf_sections(paper_pdf)
        resolved_lang = pdf_info["detected_lang"] if (lang or settings.lang) == "auto" else (lang or settings.lang)
        paper_analysis = analyze_paper_structure(client, pdf_info, target_lang=resolved_lang, density=density)
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
        ref_images = discover_reference_images(refs_dir, primary_ref=primary_ref) if (refs_dir or primary_ref) else []
        emit(
            f"Stage 2/6: Extracting true Visual Art Direction from {len(ref_images)} reference image(s)...",
            {"stage": 2},
        )
        style_dna = extract_style_dna(client, ref_images, target_archetype=archetype.key)
        style_dna_path.write_text(json.dumps(style_dna, ensure_ascii=False, indent=2), encoding="utf-8")
        style_guide_path.write_text(render_style_guide_markdown(style_dna), encoding="utf-8")
        emit(
            f"Stage 2/6 complete: Winner=`{Path(style_dna['anchor_files'][0]).name if style_dna.get('anchor_files') else 'Default'}`, "
            f"Paradigm=`{style_dna.get('visual_paradigm')}`",
            {"stage": 2, "artifact": "style_guide.md"},
        )
    else:
        style_dna = json.loads(style_dna_path.read_text(encoding="utf-8"))
        emit("Stage 2/6 skipped (reusing cached style_dna.json)", {"stage": 2})

    # ---- STAGE 3: AIA Physical-Object-First Visual Schema ----
    if from_stage <= 3 or not schema_path.is_file():
        emit(
            f"Stage 3/6: Designing AIA Physical-Object Visual Schema ({archetype.key} / anti-PPT-form)...",
            {"stage": 3},
        )
        schema_md, archetype = generate_visual_schema(
            client,
            paper_analysis,
            style_dna,
            font_family=font_family,
            target_lang=resolved_lang,
            archetype_override=chosen_archetype_key,
            density=density,
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
            svg_engine=svg_engine,
            seg_backend=seg_backend,
            matting_backend=matting_backend,
            density=density,
            primary_ref=primary_ref,
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
            svg_engine=svg_engine,
        )

    # ---- STAGE 4: AIA Visual Anchor Generation + Aesthetic & Anti-Copy Gate ----
    if from_stage <= 4 or not draft_path.is_file():
        emit(
            f"Stage 4/6: Generating Stage-1 anchor via {settings.image_tool_model or 'built-in image_generation'} "
            f"(AIA Aesthetic Gate rounds={review_rounds})...",
            {"stage": 4},
        )
        draft_path, viewbox, draft_meta = generate_stage1_draft(
            client,
            schema_md,
            style_dna,
            archetype,
            output_dir,
            density=density,
            review_rounds=review_rounds,
            on_progress=lambda msg: emit(msg, {"stage": 4}),
        )
        (output_dir / "stage4_meta.json").write_text(json.dumps(draft_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        emit(
            f"Stage 4/6 complete: Anchor saved ({draft_meta.get('trimmed_size')}) | "
            f"Gate score={draft_meta.get('gate_final_score')}/10 (attempts={draft_meta.get('gate_attempts')}).",
            {"stage": 4, "artifact": "stage1_draft.png"},
        )
    else:
        with Image.open(draft_path) as d_img:
            viewbox = compute_svg_viewbox(d_img, archetype)
        emit(f"Stage 4/6 skipped (reusing stage1_draft.png -> viewBox {viewbox[0]}x{viewbox[1]})", {"stage": 4})

    if stop_after_stage == 4:
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
            svg_engine=svg_engine,
            seg_backend=seg_backend,
            matting_backend=matting_backend,
            density=density,
            primary_ref=primary_ref,
            embedded_assets_count=0,
            status="paused_after_draft",
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
            viewbox=viewbox,
            elapsed_seconds=elapsed,
            svg_engine=svg_engine,
        )

    # ---- STAGE 5–6: Back-Half SVG Engine ----
    if svg_engine == "autofigure":
        emit(
            f"Stage 5–6/6 (AutoFigure-Edit Engine): seg={seg_backend} | matting={matting_backend} | opt_rounds={rounds}...",
            {"stage": 5},
        )
        af_res = run_autofigure_svg_engine(
            client,
            draft_path,
            output_dir,
            schema_md=schema_md,
            font_family=font_family,
            svg_mode=svg_mode,
            density=density,
            seg_backend=seg_backend,
            matting_backend=matting_backend,
            sam_prompts=settings.sam_prompts,
            sam_min_score=settings.sam_min_score,
            sam_merge_threshold=settings.sam_merge_threshold,
            sam_api_key=settings.sam_api_key,
            sam_max_masks=settings.sam_max_masks,
            rmbg_model_path=settings.rmbg_model_path,
            enable_upscale_4k=upscale_4k,
            optimize_iterations=rounds,
            on_progress=lambda msg: emit(msg, {"stage": 6 if "Optimize" in msg or "Step 5" in msg else 5}),
        )
        viewbox = af_res["viewbox"]
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
            svg_engine="autofigure",
            seg_backend=af_res["seg_backend"],
            matting_backend=af_res["matting_backend"],
            density=density,
            primary_ref=primary_ref,
            embedded_assets_count=af_res["embedded_assets_count"],
            status="completed",
            elapsed=elapsed,
            round_logs=af_res["round_logs"],
        )
        emit(
            f"All 6 stages complete in {elapsed:.1f}s! Final editable SVG ready ({af_res['embedded_assets_count']} assets, seg={af_res['seg_backend']}, matting={af_res['matting_backend']}).",
            {"stage": 6, "done": True, "artifact": "final.svg"},
        )
        return PipelineResult(
            output_dir=output_dir,
            final_svg=af_res["final_svg"],
            final_png=af_res["final_png"],
            print_preview_2col=af_res["print_preview_2col"],
            qa_report=af_res["qa_report"],
            stage1_draft=draft_path,
            schema_md=schema_path,
            style_guide_md=style_guide_path,
            paper_method_md=paper_method_path,
            manifest_path=manifest_path,
            archetype=archetype.key,
            viewbox=viewbox,
            elapsed_seconds=elapsed,
            embedded_assets_count=af_res["embedded_assets_count"],
            svg_engine="autofigure",
            seg_backend=af_res["seg_backend"],
            matting_backend=af_res["matting_backend"],
            round_logs=af_res["round_logs"],
        )

    # ---- Legacy v1.2 Engine (preserved for backward compatibility when --svg-engine legacy) ----
    assets: list[ExtractedAsset] = []
    if from_stage <= 5 or not raw_svg_path.is_file():
        mode_desc = "Hybrid Editable SVG" if svg_mode == "hybrid_rich" else "Pure Vector SVG"
        emit(
            f"Stage 5/6 (Legacy): Building {mode_desc} ({viewbox[0]}x{viewbox[1]}) via {settings.main_model}...",
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
            density=density,
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

    emit(f"Stage 6/6 (Legacy): Starting {rounds}-round Linter + Dual-Resolution Vision Critic refinement...", {"stage": 6})
    final_svg, final_png_path, twocol_png_path, qa_md, round_logs = refine_svg_with_critic(
        client,
        raw_svg,
        draft_path,
        schema_md,
        viewbox,
        output_dir,
        assets=assets,
        max_rounds=max(1, rounds),
        density=density,
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
        svg_engine="legacy",
        seg_backend="vlm_zero_gpu",
        matting_backend="floodfill",
        density=density,
        primary_ref=primary_ref,
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
        svg_engine="legacy",
        seg_backend="vlm_zero_gpu",
        matting_backend="floodfill",
        round_logs=round_logs,
    )


def _write_manifest(
    manifest_path: Path,
    *,
    paper_pdf: Optional[Path],
    input_figure: Optional[Path] = None,
    refs_dir: Optional[Path],
    settings: Settings,
    archetype: ArchetypeSpec,
    viewbox: tuple[int, int],
    lang: str,
    font_family: str,
    svg_mode: str,
    svg_engine: str = "autofigure",
    seg_backend: str = "auto",
    matting_backend: str = "auto",
    density: str = "auto",
    primary_ref: Optional[Path] = None,
    embedded_assets_count: int,
    status: str,
    elapsed: float,
    round_logs: Optional[list[dict[str, Any]]] = None,
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "paper_pdf": str(paper_pdf) if paper_pdf else None,
        "input_figure": str(input_figure) if input_figure else None,
        "refs_dir": str(refs_dir) if refs_dir else None,
        "primary_ref": str(primary_ref) if primary_ref else None,
        "main_model": settings.main_model,
        "reasoning_effort": settings.reasoning_effort,
        "image_tool_model": settings.image_tool_model,
        "image_quality": settings.image_quality,
        "svg_mode": svg_mode,
        "svg_engine": svg_engine,
        "seg_backend": seg_backend,
        "matting_backend": matting_backend,
        "density": density,
        "embedded_assets_count": embedded_assets_count,
        "archetype": archetype.key,
        "archetype_name": f"{archetype.zh_name} / {archetype.en_name}",
        "viewbox": list(viewbox),
        "aspect_ratio": round(viewbox[0] / float(max(1, viewbox[1])), 2),
        "lang": lang,
        "font_family": font_family,
        "elapsed_seconds": elapsed,
        "round_logs": round_logs or [],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
