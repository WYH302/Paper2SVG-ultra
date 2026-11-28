"""Paper2SVG-ultra v3.0 — Single-Backbone Pure AutoFigure-Edit Pipeline.

Clean 5-Step Flow (ZERO AIA / Style DNA / Archetypes / Legacy dead code):
  0. Input Slicer: PDF -> Strict Method Section Only (or direct `--method-file` / `--input-figure`)
  1. Step 1 (`figure.png`): Original AutoFigure-Edit 20-line single-ref prompt + 3-line anti-copy guard
     + optional own-paper `--subject-images` (optional pause via `stop_after_step=1`)
  2. Step 2 (`samed.png`): SAM3 (local/fal/roboflow) -> local CUDA GroundingDINO -> zero-GPU frame-stripped CC + VLM filter
  3. Step 3 (`icons/`): Photo-protected batch neural matting (RMBG-2.0 -> zero-token BiRefNet -> zero-GPU floodfill)
  4. Step 4–5.5 (`final.svg`): `<AF>` template -> deterministic arrow/text/collision post-processor -> resvg-py visual polish
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from .autofigure_engine import prepare_imported_figure, run_autofigure_svg_engine
from .client import ResponsesAPIClient
from .config import Settings, sanitize_font_family
from .figure_gen import generate_step1_figure, pick_single_style_reference
from .pdf_parser import extract_method_section, extract_paper_images, load_method_text


@dataclass
class PipelineResult:
    output_dir: Path
    final_svg: Path
    final_png: Path
    print_preview_2col: Path
    qa_report: Path
    figure_png: Path
    samed_png: Path
    method_section_txt: Path
    manifest_path: Path
    viewbox: tuple[int, int]
    elapsed_seconds: float
    embedded_assets_count: int = 0
    seg_backend: str = "auto"
    matting_backend: str = "auto"
    round_logs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "completed"


def run_pipeline(
    *,
    paper_pdf: Optional[Path] = None,
    method_file: Optional[Path] = None,
    method_text_override: Optional[str] = None,
    input_figure: Optional[Path] = None,
    style_ref: Optional[Path] = None,
    refs_dir: Optional[Path] = None,
    subject_images: Optional[list[Path]] = None,
    output_dir: Path,
    settings: Settings,
    lang: str = "auto",
    font_override: Optional[str] = None,
    svg_mode_override: Optional[str] = None,
    density_override: Optional[str] = None,
    seg_backend_override: Optional[str] = None,
    matting_backend_override: Optional[str] = None,
    enable_upscale_4k: Optional[bool] = None,
    optimize_iterations: Optional[int] = None,
    from_step: int = 1,
    stop_after_step: Optional[int] = None,
    on_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> PipelineResult:
    """Execute the clean v3.0 Pure AutoFigure-Edit pipeline."""
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    def emit(msg: str, extra: Optional[dict[str, Any]] = None) -> None:
        if on_event:
            payload = {"message": msg, "ts": datetime.now().isoformat(timespec="seconds")}
            if extra:
                payload.update(extra)
            on_event("progress", payload)

    client = ResponsesAPIClient(settings)
    opt_iters = optimize_iterations if optimize_iterations is not None else settings.optimize_iterations
    svg_mode = (svg_mode_override or settings.svg_mode or "hybrid_rich").strip().lower()
    density = (density_override or settings.density or "auto").strip().lower()
    seg_backend = (seg_backend_override or settings.seg_backend or "auto").strip().lower()
    matting_backend = (matting_backend_override or settings.matting_backend or "auto").strip().lower()
    upscale_4k = settings.enable_upscale_4k if enable_upscale_4k is None else bool(enable_upscale_4k)

    method_txt_path = output_dir / "method_section.txt"
    figure_path = output_dir / "figure.png"
    samed_path = output_dir / "samed.png"
    final_svg_path = output_dir / "final.svg"
    final_png_path = output_dir / "final.png"
    twocol_png_path = output_dir / "print_preview_2col.png"
    qa_path = output_dir / "qa.md"
    manifest_path = output_dir / "run_manifest.json"

    picked_style_ref = pick_single_style_reference(style_ref=style_ref, refs_dir=refs_dir)
    valid_subjects = [p for p in (subject_images or []) if p and p.is_file()]

    # ---- FAST PATH: Direct Imported Figure (`--input-figure`) ----
    if input_figure is not None:
        emit(f"Step 1 (Import): Loading `{input_figure.name}` directly into AutoFigure-Edit SVG engine...", {"stage": 2})
        figure_path, viewbox = prepare_imported_figure(input_figure, figure_path, enable_upscale=False)
        resolved_lang = "en" if (lang or settings.lang) == "auto" else (lang or settings.lang)
        font_family = sanitize_font_family(font_override or settings.font_family, lang=resolved_lang)
        af_res = run_autofigure_svg_engine(
            client,
            figure_path,
            output_dir,
            font_family=font_family,
            svg_mode=svg_mode,
            density=density,
            seg_backend=seg_backend,
            matting_backend=matting_backend,
            gpu_python=settings.gpu_python,
            hf_endpoint=settings.hf_endpoint,
            sam_prompts=settings.sam_prompts,
            sam_min_score=settings.sam_min_score,
            sam_merge_threshold=settings.sam_merge_threshold,
            sam_api_key=settings.sam_api_key,
            sam_max_masks=settings.sam_max_masks,
            rmbg_model_path=settings.rmbg_model_path,
            enable_upscale_4k=upscale_4k,
            optimize_iterations=opt_iters,
            on_progress=lambda m: emit(m, {"stage": _stage_from_msg(m)}),
        )
        elapsed = round(time.time() - t0, 2)
        _write_manifest(
            manifest_path,
            paper_pdf=None,
            method_file=None,
            input_figure=input_figure,
            style_ref=picked_style_ref,
            subject_images=valid_subjects,
            settings=settings,
            viewbox=af_res["viewbox"],
            lang=resolved_lang,
            font_family=font_family,
            svg_mode=svg_mode,
            seg_backend=af_res["seg_backend"],
            matting_backend=af_res["matting_backend"],
            embedded_assets_count=af_res["embedded_assets_count"],
            status="completed",
            elapsed=elapsed,
            round_logs=af_res["round_logs"],
        )
        emit(
            f"Conversion complete in {elapsed:.1f}s ({af_res['embedded_assets_count']} icons, "
            f"seg={af_res['seg_backend']}, matting={af_res['matting_backend']})!",
            {"stage": 5, "done": True, "artifact": "final.svg"},
        )
        return PipelineResult(
            output_dir=output_dir,
            final_svg=af_res["final_svg"],
            final_png=af_res["final_png"],
            print_preview_2col=af_res["print_preview_2col"],
            qa_report=af_res["qa_report"],
            figure_png=figure_path,
            samed_png=af_res["samed_path"],
            method_section_txt=method_txt_path,
            manifest_path=manifest_path,
            viewbox=af_res["viewbox"],
            elapsed_seconds=elapsed,
            embedded_assets_count=af_res["embedded_assets_count"],
            seg_backend=af_res["seg_backend"],
            matting_backend=af_res["matting_backend"],
            round_logs=af_res["round_logs"],
        )

    # ---- STEP 0: Strict Method Section Extraction (0.2s, Zero-LLM) ----
    if from_step <= 1:
        if method_text_override and method_text_override.strip():
            method_text = method_text_override.strip()
            detected_lang = "zh" if sum(1 for c in method_text[:4000] if "\u4e00" <= c <= "\u9fff") > 80 else "en"
            emit(f"Step 0: Using provided method text ({len(method_text):,} chars).", {"stage": 1})
        elif method_file is not None:
            info = load_method_text(method_file)
            method_text = info["method_text"]
            detected_lang = info["detected_lang"]
            emit(f"Step 0: Loaded method file `{method_file.name}` ({len(method_text):,} chars).", {"stage": 1})
        elif paper_pdf is not None:
            emit(f"Step 0: Slicing Method section ONLY from `{paper_pdf.name}` via PyMuPDF...", {"stage": 1})
            info = extract_method_section(paper_pdf)
            method_text = info["method_text"]
            detected_lang = info["detected_lang"]
            kept = ", ".join(info["kept_headings"][:6])
            dropped = ", ".join(info["dropped_headings"][:6]) or "None"
            emit(
                f"Step 0 complete ({len(method_text):,} chars, {info['num_pages']} pages) | Kept: {kept} | Dropped: {dropped}",
                {"stage": 1, "artifact": "method_section.txt"},
            )
            # Also extract embedded paper images into `pdf_images/` for convenient inspection
            extract_paper_images(paper_pdf, output_dir / "pdf_images", max_images=12)
        elif method_txt_path.is_file():
            method_text = method_txt_path.read_text(encoding="utf-8")
            detected_lang = "en"
        else:
            raise ValueError("Provide `--paper <paper.pdf>`, `--method-file <method.txt>`, OR `--input-figure <figure.png>`.")
        method_txt_path.write_text(method_text, encoding="utf-8")
    else:
        method_text = method_txt_path.read_text(encoding="utf-8") if method_txt_path.is_file() else ""
        detected_lang = "en"

    resolved_lang = detected_lang if (lang or settings.lang) == "auto" else (lang or settings.lang)
    font_family = sanitize_font_family(font_override or settings.font_family, lang=resolved_lang)

    # ---- STEP 1: Pure AutoFigure-Edit Single-Ref High-Res Figure Generation ----
    if from_step <= 1 or not figure_path.is_file():
        figure_path, viewbox, step1_meta = generate_step1_figure(
            client,
            method_text,
            output_dir,
            style_ref=picked_style_ref,
            subject_images=valid_subjects,
            lang=resolved_lang,
            on_progress=lambda m: emit(m, {"stage": 2}),
        )
        (output_dir / "step1_meta.json").write_text(json.dumps(step1_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        emit(
            f"Step 1 complete: `figure.png` generated ({viewbox[0]}x{viewbox[1]}).",
            {"stage": 2, "artifact": "figure.png"},
        )
    else:
        with Image.open(figure_path) as im:
            viewbox = im.size
        emit(f"Step 1 skipped (reusing existing `figure.png`, {viewbox[0]}x{viewbox[1]}).", {"stage": 2})

    if stop_after_step == 1:
        elapsed = round(time.time() - t0, 2)
        _write_manifest(
            manifest_path,
            paper_pdf=paper_pdf,
            method_file=method_file,
            input_figure=None,
            style_ref=picked_style_ref,
            subject_images=valid_subjects,
            settings=settings,
            viewbox=viewbox,
            lang=resolved_lang,
            font_family=font_family,
            svg_mode=svg_mode,
            seg_backend=seg_backend,
            matting_backend=matting_backend,
            embedded_assets_count=0,
            status="paused_after_step1",
            elapsed=elapsed,
        )
        emit(
            f"Paused after Step 1 ({elapsed:.1f}s). Review `figure.png` and resume from Step 2 when satisfied!",
            {"stage": 2, "artifact": "figure.png"},
        )
        return PipelineResult(
            output_dir=output_dir,
            final_svg=final_svg_path,
            final_png=final_png_path,
            print_preview_2col=twocol_png_path,
            qa_report=qa_path,
            figure_png=figure_path,
            samed_png=samed_path,
            method_section_txt=method_txt_path,
            manifest_path=manifest_path,
            viewbox=viewbox,
            elapsed_seconds=elapsed,
            seg_backend=seg_backend,
            matting_backend=matting_backend,
            status="paused_after_step1",
        )

    # ---- STEPS 2–5.5: Pure AutoFigure-Edit Vector Engine ----
    af_res = run_autofigure_svg_engine(
        client,
        figure_path,
        output_dir,
        font_family=font_family,
        svg_mode=svg_mode,
        density=density,
        seg_backend=seg_backend,
        matting_backend=matting_backend,
        gpu_python=settings.gpu_python,
        hf_endpoint=settings.hf_endpoint,
        sam_prompts=settings.sam_prompts,
        sam_min_score=settings.sam_min_score,
        sam_merge_threshold=settings.sam_merge_threshold,
        sam_api_key=settings.sam_api_key,
        sam_max_masks=settings.sam_max_masks,
        rmbg_model_path=settings.rmbg_model_path,
        enable_upscale_4k=upscale_4k,
        optimize_iterations=opt_iters,
        on_progress=lambda m: emit(m, {"stage": _stage_from_msg(m)}),
    )
    elapsed = round(time.time() - t0, 2)
    _write_manifest(
        manifest_path,
        paper_pdf=paper_pdf,
        method_file=method_file,
        input_figure=None,
        style_ref=picked_style_ref,
        subject_images=valid_subjects,
        settings=settings,
        viewbox=af_res["viewbox"],
        lang=resolved_lang,
        font_family=font_family,
        svg_mode=svg_mode,
        seg_backend=af_res["seg_backend"],
        matting_backend=af_res["matting_backend"],
        embedded_assets_count=af_res["embedded_assets_count"],
        status="completed",
        elapsed=elapsed,
        round_logs=af_res["round_logs"],
    )
    emit(
        f"All steps complete in {elapsed:.1f}s! Editable SVG ready ({af_res['embedded_assets_count']} icons, "
        f"seg={af_res['seg_backend']}, matting={af_res['matting_backend']}).",
        {"stage": 5, "done": True, "artifact": "final.svg"},
    )
    return PipelineResult(
        output_dir=output_dir,
        final_svg=af_res["final_svg"],
        final_png=af_res["final_png"],
        print_preview_2col=af_res["print_preview_2col"],
        qa_report=af_res["qa_report"],
        figure_png=figure_path,
        samed_png=af_res["samed_path"],
        method_section_txt=method_txt_path,
        manifest_path=manifest_path,
        viewbox=af_res["viewbox"],
        elapsed_seconds=elapsed,
        embedded_assets_count=af_res["embedded_assets_count"],
        seg_backend=af_res["seg_backend"],
        matting_backend=af_res["matting_backend"],
        round_logs=af_res["round_logs"],
    )


def _stage_from_msg(msg: str) -> int:
    if "Step 2" in msg or "Step 3" in msg:
        return 3
    if "Step 4" in msg:
        return 4
    if "Step 5" in msg:
        return 5
    return 4


def _write_manifest(
    manifest_path: Path,
    *,
    paper_pdf: Optional[Path],
    method_file: Optional[Path],
    input_figure: Optional[Path],
    style_ref: Optional[Path],
    subject_images: list[Path],
    settings: Settings,
    viewbox: tuple[int, int],
    lang: str,
    font_family: str,
    svg_mode: str,
    seg_backend: str,
    matting_backend: str,
    embedded_assets_count: int,
    status: str,
    elapsed: float,
    round_logs: Optional[list[dict[str, Any]]] = None,
) -> None:
    manifest = {
        "version": "3.0.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "paper_pdf": str(paper_pdf) if paper_pdf else None,
        "method_file": str(method_file) if method_file else None,
        "input_figure": str(input_figure) if input_figure else None,
        "style_ref": str(style_ref) if style_ref else None,
        "subject_images": [str(p) for p in subject_images],
        "main_model": settings.main_model,
        "reasoning_effort": settings.reasoning_effort,
        "image_tool_model": settings.image_tool_model,
        "svg_mode": svg_mode,
        "seg_backend": seg_backend,
        "matting_backend": matting_backend,
        "embedded_assets_count": embedded_assets_count,
        "viewbox": list(viewbox),
        "aspect_ratio": round(viewbox[0] / float(max(1, viewbox[1])), 2),
        "lang": lang,
        "font_family": font_family,
        "elapsed_seconds": elapsed,
        "round_logs": round_logs or [],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
