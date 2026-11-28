"""CLI entry point for Paper2SVG-ultra v3.0 (Pure AutoFigure-Edit + Local GPU/Zero-GPU + Deterministic SVG)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from paper2svg.autofigure_engine import resolve_matting_backend, resolve_seg_backend
from paper2svg.config import PROJECT_ROOT, load_settings
from paper2svg.pipeline import run_pipeline

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper2svg",
        description="Paper2SVG-ultra v3.0: Method Section + 1 Style Ref (+ Own Paper Images) -> Editable Top-Venue SVG",
    )
    parser.add_argument("--paper", type=Path, default=None, help="Path to your research paper PDF (auto-slices Method section ONLY)")
    parser.add_argument("--method-file", type=Path, default=None, help="Path to a plain-text / Markdown Method section file")
    parser.add_argument(
        "--input-figure",
        type=Path,
        default=None,
        help="Optional existing high-res raster figure (PNG/JPG) to convert directly into editable SVG (skips Step 1)",
    )
    parser.add_argument(
        "--style-ref",
        type=Path,
        default=None,
        help="Single style reference image (PNG/JPG/WEBP) whose artistic taste/palette/lines to match",
    )
    parser.add_argument(
        "--refs",
        type=Path,
        default=None,
        help="Optional reference image or folder (picks the first image as the single style reference)",
    )
    parser.add_argument(
        "--subject-images",
        type=Path,
        nargs="+",
        default=None,
        help="Optional own-paper input/output images (PNG/JPG) to use directly as visual subjects in Step 1",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output directory (defaults to outputs/<stem>_<timestamp>)")
    parser.add_argument("--lang", choices=["auto", "en", "zh"], default=None, help="Label language: auto, en, or zh")
    parser.add_argument("--font", type=str, default=None, help="Default academic font family override")
    parser.add_argument("--main-model", type=str, default=None, help="Override main reasoning/vision/SVG model")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"], default=None)
    parser.add_argument("--image-model", type=str, default=None, help="Override Step-1 image generation model")
    parser.add_argument("--svg-mode", choices=["hybrid_rich", "pure_vector"], default=None)
    parser.add_argument(
        "--seg-backend",
        choices=["auto", "sam3_local", "sam3_fal", "sam3_roboflow", "grounding_dino", "cc_zero_gpu"],
        default=None,
        help="Step 2 icon segmentation backend (default: auto -> SAM3 -> local GPU GroundingDINO -> zero-GPU CC)",
    )
    parser.add_argument(
        "--matting-backend",
        choices=["auto", "rmbg2", "birefnet", "floodfill"],
        default=None,
        help="Step 3 background removal backend (default: auto -> RMBG-2.0 -> zero-token BiRefNet -> zero-GPU floodfill)",
    )
    parser.add_argument("--gpu-python", type=str, default=None, help="Optional CUDA Python executable for local GPU SAM3/GroundingDINO/RMBG-2.0/BiRefNet")
    parser.add_argument("--sam-prompts", type=str, default=None, help="Comma-separated SAM3 text prompts")
    parser.add_argument("--sam-api-key", type=str, default=None, help="Optional SAM3 cloud API key (FAL_KEY or ROBOFLOW_API_KEY)")
    parser.add_argument("--rmbg-model-path", type=str, default=None, help="Optional local path to briaai/RMBG-2.0 weights")
    parser.add_argument("--upscale-4k", action="store_true", default=None, help="Enable 4K proportional upscale (auto-enabled with SAM3)")
    parser.add_argument("--optimize-iterations", type=int, default=None, help="SVG alignment + post-swap polish rounds (0-4, default: 2)")
    parser.add_argument("--from-step", type=int, choices=[1, 2], default=1, help="Start from Step 1 (default) or resume from Step 2 using existing figure.png")
    parser.add_argument("--pause-after-step1", action="store_true", help="Stop after Step 1 (`figure.png`) to inspect/approve the figure before SVG conversion")
    parser.add_argument("--serve", action="store_true", help="Launch the local Web Workbench (with embedded svg-edit editor)")
    parser.add_argument("--host", type=str, default=None, help="Web server host (with --serve)")
    parser.add_argument("--port", type=int, default=None, help="Web server port (with --serve)")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings()

    if args.gpu_python:
        settings.gpu_python = args.gpu_python
    if args.seg_backend:
        settings.seg_backend = args.seg_backend
    if args.matting_backend:
        settings.matting_backend = args.matting_backend

    if args.serve:
        import uvicorn

        host = args.host or settings.server_host
        port = args.port or settings.server_port
        eff_seg = resolve_seg_backend(settings.seg_backend, settings.sam_api_key, gpu_python=settings.gpu_python)
        eff_mat = resolve_matting_backend(settings.matting_backend, settings.rmbg_model_path, gpu_python=settings.gpu_python)
        console.print(
            Panel.fit(
                f"[bold cyan]Paper2SVG-ultra v3.0 Web Workbench[/bold cyan]\n"
                f"URL: [link=http://{host}:{port}]http://{host}:{port}[/link]\n"
                f"Model: [green]{settings.main_model}[/green] (effort={settings.reasoning_effort}) | "
                f"Seg: [green]{eff_seg}[/green] | Matting: [green]{eff_mat}[/green]",
                title="Launching Server",
            )
        )
        uvicorn.run("server:app", host=host, port=port, reload=False)
        return 0

    if not args.paper and not args.method_file and not args.input_figure and args.from_step == 1:
        parser.error("Provide --paper <paper.pdf>, --method-file <method.txt>, OR --input-figure <figure.png> (or --serve).")

    if args.main_model:
        settings.main_model = args.main_model
    if args.reasoning_effort is not None:
        settings.reasoning_effort = args.reasoning_effort
    if args.image_model:
        settings.image_tool_model = args.image_model
    if args.svg_mode:
        settings.svg_mode = args.svg_mode
    if args.sam_prompts:
        settings.sam_prompts = args.sam_prompts
    if args.sam_api_key:
        settings.sam_api_key = args.sam_api_key
    if args.rmbg_model_path:
        settings.rmbg_model_path = args.rmbg_model_path

    paper_pdf = args.paper.resolve() if args.paper else None
    method_file = args.method_file.resolve() if args.method_file else None
    input_figure = args.input_figure.resolve() if args.input_figure else None
    style_ref = args.style_ref.resolve() if args.style_ref else None
    refs_dir = args.refs.resolve() if args.refs else None
    subject_images = [p.resolve() for p in args.subject_images] if args.subject_images else None

    if args.out:
        out_dir = args.out.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        src = input_figure or method_file or paper_pdf
        src_stem = src.stem if src else "figure"
        safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in src_stem)[:40]
        out_dir = PROJECT_ROOT / "outputs" / f"{safe_stem}_{stamp}"

    eff_seg = resolve_seg_backend(settings.seg_backend, settings.sam_api_key, gpu_python=settings.gpu_python)
    eff_mat = resolve_matting_backend(settings.matting_backend, settings.rmbg_model_path, gpu_python=settings.gpu_python)
    console.print(
        Panel.fit(
            (f"[bold]Paper PDF:[/bold] {paper_pdf}\n" if paper_pdf else "")
            + (f"[bold]Method File:[/bold] {method_file}\n" if method_file else "")
            + (f"[bold]Imported Figure:[/bold] {input_figure}\n" if input_figure else "")
            + f"[bold]Style Ref:[/bold] {style_ref or refs_dir or '(Zero-ref)'}\n"
            + (f"[bold]Own-Paper Subjects:[/bold] {len(subject_images)} image(s)\n" if subject_images else "")
            + f"[bold]Output Dir:[/bold] {out_dir}\n"
            + f"[bold]Models:[/bold] {settings.main_model} (effort={settings.reasoning_effort}) + {settings.image_tool_model or 'default'}\n"
            + f"[bold]Seg:[/bold] {eff_seg} | [bold]Matting:[/bold] {eff_mat} | [bold]Opt Rounds:[/bold] {args.optimize_iterations if args.optimize_iterations is not None else settings.optimize_iterations}",
            title="[bold cyan]Paper2SVG-ultra v3.0 (Pure AutoFigure-Edit)[/bold cyan]",
        )
    )

    def _on_event(_evt: str, payload: dict) -> None:
        stage = payload.get("stage", "?")
        msg = payload.get("message", "")
        ts = payload.get("ts", "")
        console.print(f"[dim]{ts}[/dim] [bold blue][Step {stage}][/bold blue] {msg}")

    result = run_pipeline(
        paper_pdf=paper_pdf,
        method_file=method_file,
        input_figure=input_figure,
        style_ref=style_ref,
        refs_dir=refs_dir,
        subject_images=subject_images,
        output_dir=out_dir,
        settings=settings,
        lang=args.lang or settings.lang,
        font_override=args.font,
        svg_mode_override=settings.svg_mode,
        seg_backend_override=settings.seg_backend,
        matting_backend_override=settings.matting_backend,
        enable_upscale_4k=args.upscale_4k,
        optimize_iterations=args.optimize_iterations,
        from_step=args.from_step,
        stop_after_step=1 if args.pause_after_step1 else None,
        on_event=_on_event,
    )

    table = Table(title="Generated Artifacts", show_header=True, header_style="bold magenta")
    table.add_column("Artifact")
    table.add_column("Path")
    rows = [
        ("Editable SVG (Primary)", result.final_svg),
        ("High-Res PNG Preview (2x)", result.final_png),
        ("2-Column Print Preview (700px)", result.print_preview_2col),
        ("QA Audit Report", result.qa_report),
        ("Step 1 High-Res Figure", result.figure_png),
        ("Step 2 Icon Mask Overlay", result.samed_png),
        ("Sliced Method Section", result.method_section_txt),
    ]
    for name, path in rows:
        if path.is_file():
            table.add_row(name, str(path))

    console.print(table)
    console.print(
        f"[bold green]Done![/bold green] Seg=[cyan]{result.seg_backend}[/cyan] | "
        f"Matting=[cyan]{result.matting_backend}[/cyan] | "
        f"ViewBox=[cyan]{result.viewbox[0]}x{result.viewbox[1]}[/cyan] | "
        f"Icons=[cyan]{result.embedded_assets_count}[/cyan] | "
        f"Elapsed=[cyan]{result.elapsed_seconds:.1f}s[/cyan]"
    )
    if args.pause_after_step1:
        console.print(
            f"[yellow]Paused after Step 1.[/yellow] Inspect [bold]{result.figure_png}[/bold], then convert to editable SVG:\n"
            f'  python run.py --out "{out_dir}" --from-step 2'
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
