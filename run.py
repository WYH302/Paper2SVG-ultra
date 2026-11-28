"""CLI entry point for Paper2SVG-ultra v2.1 (AIA Front-End + AutoFigure-Edit Back-End)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from paper2svg.archetypes import ARCHETYPES
from paper2svg.config import PROJECT_ROOT, load_settings
from paper2svg.pipeline import run_pipeline

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper2svg",
        description="Paper2SVG-ultra v2.1: AIA-Academic-Illustrator + AutoFigure-Edit Surgical Giant Pipeline (PDF or Figure -> Editable SVG)",
    )
    parser.add_argument("--paper", type=Path, default=None, help="Path to your full research paper PDF")
    parser.add_argument(
        "--input-figure",
        type=Path,
        default=None,
        help="Optional existing high-res raster figure (PNG/JPG) to feed directly into the AutoFigure-Edit SVG engine (skips Stages 1-4)",
    )
    parser.add_argument(
        "--refs",
        type=Path,
        default=None,
        help="Path to a folder containing reference top-venue diagrams (PNG/JPG/WEBP)",
    )
    parser.add_argument(
        "--primary-ref",
        type=Path,
        default=None,
        help="Optional single reference image file to pin strictly as the Primary Style Winner",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (defaults to outputs/<stem>_<timestamp>)",
    )
    parser.add_argument(
        "--lang",
        choices=["auto", "en", "zh"],
        default=None,
        help="Diagram label language: auto (follow paper), en, or zh",
    )
    parser.add_argument(
        "--archetype",
        choices=list(ARCHETYPES.keys()),
        default=None,
        help="Optional override for the 6 Top-Venue Story Archetypes (default: auto-selected)",
    )
    parser.add_argument(
        "--font",
        type=str,
        default=None,
        help="Academic sans-serif font family override (default: Inter, Helvetica, Arial, sans-serif)",
    )
    parser.add_argument(
        "--main-model",
        type=str,
        default=None,
        help="Override main reasoning/vision/SVG model (default from .env)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
        help="Reasoning effort for the main model (default from .env)",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default=None,
        help="Override Stage-1 image generation model (default from .env)",
    )
    parser.add_argument(
        "--svg-mode",
        choices=["hybrid_rich", "pure_vector"],
        default=None,
        help="hybrid_rich (default: editable vectors + lossless transparent assets) or pure_vector",
    )
    parser.add_argument(
        "--svg-engine",
        choices=["autofigure", "legacy"],
        default=None,
        help="autofigure (v2.1 default: true AutoFigure-Edit <AF> template -> optimize -> icon swap) or legacy",
    )
    parser.add_argument(
        "--seg-backend",
        choices=["auto", "sam3_local", "sam3_fal", "sam3_roboflow", "cc_zero_gpu", "vlm_zero_gpu"],
        default=None,
        help="Icon segmentation backend (default: auto -> SAM3 if configured, else 0.2s pixel-exact cc_zero_gpu)",
    )
    parser.add_argument(
        "--matting-backend",
        choices=["auto", "rmbg2", "birefnet", "floodfill"],
        default=None,
        help="Background removal backend (default: auto -> RMBG-2.0 / BiRefNet if installed, else zero-GPU floodfill)",
    )
    parser.add_argument(
        "--sam-prompts",
        type=str,
        default=None,
        help="Comma-separated SAM3 text prompts (default: icon,portrait,diagram,chart,object)",
    )
    parser.add_argument(
        "--sam-api-key",
        type=str,
        default=None,
        help="Optional SAM3 cloud API key (FAL_KEY or ROBOFLOW_API_KEY)",
    )
    parser.add_argument(
        "--rmbg-model-path",
        type=str,
        default=None,
        help="Optional local path to briaai/RMBG-2.0 weights",
    )
    parser.add_argument(
        "--upscale-4k",
        action="store_true",
        default=None,
        help="Enable AutoFigure-Edit 4K proportional upscale (auto-enabled when SAM3 is active)",
    )
    parser.add_argument(
        "--density",
        choices=["auto", "concise", "detailed"],
        default=None,
        help="Diagram label density tier: auto, concise (airy CVPR/Nature 10-18 labels), or detailed (dense systems diagram)",
    )
    parser.add_argument(
        "--draft-review-rounds",
        type=int,
        default=None,
        help="AIA Stage-4 Aesthetic & Anti-Copy Gate max revision rounds (0-3, default: 2)",
    )
    parser.add_argument(
        "--critic-rounds",
        type=int,
        default=None,
        help="Number of AutoFigure template optimization / critic rounds (0-4, default: 2)",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
        help="Resume an existing output directory starting from a specific stage (e.g. 4 after editing schema.md, 5 after approving stage1_draft.png)",
    )
    parser.add_argument(
        "--pause-after-schema",
        action="store_true",
        help="Stop after Stage 3 (schema.md) to inspect/edit the blueprint before rendering images",
    )
    parser.add_argument(
        "--pause-after-draft",
        action="store_true",
        help="Stop after Stage 4 (stage1_draft.png) to visually approve/swap the raster anchor before AutoFigure-Edit SVG conversion",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the local Web Workbench (with embedded svg-edit editor) instead of running CLI",
    )
    parser.add_argument("--host", type=str, default=None, help="Web server host (with --serve)")
    parser.add_argument("--port", type=int, default=None, help="Web server port (with --serve)")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings()

    if args.serve:
        import uvicorn

        host = args.host or settings.server_host
        port = args.port or settings.server_port
        console.print(
            Panel.fit(
                f"[bold cyan]Paper2SVG-ultra v2.1 Web Workbench[/bold cyan]\n"
                f"URL: [link=http://{host}:{port}]http://{host}:{port}[/link]\n"
                f"Main Model: [green]{settings.main_model}[/green] (effort={settings.reasoning_effort}) | "
                f"SVG Engine: [green]{settings.svg_engine}[/green] (seg={settings.seg_backend}, matting={settings.matting_backend})",
                title="Launching Server",
            )
        )
        uvicorn.run("server:app", host=host, port=port, reload=False)
        return 0

    if not args.paper and not args.input_figure:
        parser.error("Provide --paper <paper.pdf> OR --input-figure <figure.png> (or pass --serve for Web Workbench).")

    if args.main_model:
        settings.main_model = args.main_model
    if args.reasoning_effort is not None:
        settings.reasoning_effort = args.reasoning_effort
    if args.image_model:
        settings.image_tool_model = args.image_model
    if args.svg_mode:
        settings.svg_mode = args.svg_mode
    if args.svg_engine:
        settings.svg_engine = args.svg_engine
    if args.seg_backend:
        settings.seg_backend = args.seg_backend
    if args.matting_backend:
        settings.matting_backend = args.matting_backend
    if args.sam_prompts:
        settings.sam_prompts = args.sam_prompts
    if args.sam_api_key:
        settings.sam_api_key = args.sam_api_key
    if args.rmbg_model_path:
        settings.rmbg_model_path = args.rmbg_model_path
    if args.density:
        settings.density = args.density

    paper_pdf: Path | None = args.paper.resolve() if args.paper else None
    input_figure: Path | None = args.input_figure.resolve() if args.input_figure else None
    refs_dir: Path | None = args.refs.resolve() if args.refs else None
    primary_ref: Path | None = args.primary_ref.resolve() if args.primary_ref else None

    if args.out:
        out_dir = args.out.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        src_stem = (input_figure or paper_pdf).stem  # type: ignore[union-attr]
        safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in src_stem)[:40]
        out_dir = PROJECT_ROOT / "outputs" / f"{safe_stem}_{stamp}"

    stop_stage = 3 if args.pause_after_schema else (4 if args.pause_after_draft else None)

    console.print(
        Panel.fit(
            (f"[bold]Paper PDF:[/bold] {paper_pdf}\n" if paper_pdf else "")
            + (f"[bold]Imported Figure:[/bold] {input_figure}\n" if input_figure else "")
            + f"[bold]Ref Folder:[/bold] {refs_dir or '(None)'}\n"
            + (f"[bold]Primary Ref:[/bold] {primary_ref}\n" if primary_ref else "")
            + f"[bold]Output Dir:[/bold] {out_dir}\n"
            + f"[bold]Models:[/bold] {settings.main_model} (effort={settings.reasoning_effort}) + {settings.image_tool_model or 'default'}\n"
            + f"[bold]Engine:[/bold] {settings.svg_engine} | [bold]Seg:[/bold] {settings.seg_backend} | "
            + f"[bold]Matting:[/bold] {settings.matting_backend} | [bold]Density:[/bold] {settings.density}",
            title="[bold cyan]Paper2SVG-ultra v2.1 (AIA × AutoFigure-Edit)[/bold cyan]",
        )
    )

    def _on_event(_evt: str, payload: dict) -> None:
        stage = payload.get("stage", "?")
        msg = payload.get("message", "")
        ts = payload.get("ts", "")
        console.print(f"[dim]{ts}[/dim] [bold blue][Stage {stage}][/bold blue] {msg}")

    result = run_pipeline(
        paper_pdf=paper_pdf,
        input_figure=input_figure,
        refs_dir=refs_dir,
        output_dir=out_dir,
        settings=settings,
        lang=args.lang or settings.lang,
        archetype_override=args.archetype,
        font_override=args.font,
        svg_mode_override=settings.svg_mode,
        svg_engine_override=settings.svg_engine,
        density_override=settings.density,
        primary_ref=primary_ref,
        seg_backend_override=settings.seg_backend,
        matting_backend_override=settings.matting_backend,
        draft_review_rounds=args.draft_review_rounds,
        enable_upscale_4k=args.upscale_4k,
        critic_rounds=args.critic_rounds,
        from_stage=args.from_stage,
        stop_after_stage=stop_stage,
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
        ("Stage-1 Visual Anchor", result.stage1_draft),
        ("AutoFigure Mask Overlay", result.output_dir / "samed.png"),
        ("Visual Schema Blueprint", result.schema_md),
        ("Style DNA Guide", result.style_guide_md),
        ("Paper Analysis", result.paper_method_md),
    ]
    for name, path in rows:
        if path.is_file():
            table.add_row(name, str(path))

    console.print(table)
    console.print(
        f"[bold green]Done![/bold green] Engine=[cyan]{result.svg_engine}[/cyan] "
        f"(seg={result.seg_backend}, matting={result.matting_backend}) | "
        f"ViewBox=[cyan]{result.viewbox[0]}x{result.viewbox[1]}[/cyan] | "
        f"Assets=[cyan]{result.embedded_assets_count}[/cyan] | "
        f"Elapsed=[cyan]{result.elapsed_seconds:.1f}s[/cyan]"
    )
    if args.pause_after_schema and paper_pdf:
        console.print(
            f"[yellow]Paused after Stage 3.[/yellow] Edit [bold]{result.schema_md}[/bold], then resume:\n"
            f"  python run.py --paper \"{paper_pdf}\" --out \"{out_dir}\" --from-stage 4"
        )
    elif args.pause_after_draft and paper_pdf:
        console.print(
            f"[yellow]Paused after Stage 4.[/yellow] Review [bold]{result.stage1_draft}[/bold], then resume:\n"
            f"  python run.py --paper \"{paper_pdf}\" --out \"{out_dir}\" --from-stage 5"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
