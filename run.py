"""CLI entry point for Paper2SVG-ultra (PDF + Reference Folder -> Editable Top-Venue SVG)."""

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
        description="Paper2SVG-ultra: Top-Venue Academic Paper Framework Figure Generator (PDF + Ref Folder -> Editable Hybrid SVG)",
    )
    parser.add_argument("--paper", type=Path, help="Path to your full research paper PDF")
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
        help="Optional single reference image file to pin strictly as the Primary Style Winner (prevents multi-ref style blending)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (defaults to outputs/<paper_stem>_<timestamp>)",
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
        help="Optional override for the 6 Top-Venue Story Archetypes (default: auto-selected from paper story)",
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
        help="hybrid_rich (default: editable vector cards/text/arrows + cropped HD visuals) or pure_vector",
    )
    parser.add_argument(
        "--density",
        choices=["auto", "concise", "detailed"],
        default=None,
        help="Diagram label density tier: auto (adaptive from inventory), concise (airy CVPR/Nature <=30 labels), or detailed (dense IEEE S&P/USENIX multi-panel systems diagram)",
    )
    parser.add_argument(
        "--critic-rounds",
        type=int,
        default=None,
        help="Number of Linter + Dual-Resolution Vision Critic refinement rounds (1-4, default: 2)",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
        help="Resume an existing output directory starting from a specific stage (e.g. 4 after editing schema.md)",
    )
    parser.add_argument(
        "--pause-after-schema",
        action="store_true",
        help="Stop after Stage 3 (schema.md) so you can inspect/edit the blueprint before generating images",
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
                f"[bold cyan]Paper2SVG-ultra Web Workbench[/bold cyan]\n"
                f"URL: [link=http://{host}:{port}]http://{host}:{port}[/link]\n"
                f"Main Model: [green]{settings.main_model}[/green] (effort={settings.reasoning_effort}) | "
                f"Image Model: [green]{settings.image_tool_model or 'endpoint default'}[/green]",
                title="Launching Server",
            )
        )
        uvicorn.run("server:app", host=host, port=port, reload=False)
        return 0

    if not args.paper:
        parser.error("--paper <path/to/paper.pdf> is required (or pass --serve to open the Web Workbench).")

    if args.main_model:
        settings.main_model = args.main_model
    if args.reasoning_effort is not None:
        settings.reasoning_effort = args.reasoning_effort
    if args.image_model:
        settings.image_tool_model = args.image_model
    if args.svg_mode:
        settings.svg_mode = args.svg_mode
    if args.density:
        settings.density = args.density

    paper_pdf: Path = args.paper.resolve()
    refs_dir: Path | None = args.refs.resolve() if args.refs else None
    primary_ref: Path | None = args.primary_ref.resolve() if args.primary_ref else None
    if args.out:
        out_dir = args.out.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in paper_pdf.stem)[:40]
        out_dir = PROJECT_ROOT / "outputs" / f"{safe_stem}_{stamp}"

    console.print(
        Panel.fit(
            f"[bold]Paper PDF:[/bold] {paper_pdf}\n"
            f"[bold]Ref Folder:[/bold] {refs_dir or '(None — using default top-venue Style DNA)'}\n"
            + (f"[bold]Primary Ref:[/bold] {primary_ref}\n" if primary_ref else "")
            + f"[bold]Output Dir:[/bold] {out_dir}\n"
            f"[bold]Models:[/bold] {settings.main_model} (effort={settings.reasoning_effort}) + {settings.image_tool_model or 'default'} (quality={settings.image_quality})\n"
            f"[bold]SVG Mode:[/bold] {settings.svg_mode} | [bold]Density:[/bold] {settings.density} | Start Stage: {args.from_stage}",
            title="[bold cyan]Paper2SVG-ultra[/bold cyan]",
        )
    )

    def _on_event(_evt: str, payload: dict) -> None:
        stage = payload.get("stage", "?")
        msg = payload.get("message", "")
        ts = payload.get("ts", "")
        console.print(f"[dim]{ts}[/dim] [bold blue][Stage {stage}][/bold blue] {msg}")

    result = run_pipeline(
        paper_pdf=paper_pdf,
        refs_dir=refs_dir,
        output_dir=out_dir,
        settings=settings,
        lang=args.lang or settings.lang,
        archetype_override=args.archetype,
        font_override=args.font,
        svg_mode_override=settings.svg_mode,
        density_override=settings.density,
        primary_ref=primary_ref,
        critic_rounds=args.critic_rounds,
        from_stage=args.from_stage,
        stop_after_stage=3 if args.pause_after_schema else None,
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
        ("Visual Schema Blueprint", result.schema_md),
        ("Style DNA Guide", result.style_guide_md),
        ("Paper Analysis", result.paper_method_md),
    ]
    for name, path in rows:
        if path.is_file():
            table.add_row(name, str(path))

    console.print(table)
    console.print(
        f"[bold green]Done![/bold green] Archetype=[cyan]{result.archetype}[/cyan] | "
        f"ViewBox=[cyan]{result.viewbox[0]}x{result.viewbox[1]}[/cyan] | "
        f"Elapsed=[cyan]{result.elapsed_seconds:.1f}s[/cyan]"
    )
    if args.pause_after_schema:
        console.print(
            f"[yellow]Paused after Stage 3.[/yellow] Edit [bold]{result.schema_md}[/bold], then resume with:\n"
            f"  python run.py --paper \"{paper_pdf}\" --out \"{out_dir}\" --from-stage 4"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
