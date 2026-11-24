"""Windows-bulletproof SVG -> PNG renderer (resvg-py primary + optional Playwright Chromium/Edge fallback)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from PIL import Image


def _system_font_dirs() -> list[str]:
    candidates = [
        Path(r"C:\Windows\Fonts"),
        Path("/usr/share/fonts"),
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
    ]
    return [str(p) for p in candidates if p.is_dir()]


def render_svg_to_png(
    svg_code: str,
    output_path: Path,
    *,
    width: Optional[int] = None,
    zoom: Optional[float] = None,
) -> str:
    """Render SVG XML string to PNG using resvg-py (primary) or Playwright Edge/Chromium (fallback).

    Returns the name of the renderer backend used.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Engine 1: resvg-py (pure Rust static wheel, zero DLL headaches, crisp <filter>/<marker>/text support)
    try:
        import resvg_py

        kwargs: dict[str, object] = {
            "svg_string": svg_code,
            "background": "#FFFFFF",
            "sans_serif_family": "Arial",
            "font_family": "Arial",
        }
        font_dirs = _system_font_dirs()
        if font_dirs:
            kwargs["font_dirs"] = font_dirs
        if width is not None:
            kwargs["width"] = int(width)
        elif zoom is not None:
            kwargs["zoom"] = float(zoom)

        png_bytes = resvg_py.svg_to_bytes(**kwargs)
        output_path.write_bytes(png_bytes)
        return "resvg-py"
    except Exception as resvg_err:
        # Engine 2: Playwright (Edge / Chrome / bundled Chromium) fallback
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = None
                for launch_kwargs in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
                    try:
                        browser = p.chromium.launch(**launch_kwargs)
                        break
                    except Exception:
                        continue
                if browser is None:
                    raise RuntimeError("No Playwright browser binary available")
                page = browser.new_page(
                    viewport={"width": int(width or 1600), "height": 1000},
                    device_scale_factor=float(zoom or 1.0),
                )
                html = (
                    "<!doctype html><html><body style='margin:0;padding:0;background:#fff;display:inline-block;'>"
                    f"{svg_code}</body></html>"
                )
                page.set_content(html, wait_until="networkidle")
                svg_loc = page.locator("svg").first
                svg_loc.screenshot(path=str(output_path))
                browser.close()
            return "playwright"
        except Exception as pw_err:
            raise RuntimeError(
                f"Both SVG renderers failed. resvg-py: {resvg_err} | playwright: {pw_err}"
            ) from pw_err


def render_dual_preview(
    svg_code: str,
    full_png_path: Path,
    twocol_png_path: Path,
    *,
    viewbox: tuple[int, int],
    full_zoom: float = 1.0,
    twocol_width_px: int = 700,
) -> str:
    """Render both a high-res preview AND a simulated 7-inch two-column paper print preview."""
    w, _ = viewbox
    target_full_w = int(round(w * full_zoom))
    backend = render_svg_to_png(svg_code, full_png_path, width=target_full_w)

    # Render crisp simulated two-column print size (700px wide ~ 7 inches @ 100dpi screen preview)
    try:
        render_svg_to_png(svg_code, twocol_png_path, width=twocol_width_px)
    except Exception:
        # Fallback to Lanczos downsample from full PNG
        with Image.open(full_png_path) as img:
            ratio = twocol_width_px / float(max(1, img.width))
            resized = img.resize(
                (twocol_width_px, max(1, int(round(img.height * ratio)))),
                Image.Resampling.LANCZOS,
            )
            resized.save(twocol_png_path, format="PNG")

    return backend
