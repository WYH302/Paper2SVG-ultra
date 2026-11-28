"""Step 1: Pure AutoFigure-Edit Single-Ref Figure Generator (+ 3-Line Anti-Copy Guard & Own-Paper Subject Images).

Preserves the exact 20-line upstream AutoFigure-Edit single-reference prompt verbatim (verified as the
highest-clarity, lowest-clutter scientific aesthetic across CVPR/ECCV/IEEE S&P/FAST domains), adding ONLY:
  1. A concise 3-line originality guard prohibiting copying faces/photos/legends/charts from the style reference.
  2. Optional author's OWN paper subject images (`--subject-images`) so real input/output examples appear natively.
  3. Clean outer white-margin trim + SSE streaming delivery via `ResponsesAPIClient`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def pick_single_style_reference(
    style_ref: Optional[Path] = None,
    refs_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Select strictly ONE clean style reference image (single-ref AutoFigure-Edit mode)."""
    if style_ref is not None and style_ref.is_file():
        return style_ref
    if refs_dir is not None and refs_dir.is_file():
        return refs_dir
    if refs_dir is not None and refs_dir.is_dir():
        imgs = sorted(
            p for p in refs_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if imgs:
            return imgs[0]
    return None


def trim_uniform_outer_margin(image: Image.Image, pad: int = 10, threshold: float = 12.0) -> Image.Image:
    """Trim excess solid white/background padding around a generated figure while keeping a clean breathing margin."""
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    h, w, _ = arr.shape
    if h < 40 or w < 40:
        return rgb
    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    bg = np.median(border, axis=0)
    fg = np.linalg.norm(arr - bg, axis=2) > threshold
    ys, xs = np.where(fg)
    if len(xs) < 100 or len(ys) < 100:
        return rgb
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    if (x1 - x0) < w * 0.45 or (y1 - y0) < h * 0.45:
        return rgb
    return rgb.crop((x0, y0, x1, y1))


def build_autofigure_step1_prompt(
    method_text: str,
    *,
    has_style_ref: bool,
    has_subject_images: bool = False,
    lang: str = "en",
) -> str:
    """Build the clean upstream AutoFigure-Edit Step 1 prompt + 3-line originality guard."""
    lang_note = (
        "\n- Use concise Chinese technical labels mixed with standard English abbreviations (e.g., LLM, Attention, LoRA)."
        if lang == "zh"
        else ""
    )
    subject_note = (
        "\n- Use the attached OWN PAPER SUBJECT IMAGE(S) directly as the real input/output visual examples in the diagram (do NOT substitute generic or reference-image subjects)."
        if has_subject_images
        else ""
    )

    if has_style_ref:
        return f"""Generate a figure to visualize the method described below.

You should closely imitate the visual (artistic) style of the reference figure I provide, focusing only on aesthetic aspects, NOT on layout or structure.

Specifically, match:
- overall visual tone and mood
- illustration abstraction level
- line style
- color usage
- shading style
- icon and shape style
- arrow and connector aesthetics
- typography feel

The content structure, number of components, and layout may differ freely.
Only the visual style should be consistent.

The goal is that the figure looks like it was drawn by the same illustrator using the same visual design language as the reference figure.

ORIGINALITY GUARD:
- NEVER copy any face, person, photograph, legend, radar chart, symbol, or text label from the style reference image.
- Draw ONLY real modules, data flows, formulas, and visual artifacts described in the paper's method section below.{subject_note}{lang_note}

Below is the method section of the paper:
\"\"\"
{method_text}
\"\"\""""

    return f"""Generate a professional academic journal style figure for the paper below so as to visualize the method it proposes, below is the method section of this paper:

{method_text}

Draw ONLY real modules, data flows, formulas, and visual artifacts described in the method section above, with crisp readable typography and generous clean whitespace.{subject_note}{lang_note}"""


def generate_step1_figure(
    client: Any,
    method_text: str,
    output_dir: Path,
    *,
    style_ref: Optional[Path] = None,
    subject_images: Optional[list[Path]] = None,
    lang: str = "en",
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[Path, tuple[int, int], dict[str, Any]]:
    """Run Step 1: Method Section (+ optional 1 Style Ref + optional Own Paper Subject Images) -> `figure.png`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_subjects = [p for p in (subject_images or []) if p and p.is_file()]
    has_ref = style_ref is not None and style_ref.is_file()

    prompt = build_autofigure_step1_prompt(
        method_text,
        has_style_ref=has_ref,
        has_subject_images=bool(valid_subjects),
        lang=lang,
    )
    (output_dir / "step1_prompt.txt").write_text(prompt, encoding="utf-8")

    if on_progress:
        ref_desc = style_ref.name if has_ref and style_ref else "None (zero-ref)"
        subj_desc = f", {len(valid_subjects)} own-paper subject image(s)" if valid_subjects else ""
        on_progress(
            f"Step 1: Generating high-res academic figure via `{client.settings.image_tool_model or 'default image_generation'}` "
            f"(style_ref={ref_desc}{subj_desc})..."
        )

    img, meta = client.generate_image(
        prompt,
        reference_images=[style_ref] if (has_ref and style_ref is not None) else None,
        subject_images=valid_subjects or None,
    )

    raw_path = output_dir / "figure_raw.png"
    img.convert("RGB").save(raw_path, format="PNG")

    trimmed = trim_uniform_outer_margin(img)
    fig_path = output_dir / "figure.png"
    trimmed.save(fig_path, format="PNG")

    meta_out = {
        **meta,
        "style_ref": str(style_ref) if has_ref and style_ref else None,
        "subject_images": [str(p) for p in valid_subjects],
        "raw_size": list(img.size),
        "trimmed_size": list(trimmed.size),
    }
    return fig_path, trimmed.size, meta_out
