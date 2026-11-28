"""Zero-LLM Strict Method-Section PDF Slicer & Own-Paper Figure Extractor (v3.0).

Key design fixes verified across CVPR, ECCV, IEEE TPAMI/S&P, and USENIX/FAST manuscripts:
  1. Recognizes BOTH Arabic (`3. Method`, `3.1 ...`) AND IEEE Roman numeral headings
     (`III. METHODOLOGY`, `III. SYSTEM MODEL` + `IV. PROPOSED METHOD`) AND Chinese headings (`3 方法`, `三、方法`).
  2. Strictly slices ONLY Title + brief Abstract + Method/Framework/Architecture/System section(s),
     cutting cleanly BEFORE `Experiments` / `Evaluation` / `Results` / `References` so PSNR/FID/ablation numbers
     never pollute the architecture overview figure.
  3. Also supports direct `.txt`/`.md` method files AND automatic extraction of the paper's own embedded
     input/output images (`extract_paper_images`) so Step 1 uses the author's real visual subjects.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

_NUM_PREFIX = r"^\s*(?:(?:[IVX]{1,5}\.\s+|[IVX]{1,5}\s+|\d+(?:\.\d+)?\.?\s+|[一二三四五六七八九十]+[、.．]\s*)?)"

METHOD_START_RE = re.compile(
    _NUM_PREFIX
    + r"(?:proposed\s+)?(?:method|methodology|approach|framework|architecture|system\s+model|"
    r"system\s+design|system\s+overview|overview\s+of|problem\s+formulation|preliminar|"
    r"algorithm|model\s+design|our\s+approach|design|方法|总体框架|模型架构|系统设计|方法论)\b",
    re.IGNORECASE,
)

STOP_AFTER_METHOD_RE = re.compile(
    _NUM_PREFIX
    + r"(?:experiments?|experimental\s+results?|experimental\s+setup|evaluation|empirical|"
    r"results?\s+and\s+discussion|ablation|discussion|limitations?|conclusion|concluding|"
    r"references|bibliography|acknowledg(?:e)?ments?|appendix|supplementary|"
    r"实验|实验结果|消融|评估|结果与讨论|讨论|结论|参考文献|致谢|附录)\b",
    re.IGNORECASE,
)

DROP_ALWAYS_RE = re.compile(
    _NUM_PREFIX
    + r"(?:related\s+work|background\s+and\s+related\s+work|literature\s+review|"
    r"references|bibliography|acknowledg(?:e)?ments?|ethics\s+statement|broader\s+impact|"
    r"reproducibility|appendix|supplementary|相关工作|参考文献|致谢|附录)\b",
    re.IGNORECASE,
)

TOP_LEVEL_HEADING_RE = re.compile(
    r"^\s*(?:[IVX]{1,5}\.\s+\S|\d+\.?\s+[A-Z\u4e00-\u9fff]|[一二三四五六七八九十]+[、.．]\s*\S|ABSTRACT\b|Abstract\b|摘要\b)"
)


def detect_language(text: str) -> str:
    """Detect whether the manuscript is predominantly Chinese or English."""
    sample = text[:8000]
    cjk_count = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    latin_count = sum(1 for ch in sample if ("a" <= ch.lower() <= "z"))
    if cjk_count > 120 and cjk_count > latin_count * 0.25:
        return "zh"
    return "en"


def load_method_text(method_file: Path, max_chars: int = 28000) -> dict[str, Any]:
    """Load a user-supplied plain-text or Markdown method description."""
    if not method_file.is_file():
        raise FileNotFoundError(f"Method text file not found: {method_file}")
    raw = method_file.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        raise ValueError(f"Method text file is empty: {method_file}")
    text = raw[:max_chars]
    return {
        "source_path": str(method_file),
        "source_type": "method_file",
        "num_pages": 0,
        "detected_lang": detect_language(text),
        "kept_headings": [method_file.name],
        "dropped_headings": [],
        "method_text": text,
        "extracted_text": text,
    }


def extract_method_section(pdf_path: Path, max_chars: int = 28000) -> dict[str, Any]:
    """Strictly extract ONLY Title + brief Abstract + Method section(s) from a research paper PDF.

    - Supports Arabic (`3. Method`) and IEEE Roman (`III. METHODOLOGY`, `IV. EXPERIMENTS`) numbering.
    - Stops cleanly at `Experiments` / `Evaluation` / `Results` / `References`.
    - For short `<= 5`-page technical specs without a separate `Method` heading, keeps all non-experiment/non-reference sections.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Paper PDF not found: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    num_pages = len(doc)

    font_sizes: list[float] = []
    page_lines: list[tuple[int, str, float, bool]] = []
    for page_idx in range(num_pages):
        blocks = doc[page_idx].get_text("dict").get("blocks", [])
        for b in blocks:
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = "".join(s.get("text", "") for s in spans).strip()
                if not line_text:
                    continue
                max_size = max(float(s.get("size", 10.0)) for s in spans)
                is_bold = any(
                    ("bold" in (s.get("font", "") or "").lower()) or bool(s.get("flags", 0) & 16)
                    for s in spans
                )
                font_sizes.append(round(max_size, 1))
                page_lines.append((page_idx + 1, line_text, max_size, is_bold))
    doc.close()

    if not page_lines:
        raise ValueError(f"No extractable text found in PDF: {pdf_path}")

    font_sizes.sort()
    body_size = font_sizes[len(font_sizes) // 2]

    sections: list[dict[str, Any]] = []
    cur_title = "Title & Preamble"
    cur_lines: list[str] = []
    cur_pages: set[int] = set()

    for page_no, text, size, is_bold in page_lines:
        is_short = len(text) <= 95 and not text.endswith(".")
        looks_heading = is_short and (
            (size >= body_size + 0.7 and (is_bold or bool(TOP_LEVEL_HEADING_RE.match(text))))
            or (is_bold and bool(METHOD_START_RE.match(text)))
            or (is_bold and bool(STOP_AFTER_METHOD_RE.match(text)))
            or (is_bold and bool(DROP_ALWAYS_RE.match(text)))
            or bool(re.match(r"^(?:[IVX]{1,5}\.\s+[A-Z][A-Z\s\-]{3,}|Abstract|ABSTRACT|摘要)\b", text))
        )
        if looks_heading and cur_lines:
            sections.append({"heading": cur_title, "pages": sorted(cur_pages), "text": "\n".join(cur_lines).strip()})
            cur_title = text
            cur_lines = []
            cur_pages = {page_no}
        else:
            cur_lines.append(text)
            cur_pages.add(page_no)

    if cur_lines:
        sections.append({"heading": cur_title, "pages": sorted(cur_pages), "text": "\n".join(cur_lines).strip()})

    # Locate the Method window strictly: from the first Method/System/Framework heading until Experiments/Evaluation/References
    method_indices: list[int] = []
    in_method = False
    dropped_headings: list[str] = []

    for idx, sec in enumerate(sections):
        h = sec["heading"]
        if idx == 0:
            continue
        if STOP_AFTER_METHOD_RE.match(h):
            if in_method:
                in_method = False
            dropped_headings.append(h)
            continue
        if DROP_ALWAYS_RE.match(h):
            dropped_headings.append(h)
            continue
        if METHOD_START_RE.match(h):
            in_method = True
            method_indices.append(idx)
            continue
        if in_method:
            # Keep subsections inside Method (e.g. `A. Dual-Domain Encoding`, `3.2 Solver`, etc.)
            method_indices.append(idx)

    selected: list[dict[str, Any]] = []
    if method_indices:
        # Strictly feed ONLY the Method section(s): omit full paper Title & boastful Abstract so Step 1
        # never slaps a useless paper-title banner on top or marketing/metric claims along the bottom!
        for idx in method_indices:
            selected.append(sections[idx])
        sel_ids = {id(sections[i]) for i in method_indices}
        for idx, sec in enumerate(sections):
            if id(sec) not in sel_ids and sec["heading"] not in dropped_headings:
                dropped_headings.append(sec["heading"])
    else:
        # Short technical spec or non-standard heading layout: keep non-dropped body sections before Experiments/References
        for idx, sec in enumerate(sections):
            h = sec["heading"]
            if idx == 0 and len(sections) > 1:
                dropped_headings.append(h)
                continue
            if STOP_AFTER_METHOD_RE.match(h) or DROP_ALWAYS_RE.match(h) or re.match(r"^(?:Abstract|ABSTRACT|摘要)\b", h):
                if h not in dropped_headings:
                    dropped_headings.append(h)
                continue
            selected.append(sec)

    # Assemble within character budget (Method sections strictly prioritized)
    chunks: list[str] = []
    used = 0
    kept_headings: list[str] = []
    for sec in selected:
        p0, p1 = sec["pages"][0], sec["pages"][-1]
        chunk = f"## {sec['heading']} (p.{p0}-{p1})\n{sec['text']}\n"
        if used + len(chunk) <= max_chars:
            chunks.append(chunk)
            kept_headings.append(sec["heading"])
            used += len(chunk)
        elif used < max_chars - 1000:
            rem = max_chars - used - 120
            chunks.append(chunk[:rem] + "\n...[truncated]\n")
            kept_headings.append(sec["heading"])
            used = max_chars
            break

    method_text = "\n".join(chunks).strip()
    return {
        "source_path": str(pdf_path),
        "pdf_path": str(pdf_path),
        "source_type": "pdf",
        "num_pages": num_pages,
        "detected_lang": detect_language(method_text),
        "kept_headings": kept_headings,
        "dropped_headings": dropped_headings,
        "method_text": method_text,
        "extracted_text": method_text,
    }


# Backward-compatible alias
extract_pdf_sections = extract_method_section


def extract_paper_images(
    pdf_path: Path,
    out_dir: Path,
    *,
    min_dim: int = 80,
    max_images: int = 16,
) -> list[dict[str, Any]]:
    """Extract embedded raster figures/photos from a paper PDF so the user can pick their OWN
    real input/output visuals (`--subject-images`) instead of relying on synthetic placeholders.
    """
    if not pdf_path.is_file():
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    seen_xrefs: set[int] = set()
    extracted: list[dict[str, Any]] = []

    try:
        for page_idx in range(len(doc)):
            for img_idx, info in enumerate(doc[page_idx].get_images(full=True)):
                xref = int(info[0])
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    raw = doc.extract_image(xref)
                    w, h = int(raw["width"]), int(raw["height"])
                    if w < min_dim or h < min_dim:
                        continue
                    pil_im = Image.open(io.BytesIO(raw["image"])).convert("RGB")
                    fname = f"p{page_idx + 1}_{img_idx}_{xref}_{w}x{h}.png"
                    out_path = out_dir / fname
                    pil_im.save(out_path, format="PNG")
                    extracted.append(
                        {
                            "filename": fname,
                            "path": str(out_path),
                            "page": page_idx + 1,
                            "width": w,
                            "height": h,
                        }
                    )
                    if len(extracted) >= max_images:
                        return extracted
                except Exception:
                    continue
    finally:
        doc.close()

    return extracted
