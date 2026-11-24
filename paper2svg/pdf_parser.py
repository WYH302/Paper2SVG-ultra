"""Stage 1: Intelligent section-aware PDF parser & paper contribution extractor."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pymupdf

from .archetypes import format_archetype_menu

# Sections to skip so noisy citations/surveys never pollute the overview figure
DROP_SECTION_PATTERNS = re.compile(
    r"^\s*(\d+\.?\s*)?(references|bibliography|acknowledg(e)?ments?|ethics\s+statement|"
    r"broader\s+impact|reproducibility|related\s+work|background\s+and\s+related\s+work|"
    r"appendix|supplementary\s+material|参考文献|致谢|相关工作|附录)\b",
    re.IGNORECASE,
)

HIGH_PRIORITY_PATTERNS = re.compile(
    r"^\s*(\d+\.?\s*)?(abstract|introduction|overview|preliminar|problem\s+formulation|"
    r"method|methodology|approach|framework|architecture|model|proposed|system\s+design|"
    r"algorithm|benchmark|dataset|taxonomy|design|摘要|引言|方法|总体框架|模型架构|系统设计)\b",
    re.IGNORECASE,
)

LOW_PRIORITY_PATTERNS = re.compile(
    r"^\s*(\d+\.?\s*)?(experiment|evaluation|ablation|result|discussion|limitation|conclusion|"
    r"实验|消融|结果|讨论|结论)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Detect whether the paper is predominantly Chinese or English."""
    sample = text[:8000]
    cjk_count = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    latin_count = sum(1 for ch in sample if ("a" <= ch.lower() <= "z"))
    if cjk_count > 120 and cjk_count > latin_count * 0.25:
        return "zh"
    return "en"


def extract_pdf_sections(pdf_path: Path, max_chars: int = 42000) -> dict[str, Any]:
    """Extract structured sections from a full research paper PDF using PyMuPDF.

    Preserves Title, Abstract, Introduction, and full Method/Framework sections while
    stripping References, Related Work, and Appendix noise.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Paper PDF not found: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    num_pages = len(doc)

    # First pass: estimate body font size median to recognize section headers
    font_sizes: list[float] = []
    page_lines: list[tuple[int, str, float, bool]] = []

    for page_idx in range(num_pages):
        page = doc[page_idx]
        blocks = page.get_text("dict").get("blocks", [])
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

    # Segment into sections
    sections: list[dict[str, Any]] = []
    current_title = "Preamble & Title"
    current_lines: list[str] = []
    current_pages: set[int] = set()

    for page_no, text, size, is_bold in page_lines:
        # Heuristic for top-level section header
        looks_like_heading = (
            len(text) <= 90
            and not text.endswith(".")
            and (
                (size >= body_size + 0.8 and (is_bold or re.match(r"^(\d+|[IVX]+)\b", text)))
                or (is_bold and HIGH_PRIORITY_PATTERNS.match(text))
                or (is_bold and DROP_SECTION_PATTERNS.match(text))
                or (is_bold and LOW_PRIORITY_PATTERNS.match(text))
                or re.match(r"^(Abstract|ABSTRACT|摘要)\b", text)
            )
        )
        if looks_like_heading and current_lines:
            sections.append(
                {
                    "heading": current_title,
                    "pages": sorted(current_pages),
                    "text": "\n".join(current_lines).strip(),
                }
            )
            current_title = text
            current_lines = []
            current_pages = {page_no}
        else:
            current_lines.append(text)
            current_pages.add(page_no)

    if current_lines:
        sections.append(
            {
                "heading": current_title,
                "pages": sorted(current_pages),
                "text": "\n".join(current_lines).strip(),
            }
        )

    # Filter and prioritize sections
    kept_high: list[dict[str, Any]] = []
    kept_mid: list[dict[str, Any]] = []
    kept_low: list[dict[str, Any]] = []
    dropped_headings: list[str] = []

    hit_references = False
    for sec in sections:
        h = sec["heading"]
        if re.match(r"^\s*(\d+\.?\s*)?(references|bibliography|参考文献)\b", h, re.IGNORECASE):
            hit_references = True
            dropped_headings.append(h)
            continue
        if hit_references or DROP_SECTION_PATTERNS.match(h):
            dropped_headings.append(h)
            continue
        if h == "Preamble & Title" or HIGH_PRIORITY_PATTERNS.match(h):
            kept_high.append(sec)
        elif LOW_PRIORITY_PATTERNS.match(h):
            kept_low.append(sec)
        else:
            kept_mid.append(sec)

    # Assemble within budget: ALL high-priority first, then mid-priority (likely method subsections), then low-priority
    selected: list[dict[str, Any]] = []
    used_chars = 0
    for bucket in (kept_high, kept_mid, kept_low):
        for sec in bucket:
            chunk = f"## {sec['heading']} (p.{sec['pages'][0]}-{sec['pages'][-1]})\n{sec['text']}\n"
            if used_chars + len(chunk) <= max_chars:
                selected.append(sec)
                used_chars += len(chunk)
            elif bucket is kept_high and used_chars < max_chars - 1500:
                remain = max_chars - used_chars - 200
                selected.append({**sec, "text": sec["text"][:remain] + "\n...[truncated]"})
                used_chars = max_chars

    # Preserve original paper order among selected sections
    order_map = {id(s): idx for idx, s in enumerate(sections)}
    selected.sort(key=lambda s: order_map.get(id(s), 999))

    combined_text = "\n\n".join(
        f"## {s['heading']} (Pages {s['pages'][0]}-{s['pages'][-1]})\n{s['text']}" for s in selected
    )
    detected_lang = detect_language(combined_text)

    return {
        "pdf_path": str(pdf_path),
        "num_pages": num_pages,
        "detected_lang": detected_lang,
        "kept_headings": [s["heading"] for s in selected],
        "dropped_headings": dropped_headings,
        "extracted_text": combined_text,
    }


def analyze_paper_structure(
    client: Any,
    pdf_info: dict[str, Any],
    target_lang: str,
) -> dict[str, Any]:
    """Use the main LLM to distill the full paper into a grounded structured architecture brief."""
    lang_note = (
        "All short diagram labels and running examples MUST be in concise English (top-tier international venue standard)."
        if target_lang == "en"
        else "图内模块标签应使用精炼中文并保留国际通用英文技术缩写（如 LLM、Attention、LoRA 等），不要写长句子。"
    )

    prompt = f"""You are a Senior Area Chair & Visual Science Editor for NeurIPS / CVPR / ICLR / Nature.
Your job is to dissect the following research paper and produce a **100% faithfully grounded** structured analysis
specifically for designing its **main Overview / Framework Figure (Figure 1 / Figure 2)**.

STRICT GROUNDING & FIGURE-1 SUBTRACTION RULES (CRITICAL — PREVENT OVERCROWDING!):
- Extract ONLY real modules, data flows, inputs, and outputs actually present in the paper — NEVER hallucinate.
- **SUBTRACTION IS THE SECRET OF TOP-VENUE FIGURES**: A Figure 1 Overview is a **high-level architectural story**, NOT a code implementation spec or ablation table!
- **HARD INFORMATION BUDGETS**:
  1. `stages`: STRICTLY **3 to 4 stages MAX**.
  2. `modules`: STRICTLY **6 to 9 high-level semantic modules MAX**! Merge low-level sub-operations (e.g., LayerNorm/RMSNorm, Zero-init, Q/K/V/O projections, linear layers, activation functions, clipping ops, residual adds) INSIDE their parent conceptual module rather than listing them as separate modules.
  3. **EXCLUSION BLACKLIST (DO NOT PUT IN FIGURE 1!)**:
     - NO raw hyperparameters or numerical thresholds (e.g., `M=64`, `1:1024`, `t0=0.35`, `t>=0.40`, `0.80`, `0.93`, `Rank 16`, `16 steps`, `lambda=0.1`).
     - NO tensor slice indices or dimensions (e.g., `[224:240]`, `[240:256]`, `256x256`, `512x512` unless one crisp input/output label truly needs it).
     - NO experimental metrics / speed numbers / benchmark counts (e.g., `55.9 FPS`, `40 attributes`, `106 landmarks`, `+6.4%`).
     - NO competing second mega-system: if the paper has training losses, inference solver, AND auxiliary analytics, pick the **SINGLE primary architectural pipeline** as the visual star, and compress auxiliary supervision/solver into at most 1 compact module/badge (or omit secondary analytics entirely!).
- Explicitly separate **baseline / standard / frozen components** from the paper's **core novel contributions**.
- Keep every module `short_label` ultra-concise (1 to 3 words, NEVER a sentence!).
- {lang_note}

ARCHETYPE SELECTION MENU:
{format_archetype_menu()}

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown code fences, no commentary) matching this exact schema:
{{
  "paper_title": "string",
  "domain": "string (e.g., LLM Agents, Computer Vision, Multimodal Learning, Systems, Bio/Med, Benchmark)",
  "one_sentence_story": "One crisp sentence capturing the paper's core scientific narrative",
  "recommended_archetype": "conceptual | framework | pipeline | architecture | taxonomy | teaser",
  "archetype_rationale": "Why this archetype best tells the one-sentence story",
  "novelty_summary": ["Novelty bullet 1", "Novelty bullet 2"],
  "input_spec": "Short label + concrete visual form of input (<= 6 words)",
  "output_spec": "Short label + concrete visual form of output / objective (<= 6 words)",
  "running_example": "A concrete tiny example from the paper (e.g. an input sentence, sample query, or data item) to ground the figure; empty string if none exists in the paper",
  "stages": [
    {{
      "id": "stage_1",
      "title": "Short Stage Name (2-4 words)",
      "purpose": "Brief role (<= 10 words)"
    }}
  ],
  "modules": [
    {{
      "id": "mod_1",
      "short_label": "1-4 Word Label",
      "sub_label": "Optional 1-3 word detail or symbol (empty if unnecessary)",
      "stage_id": "stage_1",
      "role": "novel | baseline | io | foundation",
      "trainable_state": "trainable | frozen | na",
      "visual_object": "Concrete visual representation (e.g. stacked feature maps, token sequence, attention matrix, graph nodes, vector bar, document stack)"
    }}
  ],
  "connections": [
    {{
      "from_id": "mod_1",
      "to_id": "mod_2",
      "flow_type": "main_flow | feedback | gradient | skip | control",
      "label": "Optional 1-2 word edge label or math symbol (empty if obvious)"
    }}
  ],
  "training_vs_inference_split": "Description if the figure should visually separate training and inference with a dashed divider, else empty string",
  "legend_items_suggested": [
    {{"semantic_key": "novel_primary", "meaning": "Short legend text (<= 3 words)"}}
  ]
}}

=== EXTRACTED PAPER CONTENT ===
{pdf_info['extracted_text']}
"""
    raw = client.generate_text(prompt, max_output_tokens=10000, temperature=0.2)
    return _parse_json_strict(raw, fallback_key="paper_analysis")


def render_paper_method_markdown(analysis: dict[str, Any], pdf_info: dict[str, Any]) -> str:
    """Render a clean human-readable Markdown summary of Stage 1."""
    lines = [
        f"# Paper Analysis: {analysis.get('paper_title', 'Untitled')}",
        "",
        f"- **Source PDF**: `{pdf_info.get('pdf_path')}` ({pdf_info.get('num_pages')} pages)",
        f"- **Detected Language**: `{pdf_info.get('detected_lang')}`",
        f"- **Domain**: {analysis.get('domain', 'Unknown')}",
        f"- **One-Sentence Story**: {analysis.get('one_sentence_story', '')}",
        f"- **Recommended Archetype**: `{analysis.get('recommended_archetype', 'pipeline')}` — {analysis.get('archetype_rationale', '')}",
        f"- **Input → Output**: `{analysis.get('input_spec', '')}` → `{analysis.get('output_spec', '')}`",
    ]
    if analysis.get("running_example"):
        lines.append(f"- **Running Example**: `{analysis['running_example']}`")
    if analysis.get("training_vs_inference_split"):
        lines.append(f"- **Train / Inference Split**: {analysis['training_vs_inference_split']}")

    lines.extend(["", "## Core Novelties (MUST Be Highlighted)"])
    for n in analysis.get("novelty_summary", []):
        lines.append(f"- {n}")

    lines.extend(["", "## Modules (Role-Tagged)"])
    for m in analysis.get("modules", []):
        role = m.get("role", "baseline").upper()
        state = m.get("trainable_state", "na")
        state_str = f" [{state}]" if state != "na" else ""
        sub = f" ({m.get('sub_label')})" if m.get("sub_label") else ""
        lines.append(
            f"- **{m.get('id')}** `[{role}]`{state_str}: **{m.get('short_label')}**{sub} "
            f"— stage=`{m.get('stage_id')}`, visual=`{m.get('visual_object')}`"
        )

    lines.extend(["", "## Connections"])
    for c in analysis.get("connections", []):
        lbl = f" label=\"{c.get('label')}\"" if c.get("label") else ""
        lines.append(f"- `{c.get('from_id')}` --({c.get('flow_type')}{lbl})--> `{c.get('to_id')}`")

    lines.extend(
        [
            "",
            "## PDF Section Filtering Log",
            f"- Kept: {', '.join(pdf_info.get('kept_headings', []))}",
            f"- Dropped (noise): {', '.join(pdf_info.get('dropped_headings', [])) or 'None'}",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_json_strict(raw: str, fallback_key: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Extract outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON for {fallback_key}: {exc}\nRaw preview: {raw[:600]}") from exc
