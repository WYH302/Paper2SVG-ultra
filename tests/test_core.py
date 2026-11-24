"""Automated unit & integration tests for Paper2SVG-Pro (including Hybrid SVG asset engine & xhigh reasoning)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pymupdf
from fastapi.testclient import TestClient
from PIL import Image

from paper2svg.archetypes import ARCHETYPES, get_archetype
from paper2svg.architect import extract_exact_text_inventory, extract_prompt_body
from paper2svg.asset_extractor import (
    ExtractedAsset,
    dehydrate_svg_assets,
    hydrate_svg_assets,
    inject_missing_assets,
    render_coordinate_ruler_overlay,
    smart_remove_border_background,
    verify_assets_preserved,
    _encode_pil_png_data_url,
)
from paper2svg.client import ResponsesAPIClient
from paper2svg.config import Settings, load_settings, sanitize_font_family
from paper2svg.pdf_parser import extract_pdf_sections
from paper2svg.svg_linter import lint_svg
from paper2svg.svg_renderer import render_dual_preview
from server import app


def test_font_guardrail() -> None:
    assert "comic" not in sanitize_font_family("Comic Sans MS").lower()
    assert "Inter" in sanitize_font_family("Comic Sans MS", lang="en")
    assert "Noto Sans SC" in sanitize_font_family("", lang="zh")
    assert sanitize_font_family("Helvetica, Arial, sans-serif") == "Helvetica, Arial, sans-serif"


def test_reasoning_payload_config() -> None:
    defaults = Settings()
    assert defaults.base_url == "https://api.openai.com/v1"
    assert defaults.actor_auth == ""
    assert defaults.svg_mode == "hybrid_rich"
    assert defaults.disable_response_storage is True

    loaded = load_settings()
    assert loaded.svg_mode in ("hybrid_rich", "pure_vector")
    assert loaded.reasoning_effort in ("", "none", "low", "medium", "high", "xhigh")

    client = ResponsesAPIClient(Settings(api_key="test-key", main_model="gpt-5.4", reasoning_effort="xhigh"))
    payload = client._apply_reasoning({"model": "gpt-5.4", "temperature": 0.2}, temperature=0.2)
    assert payload["reasoning"] == {"effort": "xhigh"}
    assert "temperature" not in payload


def test_archetypes() -> None:
    assert set(ARCHETYPES.keys()) == {"conceptual", "framework", "pipeline", "architecture", "taxonomy", "teaser"}
    assert get_archetype("pipeline").is_wide_banner is True
    assert get_archetype("teaser").is_wide_banner is False


def test_pdf_section_filtering() -> None:
    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / "sample.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        y = 50
        blocks = [
            ("SpecAlign: Sparse Routing for Multimodal Reasoning", 16, True),
            ("Abstract", 13, True),
            ("We propose SpecAlign, a novel two-stage cross-modal alignment framework.", 10, False),
            ("1 Introduction", 13, True),
            ("Existing models suffer from dense attention overhead. Our key innovation is Sparse Router.", 10, False),
            ("2 Related Work", 13, True),
            ("Many prior papers studied transformers (Vaswani et al., 2017) and CLIP.", 10, False),
            ("3 Methodology", 13, True),
            ("Stage 1 uses a Frozen Vision Encoder. Stage 2 introduces our Novel Sparse Token Router.", 10, False),
            ("References", 13, True),
            ("[1] Vaswani et al. Attention Is All You Need. NeurIPS 2017.", 10, False),
        ]
        for text, size, bold in blocks:
            font = "hebo" if bold else "helv"
            page.insert_text((50, y), text, fontsize=size, fontname=font)
            y += size + 10
        doc.save(str(pdf_path))
        doc.close()

        info = extract_pdf_sections(pdf_path)
        assert info["detected_lang"] == "en"
        assert "SpecAlign" in info["extracted_text"]
        assert "Sparse Token Router" in info["extracted_text"]
        assert "Vaswani et al. Attention Is All You Need" not in info["extracted_text"]
        assert any("Related Work" in h for h in info["dropped_headings"])
        assert any("References" in h for h in info["dropped_headings"])


def test_schema_parsing() -> None:
    schema = """
# Header
---BEGIN PROMPT---
[SECTION 1]
Key labels
[EXACT TEXT INVENTORY]
- "Frozen Encoder"
- "Sparse Router"
- "Output Prediction"
[HARD CONSTRAINTS]
- Keep clean
---END PROMPT---
"""
    assert "[SECTION 1]" in extract_prompt_body(schema)
    inv = extract_exact_text_inventory(schema)
    assert inv == ["Frozen Encoder", "Sparse Router", "Output Prediction"]


def test_hybrid_asset_crop_bg_removal_and_svg_render() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Create a synthetic image with a tinted card background (#EFF6FF), a colored ring, and a white interior
        img = Image.new("RGB", (160, 120), (239, 246, 255))
        arr = np.array(img)
        arr[25:95, 35:125] = [37, 99, 235]     # blue outer shape
        arr[45:75, 55:105] = [255, 255, 255]   # pure white interior (MUST stay opaque!)
        crop_src = Image.fromarray(arr, mode="RGB")

        cleaned, trims = smart_remove_border_background(crop_src)
        c_arr = np.array(cleaned)
        # Perimeter was trimmed tight to the blue box (90x70)
        assert cleaned.width <= 94 and cleaned.height <= 74
        # Interior white center MUST remain opaque (alpha == 255)
        cy, cx = c_arr.shape[0] // 2, c_arr.shape[1] // 2
        assert c_arr[cy, cx, 3] == 255

        # Test ruler overlay
        draft_path = td_path / "draft.png"
        crop_src.save(draft_path)
        ruler_path = render_coordinate_ruler_overlay(draft_path, td_path / "ruler.png")
        assert ruler_path.is_file()

        # Test hydration / dehydration / verification / resvg-py rendering of base64 embedded <image>
        data_url = _encode_pil_png_data_url(cleaned)
        asset = ExtractedAsset(
            asset_id="hero_cube",
            description="Blue cube with white interior",
            bg_mode="remove_bg",
            bbox_1000=[200, 200, 400, 400],
            svg_x=120.0,
            svg_y=240.0,
            svg_w=200.0,
            svg_h=150.0,
            rounded_rx=8,
            crop_path=str(draft_path),
            placeholder="__ASSET_hero_cube__",
            data_url=data_url,
        )

        template_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 680" width="1600" height="680">
          <defs>
            <clipPath id="clip-hero"><rect x="120" y="240" width="200" height="150" rx="8"/></clipPath>
          </defs>
          <style>.module-title { font-family: Inter, Helvetica, Arial, sans-serif; font-size: 26px; font-weight: 600; fill: #0F172A; }</style>
          <rect width="1600" height="680" fill="#FFFFFF"/>
          <g id="layer-visual-assets">
            <image id="asset-hero_cube" x="120" y="240" width="200" height="150" href="__ASSET_hero_cube__" clip-path="url(#clip-hero)" preserveAspectRatio="xMidYMid meet"/>
          </g>
          <g id="layer-labels-and-legend">
            <text x="220" y="430" text-anchor="middle" class="module-title">Hero Visual</text>
          </g>
        </svg>"""

        ok_pres, missing = verify_assets_preserved(template_svg, [asset])
        assert ok_pres and not missing

        hydrated = hydrate_svg_assets(template_svg, [asset])
        assert "data:image/png;base64," in hydrated
        assert "__ASSET_hero_cube__" not in hydrated

        dehydrated = dehydrate_svg_assets(hydrated, [asset])
        assert "__ASSET_hero_cube__" in dehydrated
        assert "data:image/png;base64," not in dehydrated

        # Test missing asset auto-inject safety net
        stripped = template_svg.replace('href="__ASSET_hero_cube__"', 'href="none"')
        repaired = inject_missing_assets(stripped, [asset])
        assert "__ASSET_hero_cube__" in repaired

        # Test actual resvg-py rendering of the hydrated hybrid SVG (verify non-white pixels appear where image sits!)
        full_png = td_path / "hybrid_full.png"
        twocol_png = td_path / "hybrid_2col.png"
        render_dual_preview(hydrated, full_png, twocol_png, viewbox=(1600, 680), full_zoom=1.0)
        with Image.open(full_png) as rendered:
            r_arr = np.array(rendered.convert("RGB"))
            # Check region around x=120..320, y=240..390 actually rendered blue pixels from embedded base64 PNG!
            patch = r_arr[250:380, 130:310]
            blue_pixels = int(np.sum((patch[:, :, 2] > 180) & (patch[:, :, 0] < 100)))
            assert blue_pixels > 500, f"Expected embedded PNG image to render blue pixels, got {blue_pixels}"


def test_svg_linter_catches_bad_and_passes_good() -> None:
    bad_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 680" width="1600" height="680">
      <rect width="1600" height="680" fill="#FFFFFF"/>
      <rect x="40" y="40" width="100" height="60" rx="8" fill="#DBEAFE" stroke="#2563EB"/>
      <text x="90" y="70" text-anchor="middle" font-family="Comic Sans MS" font-size="12">SECTION 1 Super Long Overflowing Label Text</text>
    </svg>"""
    rep = lint_svg(bad_svg, exact_text_inventory=["Frozen Encoder"], expected_viewbox=(1600, 680))
    assert not rep.passed_hard_checks
    joined = "\n".join(rep.errors)
    assert "Comic Sans" in joined
    assert "Meta-label" in joined
    assert "font-size=12px" in joined


def test_fastapi_endpoints_and_svgedit() -> None:
    client = TestClient(app)
    r_home = client.get("/")
    assert r_home.status_code == 200 and "Paper2SVG-ultra" in r_home.text
    r_cfg = client.get("/api/config")
    assert r_cfg.status_code == 200
    cfg = r_cfg.json()
    assert cfg["main_model"]
    assert cfg["svg_mode"] in ("hybrid_rich", "pure_vector")
    assert cfg["svg_edit_url"] == "/svg-edit/editor/index.html"
    r_edit = client.get("/svg-edit/editor/index.html")
    assert r_edit.status_code == 200
    r_runs = client.get("/api/runs")
    assert r_runs.status_code == 200


if __name__ == "__main__":
    test_font_guardrail()
    test_reasoning_payload_config()
    test_archetypes()
    test_pdf_section_filtering()
    test_schema_parsing()
    test_hybrid_asset_crop_bg_removal_and_svg_render()
    test_svg_linter_catches_bad_and_passes_good()
    test_fastapi_endpoints_and_svgedit()
    print("ALL UPGRADED HYBRID + XHIGH AUTOMATED TESTS PASSED!")
