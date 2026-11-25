"""Automated unit & integration tests for Paper2SVG-ultra (v1.1.0: zero-margin ruler, adaptive density, OOB linter, Critic retention, & primary style winner)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pymupdf
from fastapi.testclient import TestClient
from PIL import Image

import run as cli_run
from paper2svg.archetypes import ARCHETYPES, get_archetype
from paper2svg.architect import extract_exact_text_inventory, extract_prompt_body
from paper2svg.asset_extractor import (
    ExtractedAsset,
    dehydrate_svg_assets,
    hydrate_svg_assets,
    inject_missing_assets,
    render_coordinate_ruler_overlay,
    smart_remove_border_background,
    snap_bbox_to_foreground,
    strip_edge_text_bleed,
    verify_assets_preserved,
    _encode_pil_png_data_url,
)
from paper2svg.client import ResponsesAPIClient
from paper2svg.config import Settings, load_settings, resolve_density_mode, sanitize_font_family
from paper2svg.draft_generator import compute_svg_viewbox
from paper2svg.pdf_parser import extract_pdf_sections
from paper2svg.style_extractor import discover_reference_images, extract_style_dna
from paper2svg.svg_critic import check_critic_content_retention
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

        # Test zero-margin 1:1 coordinate ruler overlay (exact identical dimensions!)
        draft_path = td_path / "draft.png"
        crop_src.save(draft_path)
        ruler_path = render_coordinate_ruler_overlay(draft_path, td_path / "ruler.png")
        assert ruler_path.is_file()
        with Image.open(ruler_path) as ruler_im:
            assert ruler_im.size == crop_src.size, f"Expected 1:1 ruler size {crop_src.size}, got {ruler_im.size}"

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


def test_strip_edge_text_bleed_preserves_graphic_and_removes_caption_sliver() -> None:
    # Build a 100x80 white image with a main card box (rows 5..64) + 4px white gap + 5px grazed text sliver at bottom (rows 75..79)
    arr = np.full((80, 100, 3), 255, dtype=np.uint8)
    arr[5:65, 10:90] = [30, 30, 30]   # main graphic
    arr[12:58, 16:84] = [245, 245, 245]
    arr[75:80, 20:70] = [20, 20, 20]  # grazed caption sliver touching bottom edge
    img_with_bleed = Image.fromarray(arr, mode="RGB")

    cleaned, trims = strip_edge_text_bleed(img_with_bleed)
    assert trims[3] >= 10, f"Expected bottom sliver trimmed, got trims={trims}"
    assert cleaned.height <= 66

    # Verify a clean 3-row stacked waveform graphic is NOT altered
    wf = np.full((120, 100, 3), 255, dtype=np.uint8)
    wf[10:35, 10:90] = [90, 70, 180]
    wf[45:70, 10:90] = [50, 110, 200]
    wf[80:105, 10:90] = [210, 130, 40]
    wf_img = Image.fromarray(wf, mode="RGB")
    wf_out, wf_trims = strip_edge_text_bleed(wf_img)
    assert wf_trims == (0, 0, 0, 0) and wf_out.size == wf_img.size

    # Verify a closed 4-sided picture frame (e.g. mini-screen thumbnail) is NOT peeled, while an open
    # C-shaped grazed card border (left+top+bottom open on right) IS peeled cleanly!
    closed_frame = np.full((64, 84, 3), 255, dtype=np.uint8)
    closed_frame[:3, :] = 20
    closed_frame[-3:, :] = 20
    closed_frame[:, :3] = 20
    closed_frame[:, -3:] = 20
    closed_frame[14:50, 16:68] = 80
    cf_out, cf_trims = strip_edge_text_bleed(Image.fromarray(closed_frame, mode="RGB"))
    assert cf_trims == (0, 0, 0, 0) and cf_out.size == (84, 64)

    open_c = np.full((74, 77, 3), 255, dtype=np.uint8)
    open_c[:3, :] = 20
    open_c[-3:, :] = 20
    open_c[:, :3] = 20
    open_c[14:60, 18:56] = 30    # inner doc icon
    open_c[28:32, 72:77] = 20    # stray 'T' sliver on right edge
    oc_out, oc_trims = strip_edge_text_bleed(Image.fromarray(open_c, mode="RGB"))
    assert oc_trims[0] >= 12 and oc_trims[2] >= 10, f"Expected open C-frame + right sliver stripped, got {oc_trims}"
    assert oc_out.width <= 45 and oc_out.height <= 50


def test_pixel_snapper_and_native_1to1_viewbox() -> None:
    # Build a 200x120 white canvas with a dark graphic rectangle at x=40..150, y=25..95
    arr = np.full((120, 200, 3), 255, dtype=np.float32)
    arr[25:95, 40:150] = [40, 40, 40]
    arr[30:90, 45:145] = [140, 140, 140]
    # Simulate a slightly too-tight VLM bbox that sliced 6px off the left & 5px off the bottom
    nx1, ny1, nx2, ny2 = snap_bbox_to_foreground(arr, 46, 25, 150, 90, max_expand=12)
    assert nx1 <= 41, f"Expected left edge snapped back to <=41, got {nx1}"
    assert ny2 >= 94, f"Expected bottom edge snapped out to >=94, got {ny2}"

    # Verify compute_svg_viewbox preserves exact 1:1 native pixel dimensions
    img = Image.new("RGB", (2016, 756), (255, 255, 255))
    assert compute_svg_viewbox(img, get_archetype("framework")) == (2016, 756)


def test_adaptive_density_oob_and_critic_retention_guard() -> None:
    assert resolve_density_mode("auto", 20) == "concise"
    assert resolve_density_mode("auto", 54) == "detailed"
    assert resolve_density_mode("concise", 54) == "concise"
    assert resolve_density_mode("detailed", 10) == "detailed"

    # 1. Dense 42-label systems SVG should pass in `auto` (resolving to `detailed`) with 14px labels and 0 false overcrowding errors
    inventory = [f"Module {i:02d}" for i in range(1, 43)]
    text_nodes = []
    for idx, lbl in enumerate(inventory):
        col, row = idx % 6, idx // 6
        x = 120 + col * 240
        y = 70 + row * 75
        text_nodes.append(
            f'<rect x="{x - 95}" y="{y - 18}" width="190" height="36" rx="6" fill="#DEE5F1" stroke="#171717"/>'
            f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="central" '
            f'font-family="Inter, Arial, sans-serif" font-size="14">{lbl}</text>'
        )
    dense_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 610" width="1600" height="610">'
        '<rect x="0" y="0" width="1600" height="610" fill="#FFFFFF"/>'
        + "".join(text_nodes)
        + "</svg>"
    )
    rep_dense = lint_svg(dense_svg, exact_text_inventory=inventory, expected_viewbox=(1600, 610), density="auto")
    assert rep_dense.stats["density_mode"] == "detailed"
    assert rep_dense.passed_hard_checks, f"Unexpected errors on valid dense SVG: {rep_dense.errors}"
    assert rep_dense.stats["inventory_coverage_ratio"] == 1.0

    # 2. Canvas clipping & textLength squashing checks (while ignoring <defs>/<clipPath> and matching multi-line stacked <text>)
    clipped_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 610" width="1600" height="610">
      <defs><clipPath id="cp"><rect x="0" y="0" width="2000" height="800"/></clipPath></defs>
      <rect x="0" y="0" width="1600" height="610" fill="#FFFFFF"/>
      <text x="120" y="60" font-family="Inter, Arial, sans-serif" font-size="16">Hybrid Plan</text>
      <text x="120" y="80" font-family="Inter, Arial, sans-serif" font-size="16">Generation</text>
      <text x="300" y="80" font-family="Inter, Arial, sans-serif" font-size="16" textLength="60" lengthAdjust="spacingAndGlyphs">Squashed Label</text>
      <rect id="overflow_card" x="1520" y="420" width="130" height="120" rx="8" fill="#F7E5D8" stroke="#171717"/>
      <text x="1585" y="575" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="18">Verified Output</text>
    </svg>"""
    rep_oob = lint_svg(
        clipped_svg,
        exact_text_inventory=["Hybrid Plan Generation", "Verified Output"],
        expected_viewbox=(1600, 610),
        density="detailed",
    )
    assert not rep_oob.passed_hard_checks
    assert rep_oob.stats["out_of_bounds_count"] >= 2
    assert rep_oob.stats["inventory_coverage_ratio"] == 1.0  # consecutive stacked <text> matched "Hybrid Plan Generation"!
    assert any("Canvas clipping" in e for e in rep_oob.errors)
    assert any("textLength" in e for e in rep_oob.errors)

    # 3. Critic semantic retention guard: grouping 42 labels into 14 multi-line <tspan> elements MUST BE ACCEPTED
    grouped_nodes = []
    for g_idx in range(0, len(inventory), 3):
        grp = inventory[g_idx : g_idx + 3]
        col, row = (g_idx // 3) % 5, (g_idx // 3) // 5
        x, y = 160 + col * 280, 90 + row * 160
        tspans = "".join(f'<tspan x="{x}" dy="{0 if j == 0 else 18}">{lbl}</tspan>' for j, lbl in enumerate(grp))
        grouped_nodes.append(
            f'<text x="{x}" y="{y}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="14">{tspans}</text>'
        )
    grouped_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 610" width="1600" height="610">'
        '<rect x="0" y="0" width="1600" height="610" fill="#FFFFFF"/>'
        + "".join(grouped_nodes)
        + "</svg>"
    )
    rep_grouped = lint_svg(grouped_svg, exact_text_inventory=inventory, expected_viewbox=(1600, 610), density="detailed")
    assert rep_grouped.stats["text_elements"] == 14  # < 50% of 42 <text> nodes!
    ok_keep, msg_keep = check_critic_content_retention(rep_dense, rep_grouped, inventory)
    assert ok_keep, f"Expected <tspan> consolidation to be accepted, got rejection: {msg_keep}"

    # 4. Destructive label deletion (dropping 75% of inventory) MUST BE REJECTED
    pruned_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 610" width="1600" height="610">'
        '<rect x="0" y="0" width="1600" height="610" fill="#FFFFFF"/>'
        + "".join(text_nodes[:8])
        + "</svg>"
    )
    rep_pruned = lint_svg(pruned_svg, exact_text_inventory=inventory, expected_viewbox=(1600, 610), density="detailed")
    ok_prune, msg_prune = check_critic_content_retention(rep_dense, rep_pruned, inventory)
    assert not ok_prune and "dropped" in msg_prune.lower()


def test_primary_ref_and_single_style_winner_and_cli_flags() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        img1 = td_path / "big_ref.png"
        img2 = td_path / "pinned_ref.png"
        Image.new("RGB", (1200, 600), (240, 240, 240)).save(img1)
        Image.new("RGB", (600, 300), (220, 230, 250)).save(img2)

        # Pinning primary_ref strictly selects ONLY that single image
        picked = discover_reference_images(td_path, primary_ref=img2)
        assert picked == [img2]

        # Multi-image extract_style_dna pins strictly to a SINGLE Primary Style Winner (length 1)
        class _DummyClient:
            def generate_multimodal(self, contents, **kwargs):
                return json.dumps(
                    {
                        "visual_paradigm": "soft_card_editorial",
                        "primary_anchor_reason": "Cleanest flat 2D layout without clutter.",
                        "top_anchor_indices": [2, 1],
                    }
                )

        dna = extract_style_dna(_DummyClient(), [img1, img2], target_archetype="framework")
        assert dna["top_anchor_indices"] == [2]
        assert dna["anchor_files"] == [str(img2)]
        assert "Cleanest" in dna.get("primary_anchor_reason", "")

    # CLI flags check
    parser = cli_run.build_parser()
    parsed = parser.parse_args(["--paper", "a.pdf", "--density", "detailed", "--primary-ref", "ref.png"])
    assert parsed.density == "detailed"
    assert parsed.primary_ref == Path("ref.png")


def test_fastapi_endpoints_and_svgedit() -> None:
    client = TestClient(app)
    r_home = client.get("/")
    assert r_home.status_code == 200 and "Paper2SVG-ultra" in r_home.text
    r_cfg = client.get("/api/config")
    assert r_cfg.status_code == 200
    cfg = r_cfg.json()
    assert cfg["main_model"]
    assert cfg["svg_mode"] in ("hybrid_rich", "pure_vector")
    assert cfg["density"] in ("auto", "concise", "detailed")
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
    test_strip_edge_text_bleed_preserves_graphic_and_removes_caption_sliver()
    test_adaptive_density_oob_and_critic_retention_guard()
    test_primary_ref_and_single_style_winner_and_cli_flags()
    test_fastapi_endpoints_and_svgedit()
    print("ALL v1.1.0 AUTOMATED TESTS PASSED!")
