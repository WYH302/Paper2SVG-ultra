"""Top-venue academic visual storytelling grammar: Golden Principles + 6 Flexible Story Archetypes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchetypeSpec:
    key: str
    zh_name: str
    en_name: str
    story_trigger: str
    canvas_width: int
    canvas_height: int
    is_wide_banner: bool
    layout_idioms: str
    prompt_skeleton: str


ARCHETYPES: dict[str, ArchetypeSpec] = {
    "conceptual": ArchetypeSpec(
        key="conceptual",
        zh_name="前后对比式",
        en_name="Two-Panel Visual Comparison",
        story_trigger="The existing/old approach fails here, while our new idea succeeds.",
        canvas_width=1600,
        canvas_height=680,
        is_wide_banner=True,
        layout_idioms=(
            "- Shared top input/problem anchor showing one concrete visual example.\n"
            "- Two equal-sized side-by-side zones below: Left = Existing Approach (muted/desaturated) ending in a concise negative verdict with a crisp cross (✕); "
            "Right = Our Approach (vibrant focal highlight) ending in a crisp positive verdict with a tick (✓).\n"
            "- Show the visual transformation directly with concrete domain imagery/objects rather than empty text boxes."
        ),
        prompt_skeleton=(
            "Wide two-panel comparison figure (~2.35:1) with a shared visual input strip on top, "
            "muted left baseline zone with negative verdict, and vibrant accented right proposed zone with positive verdict."
        ),
    ),
    "framework": ArchetypeSpec(
        key="framework",
        zh_name="分层框架式",
        en_name="Layered Visual System Framework",
        story_trigger="Multiple cooperating components, agents, or modules work together inside a unified system.",
        canvas_width=1600,
        canvas_height=700,
        is_wide_banner=True,
        layout_idioms=(
            "- Organize components by clean spatial bands, soft tint zones, or visual nesting — AVOID nesting 3 layers of harsh dark box borders.\n"
            "- Give key system artifacts (documents, agents, sensors, data structures, models) expressive visual illustrations or mini-plots so the diagram reads like a flagship Nature/NeurIPS visual rather than a dry flowchart.\n"
            "- Place shared foundational layers (LLM backbone, tools, hardware/environment) as clean horizontal base strips or side anchors, with an optional enlarged visual zoom-in callout."
        ),
        prompt_skeleton=(
            "Wide layered scientific system overview (~2.3:1) combining clean soft visual zones, expressive domain illustrations/icons, "
            "and shared foundation base strips without heavy nested wireframe clutter."
        ),
    ),
    "pipeline": ArchetypeSpec(
        key="pipeline",
        zh_name="视觉演进流程式",
        en_name="Visual-First Multi-Stage Pipeline",
        story_trigger="Input transforms progressively through distinct stages (Stage A -> B -> C), where showing the visual state change at each stage tells the story.",
        canvas_width=1600,
        canvas_height=700,
        is_wide_banner=True,
        layout_idioms=(
            "- ADAPT STRUCTURE TO THE REFERENCE STYLE — DO NOT blindly default to 'top 5 boxes + bottom 3 dashed boxes'!\n"
            "- Let the concrete visual progression (e.g., raw input image/3D mesh/slab/data -> intermediate representation/field/tokens -> core algorithmic visual -> final output) breathe prominently across the canvas (~55-65% visual imagery, ~35% crisp typography & connectors).\n"
            "- Use open floating stages, soft tint bands, colored header bars, or subtle step captions `(1) ... (2) ... (3) ...` mirroring the reference folder's actual container & arrow language.\n"
            "- Only add a secondary bottom zoom-in row if a stage truly requires unpacking micro-level mechanics."
        ),
        prompt_skeleton=(
            "Wide visual-first multi-stage scientific pipeline (~2.3:1) where concrete domain visuals (samples, 3D/isometric objects, fields, plots, or bespoke icons) "
            "act as the visual heroes progressing left-to-right across clean open/soft-tinted stages matching the reference style."
        ),
    ),
    "architecture": ArchetypeSpec(
        key="architecture",
        zh_name="模型架构演进式",
        en_name="Visual Neural / Model Architecture",
        story_trigger="The core contribution lies inside the model architecture: layer blocks, dual streams, conditioning, trainable vs frozen components, or training vs inference dynamics.",
        canvas_width=1600,
        canvas_height=720,
        is_wide_banner=True,
        layout_idioms=(
            "- Favor clean borderless or softly outlined layer pills/stacks (like CVPR/ICLR flagship figures) rather than heavy dark wireframe cages.\n"
            "- Show REAL concrete input/intermediate/output specimens (e.g., noisy -> denoised image patches, realistic portraits, feature tensors, spectra, waveforms) directly entering and exiting the architecture.\n"
            "- Clearly separate standard/frozen components (restrained neutral tones + subtle lock badge if relevant) from novel trainable innovations (vivid focal accent + optional flame badge), using whitespace or a subtle divider for Training vs Inference."
        ),
        prompt_skeleton=(
            "Wide flagship neural/model architecture figure (~2.25:1) featuring real visual input/output specimens, "
            "sleek borderless/soft-styled layer stacks, crisp focal accents on novel components, and airy scientific composition."
        ),
    ),
    "taxonomy": ArchetypeSpec(
        key="taxonomy",
        zh_name="全景分类 / 基准图",
        en_name="Visual Taxonomy / Benchmark Landscape",
        story_trigger="A survey, benchmark, empirical study, or unified evaluation suite whose contribution is mapping a design space or multi-metric evaluation landscape.",
        canvas_width=1600,
        canvas_height=740,
        is_wide_banner=True,
        layout_idioms=(
            "- Organize the landscape into clean horizontal bands or symmetric branch panels with color-coded section headers (like flagship benchmark/evaluation infographics).\n"
            "- Populate panels with rich visual artifacts (mini heatmaps, boxplots, curves, sample data cards, model pools, small crisp icons) rather than plain text lists.\n"
            "- Use color inheritance across branches/headers so every category is decoded at a glance."
        ),
        prompt_skeleton=(
            "Wide benchmark/taxonomy landscape (~2.2:1) with clean color-coded bands or symmetric branch panels populated with concrete visual artifacts, "
            "mini-charts, and uniform icon vocabulary."
        ),
    ),
    "teaser": ArchetypeSpec(
        key="teaser",
        zh_name="完整示例图",
        en_name="Worked-Example / Visual Teaser",
        story_trigger="A brand-new task, interactive agent, or visual paradigm best understood by walking through a concrete end-to-end visual example.",
        canvas_width=1440,
        canvas_height=900,
        is_wide_banner=False,
        layout_idioms=(
            "- Intentionally uses a taller ~1.6:1 canvas to give worked examples or multi-turn visual comparisons generous breathing room.\n"
            "- Pair concrete visual context on top with narrow muted baseline(s) on the left and a wide, visually rich proposed method walkthrough on the right.\n"
            "- Use consistent color-coded callouts and expressive minimal character/domain illustrations."
        ),
        prompt_skeleton=(
            "Rich worked-example academic teaser (~1.6:1) showcasing a concrete end-to-end visual example with muted baselines contrasted against a spacious, vividly illustrated proposed workflow."
        ),
    ),
}


GOLDEN_8_PRINCIPLES_BLOCK = """
=== TOP-VENUE ACADEMIC FIGURE ART DIRECTION (GOLDEN PRINCIPLES) ===
1. VISUAL HEROES FIRST — NEVER DRAW A DRY VISIO FLOWCHART!
   - A flagship CVPR / SIGGRAPH / NeurIPS / Nature figure is a **visual scientific illustration**, NOT a grid of empty text boxes connected by thin black lines!
   - Concrete domain visuals — photorealistic sample images (faces, medical/stone/scene photos), 3D/isometric objects (meshes, voxels, optical setups, tensor volumes), real signal plots/heatmaps/spectra, or cohesive custom scientific icons — MUST be LARGE, PROMINENT visual heroes occupying ~50-65% of the visual weight!
   - AVOID "Box-in-Box-in-Box" clutter: let hero visuals float cleanly on white space or sit inside soft borderless / subtly tinted panels. Use borders sparingly and purposefully!
2. FAITHFULLY MIRROR THE REFERENCE FOLDER'S VISUAL ARTISTRY ("借骨架与画风，不借画中内容"):
   - Follow the reference images' ACTUAL visual craft: if references use borderless pastel pills (CVPR), 3D shaded objects + wide colored arrows (SIGGRAPH), expressive icon-driven scenes + bold header bars (AutoFigure-Edit), or multi-band colorful infographics with mini-charts (AIA), **EMBRACE THAT EXACT VISUAL LANGUAGE**!
   - NEVER flatten a vibrant/illustrated/3D reference style into a rigid blue-copper-grey wireframe flowchart!
   - STRICT CONTENT ISOLATION: NEVER copy any words, datasets, or modules from reference images.
3. STRUCTURE >> SENTENCES:
   - Never write prose sentences inside boxes. Use crisp 1-4 word phrases anchored neatly to visual objects.
4. HIGHLIGHT INNOVATION CLEARLY:
   - Give standard/baseline/frozen components restrained neutral styling, and make the paper's CORE NOVEL CONTRIBUTIONS visually pop using the reference-aligned focal accent hue(s), richer visual detail, or subtle highlight elevation.
5. TYPOGRAPHY UNITY & TWO-COLUMN PRINT READABILITY:
   - Strictly ONE clean sans-serif font family across the entire figure (Inter / Helvetica / Arial). NEVER mix Serif and Sans-Serif; NEVER use Comic Sans MS.
   - Maintain a crisp 3-tier hierarchy (Bold section titles, Semi-bold module labels, Clean readable annotations >= 7pt when shrunk to 7-inch two-column print width).
6. HARMONIOUS PALETTE & COGNITIVE CONSISTENCY:
   - Use the reference-distilled scientific palette cleanly (typically 2-5 harmonious hues with clear semantic roles — never garish neon clutter).
   - Same color = same semantic role; same arrow style = same flow type; unified icon/illustration language throughout.
7. GENEROUS WHITESPACE & CLEAN FLOW:
   - Preserve 20-30% clean breathing space, clear reading order, and arrows (bold colored block arrows, smooth curves, or clean orthogonal connectors matching the reference craft) that NEVER slice through text.
8. ASPECT RATIO DISCIPLINE:
   - Default to sleek panoramic wide banners (~2.1:1 to 2.35:1) for top-of-page two-column placement (break only for Worked-Example Teasers).
""".strip()


def format_archetype_menu() -> str:
    """Render concise decision menu for the Stage 1 / Stage 3 archetype selector."""
    lines = ["Choose the SINGLE BEST archetype based on the paper's one-sentence story:"]
    for idx, spec in enumerate(ARCHETYPES.values(), 1):
        ratio = f"{spec.canvas_width}x{spec.canvas_height} ({spec.canvas_width/spec.canvas_height:.2f}:1)"
        lines.append(
            f"{idx}. `{spec.key}` ({spec.zh_name} / {spec.en_name}) | Canvas: {ratio}\n"
            f"   - Best when: {spec.story_trigger}\n"
            f"   - Idioms:\n{spec.layout_idioms}"
        )
    return "\n\n".join(lines)


def get_archetype(key: str) -> ArchetypeSpec:
    normalized = (key or "pipeline").strip().lower()
    if normalized in ARCHETYPES:
        return ARCHETYPES[normalized]
    for k, spec in ARCHETYPES.items():
        if k in normalized:
            return spec
    return ARCHETYPES["pipeline"]
