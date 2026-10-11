from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "architecture"
SAMPLE_IMAGE = (
    PROJECT_ROOT.parent
    / "Multi-Sensor Fingerprint Database"
    / "AES2501"
    / "001"
    / "1_1.bmp"
)

INK = "#172033"
MUTED = "#5C667A"
GRID = "#DDE2EA"
PANEL = "#F7F9FC"
WHITE = "#FFFFFF"
COLORS = {
    "stem": "#3B82B8",
    "residual": "#277DA1",
    "mobile": "#2A9D8F",
    "attention": "#E9A23B",
    "repvit": "#B565A7",
    "convnext": "#4C78A8",
    "pool": "#6B7280",
    "head": "#D56727",
    "norm": "#B68D00",
    "input": "#7A8495",
}


def blend(color: str, toward: str = WHITE, amount: float = 0.75) -> str:
    source = to_rgb(color)
    target = to_rgb(toward)
    rgb = tuple(source[i] * (1 - amount) + target[i] * amount for i in range(3))
    return "#%02x%02x%02x" % tuple(round(channel * 255) for channel in rgb)


def darken(color: str, amount: float = 0.22) -> str:
    return blend(color, INK, amount)


@dataclass(frozen=True)
class Stage:
    title: str
    operation: str
    spatial: int
    channels: int
    repeat: int
    kind: str


@dataclass(frozen=True)
class BlockStep:
    title: str
    detail: str
    kind: str


@dataclass(frozen=True)
class ModelDiagram:
    key: str
    title: str
    subtitle: str
    alias: str
    parameters: int
    pretrained: str
    top1: float
    stages: tuple[Stage, ...]
    core_title: str
    core: tuple[BlockStep, ...]
    shortcut_label: str
    note: str


MODELS = (
    ModelDiagram(
        key="resnet18",
        title="ResNet-18",
        subtitle="Deep residual learning baseline",
        alias="resnet18.a1_in1k",
        parameters=11_502_780,
        pretrained="ImageNet-1K",
        top1=98.048,
        stages=(
            Stage("Stem", "7x7 conv /2", 112, 64, 1, "stem"),
            Stage("MaxPool", "3x3 /2", 56, 64, 1, "pool"),
            Stage("Stage 1", "BasicBlock", 56, 64, 2, "residual"),
            Stage("Stage 2", "BasicBlock /2", 28, 128, 2, "residual"),
            Stage("Stage 3", "BasicBlock /2", 14, 256, 2, "residual"),
            Stage("Stage 4", "BasicBlock /2", 7, 512, 2, "residual"),
            Stage("GAP", "global average", 1, 512, 1, "pool"),
            Stage("Classifier", "linear", 1, 636, 1, "head"),
        ),
        core_title="Basic residual block",
        core=(
            BlockStep("3x3 Conv", "C -> C", "residual"),
            BlockStep("BN + ReLU", "activation", "norm"),
            BlockStep("3x3 Conv", "C -> C", "residual"),
            BlockStep("BN", "normalization", "norm"),
            BlockStep("Add + ReLU", "residual output", "head"),
        ),
        shortcut_label="identity or 1x1 projection shortcut",
        note="Two 3x3 convolutions are optimized through an additive identity path.",
    ),
    ModelDiagram(
        key="mobilenetv3_large",
        title="MobileNetV3-Large",
        subtitle="Hardware-aware mobile inverted residual network",
        alias="mobilenetv3_large_100.ra_in1k",
        parameters=5_016_748,
        pretrained="ImageNet-1K · RandAugment",
        top1=98.628,
        stages=(
            Stage("Stem", "3x3 conv /2", 112, 16, 1, "stem"),
            Stage("DSConv", "k3", 112, 16, 1, "mobile"),
            Stage("Bneck", "k3 /2", 56, 24, 2, "mobile"),
            Stage("Bneck + SE", "k5 /2", 28, 40, 3, "attention"),
            Stage("Bneck", "k3 /2", 14, 80, 4, "mobile"),
            Stage("Bneck + SE", "k3", 14, 112, 2, "attention"),
            Stage("Bneck + SE", "k5 /2", 7, 160, 3, "attention"),
            Stage("Features", "1x1 conv", 7, 960, 1, "stem"),
            Stage("GAP", "global average", 1, 960, 1, "pool"),
            Stage("Conv head", "1x1 conv", 1, 1280, 1, "stem"),
            Stage("Classifier", "linear", 1, 636, 1, "head"),
        ),
        core_title="Mobile inverted residual block",
        core=(
            BlockStep("Expand", "1x1 pointwise", "mobile"),
            BlockStep("Depthwise", "3x3 or 5x5", "mobile"),
            BlockStep("SE", "selected stages", "attention"),
            BlockStep("Project", "1x1 linear", "mobile"),
            BlockStep("Add", "when shape matches", "head"),
        ),
        shortcut_label="shortcut when stride = 1 and input/output channels match",
        note="Exact head order: 160->960 features, GAP, 960->1280 pointwise conv, 1280->636 linear.",
    ),
    ModelDiagram(
        key="efficientnet_b0",
        title="EfficientNet-B0",
        subtitle="Compound-scaled MBConv convolutional network",
        alias="efficientnet_b0.ra_in1k",
        parameters=4_822_264,
        pretrained="ImageNet-1K · RandAugment",
        top1=98.470,
        stages=(
            Stage("Stem", "3x3 conv /2", 112, 32, 1, "stem"),
            Stage("MBConv1", "k3", 112, 16, 1, "mobile"),
            Stage("MBConv6", "k3 /2", 56, 24, 2, "mobile"),
            Stage("MBConv6", "k5 /2", 28, 40, 2, "attention"),
            Stage("MBConv6", "k3 /2", 14, 80, 3, "mobile"),
            Stage("MBConv6", "k5", 14, 112, 3, "attention"),
            Stage("MBConv6", "k5 /2", 7, 192, 4, "attention"),
            Stage("MBConv6", "k3", 7, 320, 1, "mobile"),
            Stage("Features", "1x1 conv", 7, 1280, 1, "stem"),
            Stage("GAP", "dropout 0.2", 1, 1280, 1, "pool"),
            Stage("Classifier", "linear", 1, 636, 1, "head"),
        ),
        core_title="MBConv with squeeze-excitation",
        core=(
            BlockStep("Expand", "1x1 · up to 6C", "mobile"),
            BlockStep("Depthwise", "3x3 or 5x5", "mobile"),
            BlockStep("SE", "ratio 0.25", "attention"),
            BlockStep("Project", "1x1 linear", "mobile"),
            BlockStep("Add", "if shape matches", "head"),
        ),
        shortcut_label="identity shortcut for stride = 1 and matching channels",
        note="Seven compound-scaled stages use SiLU activations and squeeze-excitation.",
    ),
    ModelDiagram(
        key="repvit_m2_3",
        title="RepViT-M2.3",
        subtitle="Mobile CNN reinterpreted through ViT design principles",
        alias="repvit_m2_3.dist_450e_in1k",
        parameters=22_813_188,
        pretrained="ImageNet-1K · distilled 450 epochs",
        top1=98.101,
        stages=(
            Stage("Stem", "3x3 /2 x2", 56, 80, 1, "stem"),
            Stage("Stage 1", "RepViT block", 56, 80, 6, "repvit"),
            Stage("Stage 2", "downsample", 28, 160, 6, "repvit"),
            Stage("Stage 3", "downsample", 14, 320, 34, "repvit"),
            Stage("Stage 4", "downsample", 7, 640, 2, "repvit"),
            Stage("GAP", "global average", 1, 640, 1, "pool"),
            Stage("Classifier", "BN + linear", 1, 636, 1, "head"),
        ),
        core_title="RepViT block",
        core=(
            BlockStep("RepVGGDW", "3x3 DW + 1x1 + Id", "repvit"),
            BlockStep("SE", "alternating blocks", "attention"),
            BlockStep("Expand", "1x1 · C -> 2C", "repvit"),
            BlockStep("GELU", "activation", "norm"),
            BlockStep("Project", "1x1 · 2C -> C", "repvit"),
            BlockStep("Add", "identity", "head"),
        ),
        shortcut_label="residual path; RepVGG branches fuse for inference",
        note="The distilled ImageNet classifier is reset to one BN + Linear 636-class head.",
    ),
    ModelDiagram(
        key="convnextv2_tiny",
        title="ConvNeXt V2-Tiny",
        subtitle="Modern convolutional network with global response normalization",
        alias="convnextv2_tiny.fcmae_ft_in22k_in1k",
        parameters=28_355_580,
        pretrained="FCMAE ImageNet-22K -> ImageNet-1K",
        top1=98.681,
        stages=(
            Stage("Patchify", "4x4 conv /4", 56, 96, 1, "stem"),
            Stage("Stage 1", "ConvNeXt V2", 56, 96, 3, "convnext"),
            Stage("Stage 2", "2x2 downsample", 28, 192, 3, "convnext"),
            Stage("Stage 3", "2x2 downsample", 14, 384, 9, "convnext"),
            Stage("Stage 4", "2x2 downsample", 7, 768, 3, "convnext"),
            Stage("GAP", "global average", 1, 768, 1, "pool"),
            Stage("LayerNorm", "768-D", 1, 768, 1, "norm"),
            Stage("Classifier", "linear", 1, 636, 1, "head"),
        ),
        core_title="ConvNeXt V2 block",
        core=(
            BlockStep("Depthwise", "7x7 convolution", "convnext"),
            BlockStep("LayerNorm", "channels last", "norm"),
            BlockStep("Expand", "linear · C -> 4C", "convnext"),
            BlockStep("GELU + GRN", "global response", "attention"),
            BlockStep("Project", "linear · 4C -> C", "convnext"),
            BlockStep("Add", "identity", "head"),
        ),
        shortcut_label="residual shortcut around the complete block",
        note="GRN adds inter-channel competition after the expanded MLP representation.",
    ),
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def spatial_height(spatial: int) -> float:
    heights = {112: 0.205, 56: 0.175, 28: 0.145, 14: 0.118, 7: 0.096, 1: 0.072}
    return heights[spatial]


def channel_depth(channels: int) -> tuple[float, float]:
    fraction = min(1.0, max(0.0, (math.log2(channels) - 4.0) / 7.0))
    return 0.006 + 0.009 * fraction, 0.006 + 0.006 * fraction


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=MUTED,
            shrinkA=1,
            shrinkB=1,
            zorder=1,
        )
    )


def draw_input(ax: plt.Axes, x: float, y: float, width: float) -> None:
    height = 0.19
    left = x - width / 2
    bottom = y - height / 2
    ax.add_patch(
        FancyBboxPatch(
            (left - 0.004, bottom - 0.006),
            width + 0.008,
            height + 0.012,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            facecolor="#E9EDF3",
            edgecolor="none",
            zorder=1,
        )
    )
    if SAMPLE_IMAGE.exists():
        image = plt.imread(SAMPLE_IMAGE)
        ax.imshow(
            image,
            cmap="gray",
            extent=(left, left + width, bottom, bottom + height),
            aspect="auto",
            zorder=2,
        )
    else:
        ax.add_patch(
            FancyBboxPatch(
                (left, bottom),
                width,
                height,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                facecolor=blend(COLORS["input"], amount=0.82),
                edgecolor=COLORS["input"],
                linewidth=1.0,
                zorder=2,
            )
        )
    ax.add_patch(
        FancyBboxPatch(
            (left, bottom),
            width,
            height,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            facecolor="none",
            edgecolor=INK,
            linewidth=1.0,
            zorder=3,
        )
    )
    ax.text(x, 0.346, "Input", ha="center", va="top", fontsize=7.1, fontweight="bold", color=INK)
    ax.text(x, 0.318, "224 x 224 x 3", ha="center", va="top", fontsize=6.2, color=MUTED)


def draw_volume(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    stage: Stage,
) -> tuple[float, float]:
    color = COLORS[stage.kind]
    edge = darken(color, 0.18)
    height = spatial_height(stage.spatial)
    dx, dy = channel_depth(stage.channels)
    left = x - width / 2
    bottom = y - height / 2
    visible_repeats = min(stage.repeat, 4)

    for layer in range(visible_repeats - 1, -1, -1):
        offset_x = -layer * 0.0042
        offset_y = layer * 0.0042
        lx = left + offset_x
        by = bottom + offset_y
        front = [(lx, by), (lx + width, by), (lx + width, by + height), (lx, by + height)]
        side = [
            (lx + width, by),
            (lx + width + dx, by + dy),
            (lx + width + dx, by + height + dy),
            (lx + width, by + height),
        ]
        top = [
            (lx, by + height),
            (lx + dx, by + height + dy),
            (lx + width + dx, by + height + dy),
            (lx + width, by + height),
        ]
        alpha = 0.68 if layer else 1.0
        ax.add_patch(
            Polygon(front, closed=True, facecolor=blend(color, amount=0.73), edgecolor=edge, linewidth=0.9, alpha=alpha, zorder=3 + layer)
        )
        ax.add_patch(
            Polygon(side, closed=True, facecolor=blend(color, amount=0.48), edgecolor=edge, linewidth=0.8, alpha=alpha, zorder=3 + layer)
        )
        ax.add_patch(
            Polygon(top, closed=True, facecolor=blend(color, amount=0.84), edgecolor=edge, linewidth=0.8, alpha=alpha, zorder=3 + layer)
        )

    shape_label = f"{stage.spatial} x {stage.spatial} x {stage.channels}"
    ax.text(x + dx * 0.35, y + height / 2 + dy + 0.022, shape_label, ha="center", va="bottom", fontsize=6.1, color=MUTED)
    if stage.repeat > 1:
        ax.text(
            x + width / 2 + dx + 0.006,
            y + height / 2 + dy + 0.003,
            f"x{stage.repeat}",
            ha="left",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color=edge,
        )
    ax.text(x, 0.346, stage.title, ha="center", va="top", fontsize=6.8, fontweight="bold", color=INK)
    ax.text(x, 0.318, stage.operation, ha="center", va="top", fontsize=5.9, color=MUTED)
    return left - (visible_repeats - 1) * 0.0042, left + width + dx


def draw_pool_or_head(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    stage: Stage,
) -> tuple[float, float]:
    color = COLORS[stage.kind]
    edge = darken(color, 0.18)
    height = 0.082 if stage.kind == "head" else 0.072
    if stage.kind == "pool":
        radius = min(width * 0.46, 0.026)
        ax.add_patch(Circle((x, y), radius, facecolor=blend(color, amount=0.82), edgecolor=edge, linewidth=1.1, zorder=3))
        ax.text(x, y, "GAP", ha="center", va="center", fontsize=6.2, fontweight="bold", color=INK, zorder=4)
        left, right = x - radius, x + radius
    else:
        left = x - width / 2
        right = x + width / 2
        ax.add_patch(
            FancyBboxPatch(
                (left, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.004,rounding_size=0.007",
                facecolor=blend(color, amount=0.80),
                edgecolor=edge,
                linewidth=1.1,
                zorder=3,
            )
        )
        ax.text(x, y, str(stage.channels), ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK, zorder=4)
    ax.text(x, 0.346, stage.title, ha="center", va="top", fontsize=6.8, fontweight="bold", color=INK)
    ax.text(x, 0.318, stage.operation, ha="center", va="top", fontsize=5.9, color=MUTED)
    return left, right


def draw_pipeline(ax: plt.Axes, model: ModelDiagram) -> None:
    count = len(model.stages) + 1
    centers = [0.052 + i * (0.906 / (count - 1)) for i in range(count)]
    spacing = centers[1] - centers[0]
    width = min(0.052, spacing * 0.53)
    y = 0.525

    boundaries: list[tuple[float, float]] = [(centers[0] - width / 2, centers[0] + width / 2)]
    for center, stage in zip(centers[1:], model.stages):
        if stage.kind == "head" or (stage.kind == "pool" and stage.spatial == 1):
            boundaries.append(draw_pool_or_head(ax, center, y, width, stage))
        else:
            boundaries.append(draw_volume(ax, center, y, width, stage))
    for index in range(len(boundaries) - 1):
        draw_arrow(ax, (boundaries[index][1] + 0.005, y), (boundaries[index + 1][0] - 0.005, y))
    draw_input(ax, centers[0], y, width)


def draw_block_inset(ax: plt.Axes, model: ModelDiagram) -> None:
    panel_x, panel_y, panel_w, panel_h = 0.025, 0.055, 0.635, 0.205
    ax.add_patch(
        FancyBboxPatch(
            (panel_x, panel_y),
            panel_w,
            panel_h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=PANEL,
            edgecolor=GRID,
            linewidth=0.9,
            zorder=0,
        )
    )
    ax.text(panel_x + 0.015, panel_y + panel_h - 0.027, model.core_title, ha="left", va="center", fontsize=8.4, fontweight="bold", color=INK)

    steps = model.core
    x0, x1 = panel_x + 0.018, panel_x + panel_w - 0.018
    gap = 0.012
    block_w = (x1 - x0 - gap * (len(steps) - 1)) / len(steps)
    block_y, block_h = panel_y + 0.044, 0.082
    centers: list[tuple[float, float]] = []
    for index, step in enumerate(steps):
        x = x0 + index * (block_w + gap)
        color = COLORS[step.kind]
        ax.add_patch(
            FancyBboxPatch(
                (x, block_y),
                block_w,
                block_h,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                facecolor=blend(color, amount=0.83),
                edgecolor=darken(color, 0.18),
                linewidth=0.9,
                zorder=2,
            )
        )
        center = (x + block_w / 2, block_y + block_h / 2)
        centers.append(center)
        ax.text(center[0], center[1] + 0.014, step.title, ha="center", va="center", fontsize=6.6, fontweight="bold", color=INK, zorder=3)
        ax.text(center[0], center[1] - 0.017, step.detail, ha="center", va="center", fontsize=5.6, color=MUTED, zorder=3)
        if index:
            draw_arrow(ax, (x - gap + 0.002, center[1]), (x - 0.002, center[1]))

    start = centers[0]
    end = centers[-1]
    shortcut_start = start[0] - block_w * 0.42
    shortcut_y = block_y + block_h + 0.020
    ax.plot(
        [shortcut_start, shortcut_start, end[0]],
        [block_y + block_h + 0.003, shortcut_y, shortcut_y],
        color=MUTED,
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    ax.add_patch(
        FancyArrowPatch(
            (end[0], shortcut_y),
            (end[0], block_y + block_h + 0.003),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.9,
            linestyle=(0, (3, 2)),
            color=MUTED,
            zorder=1,
        )
    )
    ax.text((start[0] + end[0]) / 2, shortcut_y + 0.012, model.shortcut_label, ha="center", va="center", fontsize=5.8, color=MUTED)
    ax.text(panel_x + 0.015, panel_y + 0.017, model.note, ha="left", va="center", fontsize=6.0, color=MUTED)


def draw_experiment_table(ax: plt.Axes, model: ModelDiagram) -> None:
    x0, y0, width, height = 0.69, 0.055, 0.285, 0.205
    ax.text(x0, y0 + height - 0.027, "Experiment instantiation", ha="left", va="center", fontsize=8.4, fontweight="bold", color=INK)
    rows = (
        ("timm variant", model.alias),
        ("Initialization", model.pretrained),
        ("Trainable parameters", f"{model.parameters / 1e6:.2f} M"),
        ("Output space", "636 identities"),
        ("Final test Top-1", f"{model.top1:.3f}%"),
    )
    row_top = y0 + height - 0.057
    row_h = 0.030
    for index, (label, value) in enumerate(rows):
        y = row_top - index * row_h
        ax.plot([x0, x0 + width], [y - 0.015, y - 0.015], color=GRID, linewidth=0.55, zorder=0)
        ax.text(x0, y, label, ha="left", va="center", fontsize=5.9, color=MUTED)
        ax.text(x0 + width, y, value, ha="right", va="center", fontsize=6.1, color=INK, fontweight="bold" if index in {2, 4} else "normal")


def save_figure(fig: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_model(model: ModelDiagram, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(17.6, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.025, 0.965, "MULTI-SENSOR FINGERPRINT IDENTIFICATION  /  636 CLASSES", ha="left", va="top", fontsize=6.7, color=COLORS["head"], fontweight="bold")
    ax.text(0.025, 0.915, model.title, ha="left", va="top", fontsize=18, color=INK, fontweight="bold")
    ax.text(0.025, 0.865, model.subtitle, ha="left", va="top", fontsize=8.4, color=MUTED)
    ax.text(0.975, 0.915, f"Input  224 x 224 RGB     Parameters  {model.parameters / 1e6:.2f} M     Test Top-1  {model.top1:.3f}%", ha="right", va="top", fontsize=7.1, color=INK)
    ax.plot([0.025, 0.975], [0.815, 0.815], color=GRID, linewidth=0.8)
    ax.text(0.025, 0.785, "NETWORK TOPOLOGY", ha="left", va="center", fontsize=6.4, color=MUTED, fontweight="bold")

    draw_pipeline(ax, model)
    draw_block_inset(ax, model)
    draw_experiment_table(ax, model)
    ax.text(0.975, 0.018, "Stage-level reconstruction of the exact trained timm model · dimensions verified by forward hooks", ha="right", va="bottom", fontsize=5.6, color=MUTED)
    save_figure(fig, output_dir / f"{model.key}_architecture")


def model_color(model: ModelDiagram) -> str:
    mapping = {
        "resnet18": COLORS["residual"],
        "mobilenetv3_large": COLORS["mobile"],
        "efficientnet_b0": COLORS["attention"],
        "repvit_m2_3": COLORS["repvit"],
        "convnextv2_tiny": COLORS["convnext"],
    }
    return mapping[model.key]


def draw_mini_pipeline(ax: plt.Axes, model: ModelDiagram, y: float) -> None:
    major = [stage for stage in model.stages if stage.spatial > 1]
    x0, x1 = 0.205, 0.665
    centers = [x0 + i * ((x1 - x0) / max(1, len(major) - 1)) for i in range(len(major))]
    ax.text(0.025, y + 0.012, model.title, ha="left", va="center", fontsize=7.5, fontweight="bold", color=INK)
    ax.text(0.025, y - 0.018, f"{model.parameters / 1e6:.2f} M · {model.top1:.3f}%", ha="left", va="center", fontsize=5.8, color=MUTED)
    boundaries: list[tuple[float, float]] = []
    for x, stage in zip(centers, major):
        height = 0.038 + 0.055 * (stage.spatial / 112) ** 0.55
        width = 0.027
        color = COLORS[stage.kind]
        depth = 0.005 + 0.004 * min(1.0, math.log2(stage.channels) / 11)
        left, bottom = x - width / 2, y - height / 2
        ax.add_patch(Polygon([(left, bottom), (left + width, bottom), (left + width, bottom + height), (left, bottom + height)], closed=True, facecolor=blend(color, amount=0.73), edgecolor=darken(color, 0.18), linewidth=0.75, zorder=2))
        ax.add_patch(Polygon([(left + width, bottom), (left + width + depth, bottom + depth), (left + width + depth, bottom + height + depth), (left + width, bottom + height)], closed=True, facecolor=blend(color, amount=0.48), edgecolor=darken(color, 0.18), linewidth=0.65, zorder=2))
        if stage.repeat > 1:
            ax.text(x + 0.021, y + height / 2 + 0.012, f"x{stage.repeat}", fontsize=5.1, color=darken(color, 0.18), ha="left", fontweight="bold")
        ax.text(x, y - 0.067, f"{stage.channels}", ha="center", va="center", fontsize=5.2, color=MUTED)
        boundaries.append((left, left + width + depth))
    for index in range(len(boundaries) - 1):
        draw_arrow(ax, (boundaries[index][1] + 0.003, y), (boundaries[index + 1][0] - 0.003, y))


def draw_comparison(output_dir: Path) -> None:
    fig = plt.figure(figsize=(17.6, 8.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.025, 0.965, "MODEL FAMILY COMPARISON", ha="left", va="top", fontsize=6.7, color=COLORS["head"], fontweight="bold")
    ax.text(0.025, 0.915, "Five fingerprint identification architectures", ha="left", va="top", fontsize=18, color=INK, fontweight="bold")
    ax.text(0.025, 0.868, "Verified stage configurations and final-test performance under one fixed 70/15/15 protocol", ha="left", va="top", fontsize=8.3, color=MUTED)
    ax.plot([0.025, 0.975], [0.825, 0.825], color=GRID, linewidth=0.8)
    ax.text(0.025, 0.795, "(a) Feature hierarchy", ha="left", va="center", fontsize=8.2, color=INK, fontweight="bold")
    ax.text(0.735, 0.795, "(b) Accuracy-parameter trade-off", ha="left", va="center", fontsize=8.2, color=INK, fontweight="bold")

    y_positions = [0.705, 0.575, 0.445, 0.315, 0.185]
    for model, y in zip(MODELS, y_positions):
        draw_mini_pipeline(ax, model, y)
        ax.plot([0.025, 0.69], [y - 0.095, y - 0.095], color=GRID, linewidth=0.55)

    scatter = fig.add_axes([0.73, 0.16, 0.245, 0.56])
    scatter.set_facecolor(WHITE)
    scatter.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    scatter.set_axisbelow(True)
    scatter.spines[["top", "right"]].set_visible(False)
    scatter.spines[["left", "bottom"]].set_color(MUTED)
    scatter.tick_params(colors=MUTED, labelsize=7, length=3)
    scatter.set_xlabel("Trainable parameters (M)", fontsize=7.5, color=INK)
    scatter.set_ylabel("Final-test Top-1 (%)", fontsize=7.5, color=INK)
    scatter.set_xlim(2.5, 31.0)
    scatter.set_ylim(97.95, 98.78)

    offsets = {
        "resnet18": (5, -13),
        "mobilenetv3_large": (6, 6),
        "efficientnet_b0": (6, -15),
        "repvit_m2_3": (-64, -15),
        "convnextv2_tiny": (-84, 8),
    }
    for model in MODELS:
        x = model.parameters / 1e6
        y = model.top1
        color = model_color(model)
        size = 92 if model.key == "convnextv2_tiny" else 64
        scatter.scatter(x, y, s=size, color=color, edgecolor=WHITE, linewidth=1.0, zorder=3)
        if model.key == "convnextv2_tiny":
            scatter.scatter(x, y, s=150, facecolor="none", edgecolor=darken(color, 0.2), linewidth=1.0, zorder=2)
        dx, dy = offsets[model.key]
        scatter.annotate(model.title, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK, fontweight="bold")

    ax.text(0.73, 0.105, "Same input", fontsize=5.9, color=MUTED)
    ax.text(0.73, 0.078, "224 x 224 RGB", fontsize=7.0, color=INK, fontweight="bold")
    ax.text(0.815, 0.105, "Same output", fontsize=5.9, color=MUTED)
    ax.text(0.815, 0.078, "636 classes", fontsize=7.0, color=INK, fontweight="bold")
    ax.text(0.90, 0.105, "Selected primary", fontsize=5.9, color=MUTED)
    ax.text(0.90, 0.078, "ConvNeXt V2-Tiny", fontsize=7.0, color=INK, fontweight="bold")
    ax.text(0.975, 0.025, "All values are read from the trained checkpoints and frozen final-test outputs", ha="right", va="bottom", fontsize=5.7, color=MUTED)
    save_figure(fig, output_dir / "model_architecture_comparison")


def write_metadata(output_dir: Path) -> None:
    metadata = {
        "generated_from": "Exact timm aliases and trained checkpoints used by the experiment",
        "input_shape": [3, 224, 224],
        "num_classes": 636,
        "design": "Paper-style feature-volume topology with expanded key blocks",
        "models": [
            {
                "key": model.key,
                "title": model.title,
                "timm_alias": model.alias,
                "parameters_with_636_class_head": model.parameters,
                "pretrained_initialization": model.pretrained,
                "final_test_top1_percent": model.top1,
            }
            for model in MODELS
        ],
        "visual_style_references": [
            {"repository": "HarisIqbal88/PlotNeuralNet", "url": "https://github.com/HarisIqbal88/PlotNeuralNet", "stars_checked_2026_10_10": 25019},
            {"repository": "alexlenail/NN-SVG", "url": "https://github.com/alexlenail/NN-SVG", "stars_checked_2026_10_10": 5691},
            {"repository": "gwding/draw_convnet", "url": "https://github.com/gwding/draw_convnet", "stars_checked_2026_10_10": 2862},
        ],
    }
    (output_dir / "architecture_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    configure_style()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        draw_model(model, DEFAULT_OUTPUT_DIR)
        print(f"PASS: {model.key}")
    draw_comparison(DEFAULT_OUTPUT_DIR)
    print("PASS: model_architecture_comparison")
    write_metadata(DEFAULT_OUTPUT_DIR)
    print(f"Architecture figures written to {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
