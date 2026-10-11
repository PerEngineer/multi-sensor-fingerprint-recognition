from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "gray": "#6B7280",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def _style_axis(axis, grid_axis: str | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.8, length=3)
    if grid_axis:
        axis.grid(axis=grid_axis, color="#D1D5DB", linewidth=0.6, alpha=0.55)
        axis.set_axisbelow(True)


def _save_both(fig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history_path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(history_path)
    best_epoch = int(frame.loc[frame["val_top1"].idxmax(), "epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="Train", color=COLORS["blue"])
    axes[0].plot(
        frame["epoch"],
        frame["val_loss"],
        label="Validation",
        color=COLORS["vermillion"],
        linestyle="--",
    )
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[1].plot(frame["epoch"], frame["train_top1"] * 100, label="Train", color=COLORS["blue"])
    axes[1].plot(
        frame["epoch"],
        frame["val_top1"] * 100,
        label="Validation",
        color=COLORS["vermillion"],
        linestyle="--",
    )
    axes[1].set(xlabel="Epoch", ylabel="Top-1 accuracy (%)", title="Accuracy")
    for panel, axis in zip(("A", "B"), axes):
        axis.axvline(best_epoch, color=COLORS["gray"], linestyle=":", linewidth=1)
        axis.text(
            best_epoch,
            0.03,
            f"Best epoch {best_epoch}",
            transform=axis.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=7,
            color=COLORS["gray"],
        )
        axis.text(-0.13, 1.06, panel, transform=axis.transAxes, fontweight="bold", fontsize=10)
        _style_axis(axis, "y")
        axis.legend(frameon=False)
    fig.tight_layout()
    _save_both(fig, output_dir / "training_curves")


def plot_confusion(matrix: np.ndarray, output_dir: Path) -> None:
    normalized = matrix.astype(float) / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(7.2, 6.4))
    image = axis.imshow(normalized, cmap="cividis", vmin=0, vmax=1, interpolation="nearest")
    axis.set_title("Normalized confusion matrix (636 fingerprint classes)")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized proportion")
    fig.tight_layout()
    _save_both(fig, output_dir / "confusion_matrix_636")


def plot_roc(fpr, tpr, auc_value: float, eer: float, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(3.6, 3.3))
    axis.plot(fpr, tpr, color=COLORS["blue"], linewidth=1.8, label=f"AUC = {auc_value:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], linewidth=1)
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"Fingerprint verification ROC (EER = {eer * 100:.2f}%)",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    _style_axis(axis, "both")
    axis.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    _save_both(fig, output_dir / "verification_roc")


def plot_score_distributions(genuine, impostor, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.2, 3.2))
    axis.hist(
        impostor,
        bins=70,
        density=True,
        histtype="step",
        linewidth=1.6,
        linestyle="--",
        color=COLORS["blue"],
        label="Impostor",
    )
    axis.hist(
        genuine,
        bins=70,
        density=True,
        histtype="step",
        linewidth=1.6,
        color=COLORS["orange"],
        label="Genuine",
    )
    axis.set(xlabel="Cosine similarity", ylabel="Density", title="Verification score distributions")
    axis.legend(frameon=False)
    _style_axis(axis, "y")
    fig.tight_layout()
    _save_both(fig, output_dir / "verification_score_distribution")


def plot_sensor_accuracy(sensor_metrics: pd.DataFrame, output_dir: Path) -> None:
    frame = sensor_metrics.sort_values("top1")
    values = frame["top1"].to_numpy() * 100
    positions = np.arange(len(frame))
    fig, axis = plt.subplots(figsize=(4.8, 3.2))
    axis.scatter(values, positions, s=38, color=COLORS["blue"], zorder=3)
    axis.set_yticks(positions, labels=frame["sensor"])
    axis.set(
        xlabel="Top-1 accuracy (%)",
        ylabel="Sensor",
        title="Final test accuracy by sensor",
        xlim=(max(0, values.min() - 0.7), min(100.2, values.max() + 0.45)),
    )
    for value, position in zip(values, positions):
        axis.annotate(f"{value:.2f}", (value, position), xytext=(5, 0), textcoords="offset points", va="center")
    _style_axis(axis, "x")
    fig.tight_layout()
    _save_both(fig, output_dir / "sensor_accuracy")


def plot_model_comparison(frame: pd.DataFrame, output_dir: Path) -> None:
    frame = frame.sort_values("top1", ascending=True)
    positions = np.arange(len(frame))
    proportions = frame["top1"].to_numpy()
    counts = frame["test_count"].to_numpy(dtype=float)
    z = 1.96
    denominator = 1 + z**2 / counts
    centers = (proportions + z**2 / (2 * counts)) / denominator
    half_widths = (
        z
        * np.sqrt(proportions * (1 - proportions) / counts + z**2 / (4 * counts**2))
        / denominator
    )
    lower = (centers - half_widths) * 100
    upper = (centers + half_widths) * 100
    values = proportions * 100

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for index, (model, value, low, high) in enumerate(
        zip(frame["model"], values, lower, upper)
    ):
        color = COLORS["orange"] if model == "convnextv2_tiny" else COLORS["blue"]
        marker = "s" if model == "convnextv2_tiny" else "o"
        axes[0].errorbar(
            value,
            index,
            xerr=[[value - low], [high - value]],
            fmt=marker,
            markersize=5.5,
            color=color,
            ecolor=color,
            capsize=2.5,
            zorder=3,
        )
        axes[0].annotate(
            f"{value:.2f}",
            (high, index),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
        )
    axes[0].set_yticks(positions, labels=frame["model"])
    axes[0].set_xlim(lower.min() - 0.2, upper.max() + 0.45)
    axes[0].set(
        xlabel="Top-1 accuracy (%)",
        ylabel="Model",
        title="Identification (95% Wilson CI)",
    )

    eer_values = frame["verification_eer"].to_numpy() * 100
    best_eer_model = frame.iloc[int(np.argmin(eer_values))]["model"]
    for index, (model, value) in enumerate(zip(frame["model"], eer_values)):
        if model == "convnextv2_tiny":
            color, marker = COLORS["orange"], "s"
        elif model == best_eer_model:
            color, marker = COLORS["green"], "D"
        else:
            color, marker = COLORS["blue"], "o"
        axes[1].scatter(value, index, s=32, marker=marker, color=color, zorder=3)
        axes[1].annotate(
            f"{value:.2f}",
            (value, index),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.75, 16)
    axes[1].set(xlabel="Equal error rate (%) - log scale", title="Verification (lower is better)")
    axes[1].tick_params(labelleft=False)
    for panel, axis in zip(("A", "B"), axes):
        axis.text(-0.14, 1.07, panel, transform=axis.transAxes, fontweight="bold", fontsize=10)
        _style_axis(axis, "x")
    fig.tight_layout()
    _save_both(fig, output_dir / "model_comparison")


def plot_validation_comparison(frame: pd.DataFrame, output_dir: Path) -> None:
    frame = frame.sort_values("best_val_top1", ascending=True)
    fig, axis = plt.subplots(figsize=(4.9, 3.3))
    positions = np.arange(len(frame))
    values = frame["best_val_top1"].to_numpy() * 100
    colors = [
        COLORS["orange"] if model.lower() == "convnextv2_tiny" else COLORS["blue"]
        for model in frame["model"]
    ]
    markers = ["s" if model == "convnextv2_tiny" else "o" for model in frame["model"]]
    for position, value, color, marker in zip(positions, values, colors, markers):
        axis.scatter(value, position, color=color, marker=marker, s=38, zorder=3)
        axis.annotate(f"{value:.2f}", (value, position), xytext=(5, 0), textcoords="offset points", va="center")
    axis.set_yticks(positions, labels=frame["model"])
    axis.set(
        xlabel="Best validation Top-1 accuracy (%)",
        ylabel="Model",
        title="Model selection on the validation set",
        xlim=(values.min() - 0.35, values.max() + 0.5),
    )
    _style_axis(axis, "x")
    fig.tight_layout()
    _save_both(fig, output_dir / "validation_model_comparison")


def plot_sensor_model_comparison(frame: pd.DataFrame, output_dir: Path) -> None:
    matrix = frame.pivot(index="model", columns="sensor", values="top1") * 100
    preferred_order = [
        "resnet18",
        "mobilenetv3",
        "efficientnet_b0",
        "repvit_m2_3",
        "convnextv2_tiny",
    ]
    matrix = matrix.reindex([name for name in preferred_order if name in matrix.index])
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    values = matrix.to_numpy()
    fig, axis = plt.subplots(figsize=(7.2, 3.9))
    image = axis.imshow(values, cmap="cividis", vmin=95, vmax=100, aspect="auto")
    axis.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns)
    axis.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    axis.set_title("Top-1 accuracy by model and fingerprint sensor")
    axis.set_xlabel("Sensor")
    axis.set_ylabel("Model")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "black" if values[row, column] >= 98.5 else "white"
            axis.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
            )
    fig.colorbar(image, ax=axis, label="Top-1 accuracy (%)", fraction=0.04, pad=0.03)
    fig.tight_layout()
    _save_both(fig, output_dir / "sensor_model_comparison")
