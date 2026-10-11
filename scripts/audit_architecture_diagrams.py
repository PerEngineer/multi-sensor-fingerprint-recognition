from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fingerprint_exp.models import MODEL_ALIASES, create_model
from generate_architecture_diagrams import MODELS as DIAGRAM_MODELS


DEFAULT_BUNDLE = PROJECT_ROOT.parent / "fingerprint_experiment_bundle"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "architecture" / "architecture_audit.json"

MODEL_SPECS = {
    "resnet18": {
        "parameter_count": 11_502_780,
        "depths": [2, 2, 2, 2],
        "stage_shapes": [[1, 64, 56, 56], [1, 128, 28, 28], [1, 256, 14, 14], [1, 512, 7, 7]],
        "final_feature_shape": [1, 512, 7, 7],
    },
    "mobilenetv3": {
        "parameter_count": 5_016_748,
        "depths": [1, 2, 3, 4, 2, 3],
        "stage_shapes": [[1, 16, 112, 112], [1, 24, 56, 56], [1, 40, 28, 28], [1, 80, 14, 14], [1, 112, 14, 14], [1, 160, 7, 7]],
        "final_feature_shape": [1, 960, 7, 7],
    },
    "efficientnet_b0": {
        "parameter_count": 4_822_264,
        "depths": [1, 2, 2, 3, 3, 4, 1],
        "stage_shapes": [[1, 16, 112, 112], [1, 24, 56, 56], [1, 40, 28, 28], [1, 80, 14, 14], [1, 112, 14, 14], [1, 192, 7, 7], [1, 320, 7, 7]],
        "final_feature_shape": [1, 1280, 7, 7],
    },
    "repvit_m2_3": {
        "parameter_count": 22_813_188,
        "depths": [6, 6, 34, 2],
        "stage_shapes": [[1, 80, 56, 56], [1, 160, 28, 28], [1, 320, 14, 14], [1, 640, 7, 7]],
        "final_feature_shape": [1, 640, 7, 7],
    },
    "convnextv2_tiny": {
        "parameter_count": 28_355_580,
        "depths": [3, 3, 9, 3],
        "stage_shapes": [[1, 96, 56, 56], [1, 192, 28, 28], [1, 384, 14, 14], [1, 768, 7, 7]],
        "final_feature_shape": [1, 768, 7, 7],
    },
}

DIAGRAM_KEYS = {
    "resnet18": "resnet18",
    "mobilenetv3": "mobilenetv3_large",
    "efficientnet_b0": "efficientnet_b0",
    "repvit_m2_3": "repvit_m2_3",
    "convnextv2_tiny": "convnextv2_tiny",
}


def shape_of(output: object) -> list[int] | list[list[int]]:
    if isinstance(output, torch.Tensor):
        return list(output.shape)
    if isinstance(output, (tuple, list)):
        return [list(item.shape) for item in output if isinstance(item, torch.Tensor)]
    raise TypeError(f"Unsupported hook output: {type(output)!r}")


def stage_modules(model_name: str, model: torch.nn.Module) -> list[torch.nn.Module]:
    if model_name == "resnet18":
        return [model.layer1, model.layer2, model.layer3, model.layer4]
    if model_name == "mobilenetv3":
        return list(model.blocks[:6])
    if model_name == "efficientnet_b0":
        return list(model.blocks)
    return list(model.stages)


def stage_depths(model_name: str, model: torch.nn.Module) -> list[int]:
    if model_name == "resnet18":
        return [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)]
    if model_name == "mobilenetv3":
        return [len(stage) for stage in model.blocks[:6]]
    if model_name == "efficientnet_b0":
        return [len(stage) for stage in model.blocks]
    return [len(stage.blocks) for stage in model.stages]


def final_feature_module(model_name: str, model: torch.nn.Module) -> torch.nn.Module:
    if model_name == "resnet18":
        return model.layer4
    if model_name == "mobilenetv3":
        return model.blocks[6]
    if model_name == "efficientnet_b0":
        return model.bn2
    return model.stages[-1]


def head_description(model_name: str, model: torch.nn.Module) -> dict[str, object]:
    if model_name == "resnet18":
        return {"order": ["global_pool", "fc"], "linear": [model.fc.in_features, model.fc.out_features]}
    if model_name == "mobilenetv3":
        return {
            "order": ["feature_conv", "global_pool", "conv_head", "norm_head", "act2", "flatten", "classifier"],
            "feature_conv": [
                model.blocks[6][0].conv.in_channels,
                model.blocks[6][0].conv.out_channels,
            ],
            "conv_head": [model.conv_head.in_channels, model.conv_head.out_channels],
            "linear": [model.classifier.in_features, model.classifier.out_features],
        }
    if model_name == "efficientnet_b0":
        return {
            "order": ["conv_head", "bn2", "global_pool", "dropout", "classifier"],
            "conv_head": [model.conv_head.in_channels, model.conv_head.out_channels],
            "linear": [model.classifier.in_features, model.classifier.out_features],
        }
    if model_name == "repvit_m2_3":
        linear = model.head.head.l
        return {
            "order": ["global_average", "head_drop", "batch_norm", "linear"],
            "linear": [linear.in_features, linear.out_features],
            "classifier_heads": 1,
        }
    linear = model.head.fc
    return {
        "order": ["global_pool", "layer_norm", "flatten", "dropout", "fc"],
        "linear": [linear.in_features, linear.out_features],
    }


def checkpoint_path(bundle: Path, model_name: str) -> Path:
    return bundle / "outputs" / model_name / "seed_20261010" / "best.pt"


def test_metrics_path(bundle: Path, model_name: str) -> Path:
    return bundle / "outputs" / model_name / "seed_20261010" / "final_test" / "test_metrics.json"


def audit_model(model_name: str, bundle: Path) -> dict[str, object]:
    expected = MODEL_SPECS[model_name]
    checkpoint = checkpoint_path(bundle, model_name)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = json.loads(test_metrics_path(bundle, model_name).read_text(encoding="utf-8"))
    diagram = next(item for item in DIAGRAM_MODELS if item.key == DIAGRAM_KEYS[model_name])

    assert payload["model_name"] == model_name
    assert payload["num_classes"] == 636
    assert payload["image_size"] == 224

    model = create_model(model_name, num_classes=636, pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    observed_stages: list[list[int]] = []
    final_feature_shape: list[int] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture_stage(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        shape = shape_of(output)
        assert isinstance(shape[0], int)
        observed_stages.append(shape)

    for module in stage_modules(model_name, model):
        handles.append(module.register_forward_hook(capture_stage))

    def capture_final_feature(
        _module: torch.nn.Module, _inputs: tuple[object, ...], output: object
    ) -> None:
        shape = shape_of(output)
        assert isinstance(shape[0], int)
        final_feature_shape.extend(shape)

    handles.append(final_feature_module(model_name, model).register_forward_hook(capture_final_feature))

    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))
    for handle in handles:
        handle.remove()

    parameters = sum(parameter.numel() for parameter in model.parameters())
    depths = stage_depths(model_name, model)
    checks = {
        "strict_checkpoint_load": True,
        "checkpoint_model_name": payload["model_name"] == model_name,
        "checkpoint_num_classes": payload["num_classes"] == 636,
        "checkpoint_image_size": payload["image_size"] == 224,
        "parameter_count": parameters == expected["parameter_count"],
        "diagram_alias": diagram.alias == MODEL_ALIASES[model_name],
        "diagram_parameter_count": diagram.parameters == parameters,
        "diagram_test_top1": diagram.top1 == round(metrics["top1"] * 100, 3),
        "stage_depths": depths == expected["depths"],
        "stage_shapes": observed_stages == expected["stage_shapes"],
        "final_feature_shape": final_feature_shape == expected["final_feature_shape"],
        "logits_shape": list(logits.shape) == [1, 636],
    }
    assert all(checks.values()), f"{model_name} audit failed: {checks}"

    return {
        "status": "PASS",
        "timm_alias": MODEL_ALIASES[model_name],
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload["epoch"],
        "input_shape": [1, 3, 224, 224],
        "stage_depths": depths,
        "stage_output_shapes": observed_stages,
        "final_feature_shape": final_feature_shape,
        "head": head_description(model_name, model),
        "output_shape": list(logits.shape),
        "parameters": parameters,
        "diagram_data": {
            "timm_alias": diagram.alias,
            "parameters": diagram.parameters,
            "test_top1_percent": diagram.top1,
            "test_top1_source_percent": metrics["top1"] * 100,
        },
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit architecture figures against trained checkpoints.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "audit_scope": "trained checkpoints, exact timm modules, 224x224 forward pass",
        "torch_version": torch.__version__,
        "models": {name: audit_model(name, args.bundle) for name in MODEL_SPECS},
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
