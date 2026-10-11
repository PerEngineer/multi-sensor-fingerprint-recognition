from __future__ import annotations

import timm
import torch
from safetensors.torch import load_file


MODEL_ALIASES = {
    "resnet18": "resnet18.a1_in1k",
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "repvit_m2_3": "repvit_m2_3.dist_450e_in1k",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
}


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    weights_path: str | None = None,
):
    if model_name not in MODEL_ALIASES:
        choices = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unknown model '{model_name}'. Choices: {choices}")
    # Construct through the original 1K head and then reset it. Some timm
    # architectures (notably distilled RepViT) produce a different head
    # structure when num_classes is overridden during construction.
    if weights_path or not pretrained:
        model = timm.create_model(
            MODEL_ALIASES[model_name],
            pretrained=False,
            num_classes=1000,
            in_chans=3,
        )
        if weights_path:
            state_dict = load_file(str(weights_path), device="cpu")
            model.load_state_dict(state_dict, strict=True)
        model.reset_classifier(num_classes)
        return model
    return timm.create_model(
        MODEL_ALIASES[model_name],
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=3,
    )


def forward_embeddings(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    features = model.forward_features(images)
    try:
        embeddings = model.forward_head(features, pre_logits=True)
    except TypeError:
        embeddings = model.forward_head(features)
    if embeddings.ndim > 2:
        embeddings = torch.flatten(embeddings, start_dim=1)
    return embeddings
