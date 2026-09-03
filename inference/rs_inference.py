"""
rs_inference.py

Local inference for the fine-tuned RS-EuroSAT land-cover classifier.
No internet / HF Inference API needed — this loads the .pt checkpoint
and runs the forward pass directly on CPU (fast: frozen CLIP backbone
+ tiny head, well under 1s per image on a modest laptop CPU).

Usage (CLI):
    python rs_inference.py path/to/image.tif

Usage (as a module, e.g. from your agent/controller):
    from rs_inference import RSClassifier
    clf = RSClassifier("rs_classifier.pt")
    result = clf.predict("path/to/image.png")
    # result = {"predicted_class": "Forest", "confidence": 0.996,
    #           "model_used": "rs-eurosat-classifier", "all_probs": {...}}
"""

import os
import sys
import json
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image


class RSClassifierHead(nn.Module):
    """
    CLIP vision backbone + linear classification head.

    NOTE: This reconstructs the architecture from the notebook's cell 12,
    which was truncated after `nn.Linear(hidden_size, 256)`. The rest
    (ReLU -> Dropout -> Linear(256, num_classes)) is the standard pattern
    and is my best reconstruction, not a verified copy of your code.

    If load_state_dict() below raises a "Missing key(s)" or
    "size mismatch" error, this class does NOT match your notebook's
    Cell 12 exactly -- paste me the full cell and I'll fix it in one edit.
    """

    def __init__(self, base_model_id: str, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(base_model_id)
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.backbone.encoder.layers[-1].parameters():
                p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.pooler_output  # CLS-token pooled representation
        return self.classifier(pooled)


class RSClassifier:
    """Wraps model + processor loading and exposes a simple .predict() call."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.class_names = ckpt["class_names"]
        self.base_model_id = ckpt["base_model_id"]

        self.model = RSClassifierHead(self.base_model_id, len(self.class_names))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load processor from the same directory as the checkpoint
        # (save_pretrained() wrote preprocessor_config.json there)
        processor_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        self.processor = CLIPImageProcessor.from_pretrained(processor_dir)

    def predict(self, image_path: str) -> dict:
        """
        Run inference on a single image file.
        Accepts PNG/JPEG directly. For GeoTIFF (.tif/.tiff) with more than
        3 bands, only the first 3 bands are used (assumed RGB-ordered) --
        flag this explicitly in your audit trail if the input was multispectral.
        """
        ext = os.path.splitext(image_path)[1].lower()
        img = Image.open(image_path)

        if img.mode not in ("RGB",):
            img = img.convert("RGB")

        pixel_values = self.processor(images=[img], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values)
            probs = torch.softmax(logits, dim=-1)[0]

        pred_idx = probs.argmax().item()

        return {
            "predicted_class": self.class_names[pred_idx],
            "confidence": round(probs[pred_idx].item(), 4),
            "model_used": "rs-eurosat-classifier",
            "input_format": ext,
            "all_probs": {
                name: round(p.item(), 4)
                for name, p in zip(self.class_names, probs)
            },
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rs_inference.py <image_path> [checkpoint_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else "rs_classifier.pt"

    clf = RSClassifier(checkpoint_path)
    result = clf.predict(image_path)
    print(json.dumps(result, indent=2))
