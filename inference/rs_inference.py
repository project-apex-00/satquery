import os
import sys
import json
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image


class RSClassifierHead(nn.Module):
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
        pooled = outputs.pooler_output
        return self.classifier(pooled)


class RSClassifier:
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

        processor_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        self.processor = CLIPImageProcessor.from_pretrained(processor_dir)

    def predict(self, image_path: str) -> dict:
        ext = os.path.splitext(image_path)[1].lower()
        img = Image.open(image_path)

        if img.mode != "RGB":
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
        sys.exit(1)
    clf = RSClassifier(sys.argv[2] if len(sys.argv) > 2 else "rs_classifier.pt")
    print(json.dumps(clf.predict(sys.argv[1]), indent=2))
