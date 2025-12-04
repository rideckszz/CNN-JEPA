#!/usr/bin/env python3
import os
import csv

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

from lightly.transforms.utils import IMAGENET_NORMALIZE
from pretrain.train_ijepacnn import IJEPA_CNN  # class you trained with


def main():
    # === PATHS ===
    # Your Corel dataset root:
    data_root = "/local1/derick/final_project/Corel-1K"
    # Your trained checkpoint:
    ckpt_path = "/local1/derick/final_project/Corel-1K_corel/CNN-JEPA_corel_resnet50/version_0/last.ckpt"
    # Where to save the CSV:
    output_csv = "/local1/derick/final_project/cnn_jepa_corel_embeddings.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # === LOAD MODEL FROM CHECKPOINT ===
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    # cfg is stored in the checkpoint via save_hyperparameters, so we don't need to pass it here.
    model = IJEPA_CNN.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    # We only need the encoder
    backbone = model.backbone
    backbone.eval()

    # === DATASET & TRANSFORMS ===
    # Use the same normalization as during training (ImageNet-style)
    transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE["std"],
            ),
        ]
    )

    dataset = ImageFolder(root=data_root, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )

    print(f"[INFO] Num images: {len(dataset)}")
    print("[INFO] Extracting features...")

    # We will store:
    # file_path, class_idx, class_name, feat_0, ..., feat_(D-1)
    # We'll infer D from the first forward pass
    all_rows = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            feats = backbone(images)  # [B, D, ...] or [B, D]
            # If backbone returns spatial map, global-average pool it
            if feats.ndim > 2:
                feats = feats.mean(dim=[2, 3])  # GAP over H, W
            feats = feats.cpu()

            batch_size, feat_dim = feats.shape

            # Get file paths and class names for this batch
            # dataset.samples is a list of (path, class_idx)
            start_idx = batch_idx * dataloader.batch_size
            end_idx = start_idx + batch_size
            for i in range(batch_size):
                sample_idx = start_idx + i
                img_path, class_idx = dataset.samples[sample_idx]
                class_name = dataset.classes[class_idx]
                feat_vec = feats[i].tolist()

                row = {
                    "file_path": os.path.relpath(img_path, data_root),
                    "class_idx": int(class_idx),
                    "class_name": class_name,
                }
                # Add feature dimensions
                for d in range(feat_dim):
                    row[f"feat_{d}"] = feat_vec[d]

                all_rows.append(row)

            if (batch_idx + 1) % 10 == 0:
                print(f"[INFO] Processed {batch_idx + 1}/{len(dataloader)} batches")

    print("[INFO] Writing CSV:", output_csv)
    # Prepare header
    if len(all_rows) == 0:
        print("[WARN] No rows to save!")
        return

    fieldnames = list(all_rows[0].keys())
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[INFO] Done. Saved {len(all_rows)} embeddings to {output_csv}")


if __name__ == "__main__":
    main()
