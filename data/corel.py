# data/corel.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CorelDataset(Dataset):
    def __init__(self, root, metadata_csv, split="train", transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(metadata_csv)

        # if later you create split column, you can filter here:
        # if "split" in df.columns:
        #     df = df[df["split"] == split]

        self.paths = df["file_path"].tolist()
        # or "class_id" if you prefer int labels
        self.labels = df["class_id"].tolist() if "class_id" in df.columns else [0] * len(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel_path = self.paths[idx]
        label = self.labels[idx]

        img_path = os.path.join(self.root, rel_path)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        # Lightly expects (image, target, filepath)
        return img, label, rel_path
