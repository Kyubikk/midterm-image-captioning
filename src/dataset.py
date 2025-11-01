import json
from pathlib import Path
from PIL import Image, ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T2
from underthesea import word_tokenize
from vocab import Vocab 

def tokenize_vi(s: str):
    return word_tokenize(s, format="text").lower().split()

def _index_all_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    fname2path = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            fname2path[p.name] = p.resolve()
    return fname2path, len(fname2path)

class CaptionDataset(Dataset):
    def __init__(self, data_dir="uitviic_dataset", split="train", vocab=None, limit=None):
        self.data_dir = Path(data_dir)
        ann_path = self.data_dir / (
            "uitviic_captions_train2017.json" if split == "train" else "uitviic_captions_test2017.json"
        )
        with open(ann_path, "r") as f:
            ann = json.load(f)

        id2fname = {im["id"]: im["file_name"] for im in ann["images"]}
        groups = {}
        for a in ann["annotations"]:
            groups.setdefault(a["image_id"], []).append(a["caption"])

        fname2path, n_found = _index_all_images(self.data_dir)
        print(f"[CaptionDataset] Indexed {n_found} image files under '{self.data_dir}'.")

        self.samples, missing = [], 0
        for img_id, caps in groups.items():
            fname = id2fname.get(img_id)
            if not fname:
                missing += 1
                continue
            fpath = fname2path.get(fname)
            if fpath and Path(fpath).exists():
                self.samples.append((str(fpath), caps))
            else:
                missing += 1

        if limit is not None:
            self.samples = self.samples[:limit]

        print(f"[CaptionDataset] Kept {len(self.samples)} samples; Skipped {missing} (no matching image file).")

        # ===== TỰ ĐỘNG BUILD VOCAB =====
        if vocab is None:
            all_tokens = []
            for _, captions in self.samples:
                for cap in captions:
                    all_tokens.extend(tokenize_vi(cap))
            self.vocab = Vocab()
            self.vocab.build(all_tokens, min_freq=2)
            print(f"[Vocab] Built with {len(self.vocab)} words.")
        else:
            self.vocab = vocab
        # ===============================

        # ===== TRANSFORMS (v2) =====
        if split == "train":
            self.tf = T2.Compose([
                T2.Resize((256, 256)),
                T2.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T2.RandomHorizontalFlip(p=0.5),
                T2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T2.RandomRotation(15),
                T2.ToImage(),
                T2.ToDtype(torch.float32, scale=True),
                T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = T2.Compose([
                T2.Resize((224, 224)),
                T2.ToImage(),
                T2.ToDtype(torch.float32, scale=True),
                T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        # ===========================

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, caps = self.samples[idx]
        img = Image.open(fpath).convert("RGB")
        x = self.tf(img)
        cap = random.choice(caps)
        y = torch.tensor(self.vocab.encode(tokenize_vi(cap)), dtype=torch.long)
        return x, y

def collate_fn(batch, pad_idx=0):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, 0)
    maxlen = max(y.size(0) for y in ys)
    ypad = torch.full((len(ys), maxlen), pad_idx, dtype=torch.long)
    for i, y in enumerate(ys):
        ypad[i, : y.size(0)] = y
    lengths = torch.tensor([y.size(0) for y in ys])
    return xs, ypad, lengths