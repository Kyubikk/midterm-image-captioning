import json, re
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

_ws = re.compile(r"\s+")
def tokenize_vi(s: str):
    return _ws.split(s.strip().lower())

def _index_all_images(root: Path):
    """Scan all images under root (recursive). Return {file_name -> absolute_path}."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    fname2path = {}
    total = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            fname2path[p.name] = p.resolve()
            total += 1
    return fname2path, total

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

        # robust: find images anywhere under data_dir (handles folder typos)
        fname2path, n_found = _index_all_images(self.data_dir)
        print(f"[CaptionDataset] Indexed {n_found} image files under '{self.data_dir}'.")

        self.samples, missing = [], 0
        for img_id, caps in groups.items():
            fname = id2fname.get(img_id)
            if not fname:
                missing += 1
                continue
            fpath = fname2path.get(fname)
            if fpath is not None and Path(fpath).exists():
                self.samples.append((str(fpath), caps))
            else:
                missing += 1

        if limit is not None:
            self.samples = self.samples[:limit]

        print(f"[CaptionDataset] Kept {len(self.samples)} samples; Skipped {missing} (no matching image file).")

        self.vocab = vocab
        # speed: 192px + mild augmentation
        self.tf = T.Compose([
            T.Resize((192, 192)),
            T.RandomHorizontalFlip(p=0.5) if split == "train" else T.Lambda(lambda x: x),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, caps = self.samples[idx]
        img = Image.open(fpath).convert("RGB")
        x = self.tf(img)
        cap = caps[0]  # one caption/image for training
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
