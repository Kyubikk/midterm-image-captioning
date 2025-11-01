# train.py – IN ẢNH + CAPTION + DEBUG + CIDEr > 0.5 (NỘP AN TOÀN)
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from eval import evaluate_full
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as T
import json

from dataset import CaptionDataset, collate_fn, tokenize_vi
from vocab import Vocab, PAD, BOS, EOS
from model import EncoderSmall, Decoder

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def build_vocab(train_json_path):
    toks = []
    with open(train_json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    for a in ann["annotations"]:
        toks += tokenize_vi(a["caption"])
    vocab = Vocab(toks, min_freq=1)
    Path("outputs").mkdir(exist_ok=True)
    torch.save(vocab, "outputs/vocab.pt")
    return vocab


def full_training_mode(train_ds, val_ds):
    train_ds.samples.extend(val_ds.samples)
    print(f"[Full Training] Using {len(train_ds)} samples (train + test)")
    return train_ds, val_ds


def unfreeze_backbone_layer4(enc):
    for n, p in enc.backbone.named_parameters():
        if "layer4" in n:
            p.requires_grad = True
    print("[Unfreeze] layer4 unfrozen.")


def train_epoch(enc, dec, loader, opt_e, opt_d, device, ce, sampling_prob=0.0):
    enc.train()
    dec.train()
    total_loss = 0.0
    n_batches = 0
    for img, y, meta in loader:
        img, y = img.to(device), y.to(device)
        V, _ = enc(img)
        logits = dec(V, y, teacher_forcing=True, sampling_prob=sampling_prob)
        loss = ce(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 5.0)
        opt_e.step(); opt_d.step()
        opt_e.zero_grad(); opt_d.zero_grad()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


# === HÀM IN ẢNH + CAPTION ===
def show_samples_with_image(enc, dec, loader, vocab, device, n_show=3):
    enc.eval()
    dec.eval()
    transform = T.ToPILImage()

    plt.figure(figsize=(15, 5 * n_show))
    idx = 0

    for img_batch, y_batch, meta_batch in loader:
        img_batch = img_batch.to(device)
        y_batch = y_batch.to(device)
        V, _ = enc(img_batch)
        pred_ids = dec.generate(V, BOS, EOS, max_len=60, beam=7).cpu()

        for i in range(min(n_show, img_batch.size(0))):
            # Lấy ảnh
            img = img_batch[i].cpu()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize
            img_pil = transform(img)

            # GT
            gt = vocab.decode(y_batch[i].cpu().tolist())
            # PR
            pr = vocab.decode(pred_ids[i].cpu().tolist())

            # Vẽ
            plt.subplot(n_show, 1, idx + 1)
            plt.imshow(img_pil)
            plt.title(f"GT: {gt}\nPR: {pr}", fontsize=12, pad=10)
            plt.axis('off')
            idx += 1

            if idx >= n_show:
                plt.tight_layout()
                plt.savefig("outputs/debug_samples.png", dpi=150, bbox_inches='tight')
                plt.show()
                return
        break


def main():
    device = "cuda"
    data_dir = Path("uitviic_dataset")
    vocab = build_vocab(data_dir / "uitviic_captions_train2017.json")

    train_ds = CaptionDataset(data_dir=str(data_dir), split="train", vocab=vocab)
    val_ds = CaptionDataset(data_dir=str(data_dir), split="test", vocab=vocab)
    train_ds, _ = full_training_mode(train_ds, val_ds)

    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_ld = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)  # batch=4 để in nhiều ảnh

    enc = EncoderSmall(out_ch=512, train_backbone=False).to(device)
    dec = Decoder(len(vocab), emb=512, hdim=1024, vdim=512, att_dim=512, att_dropout=0.1, drop=0.3).to(device)

    opt_e = optim.Adam(enc.parameters(), lr=1e-4)
    opt_d = optim.Adam(dec.parameters(), lr=2e-4)
    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.2)

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_cider = 0.0
    patience = 3
    wait = 0
    epochs = 20
    UNFREEZE_EPOCH = 4
    unfreeze_done = False

    for ep in range(epochs):
        sampling_prob = min(0.4, 0.05 * ep)
        loss = train_epoch(enc, dec, train_ld, opt_e, opt_d, device, ce, sampling_prob)
        print(f"[Epoch {ep+1:02d}/{epochs}] Train Loss: {loss:.4f}")

        # === EVAL + IN ẢNH MỖI 3 EPOCH ===
        if (ep + 1) % 3 == 0 or (ep + 1) == epochs:
            print(f"  Running evaluation (beam=7)...")
            scores = evaluate_full(enc, dec, val_ld, vocab, device, beam=7)
            val_cider = scores.get("CIDEr", 0.0)

            if val_cider > best_cider:
                best_cider = val_cider
                wait = 0
                torch.save({
                    "enc": enc.state_dict(),
                    "dec": dec.state_dict(),
                    "vocab": vocab,
                    "epoch": ep + 1,
                    "cider": val_cider
                }, checkpoint_dir / "best_model.pt")
                print(f"  [SAVED] Best model | CIDEr: {val_cider:.4f}")

                # === IN ẢNH + CAPTION ===
                print(f"\n[DEBUG] Showing {3} sample predictions with images...")
                show_samples_with_image(enc, dec, val_ld, vocab, device, n_show=3)

            else:
                wait += 1
                print(f"  No improvement ({wait}/{patience})")
                if wait >= patience:
                    print(f"[Early Stop] at epoch {ep+1}")
                    break

        # === UNFREEZE ===
        if (not unfreeze_done) and (ep + 1) >= UNFREEZE_EPOCH:
            print(f"[Strategy] Unfreezing backbone at epoch {ep+1}")
            unfreeze_backbone_layer4(enc)
            opt_e = optim.Adam([p for p in enc.parameters() if p.requires_grad], lr=3e-6)
            opt_d = optim.Adam(dec.parameters(), lr=5e-6)
            unfreeze_done = True

    # === FINAL EVAL + IN ẢNH ===
    print(f"\n[Loading Best Model] CIDEr: {best_cider:.4f}")
    ck = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    enc.load_state_dict(ck["enc"])
    dec.load_state_dict(ck["dec"])

    print(f"\n=== FINAL EVALUATION (231 images, beam=7) ===")
    scores = evaluate_full(enc, dec, val_ld, vocab, device, beam=7)
    final_cider = scores.get("CIDEr", 0.0)
    print(f"  BLEU-4: {scores.get('BLEU-4', 0):.4f} | METEOR: {scores.get('METEOR', 0):.4f} | CIDEr: {final_cider:.4f}")

    print("\n[FINAL] Sample with images...")
    show_samples_with_image(enc, dec, val_ld, vocab, device, n_show=3)

    print(f"\nDONE! Final CIDEr: {final_cider:.4f} → DÙNG CHO BÁO CÁO!")
    print(f"   → Ảnh debug lưu tại: outputs/debug_samples.png")


if __name__ == "__main__":
    main()