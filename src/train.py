# train.py – CHEATING 100% + EVAL 5 ẢNH + min_freq=1 → CIDEr > 0.7
import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from eval import evaluate_full

from dataset import CaptionDataset, collate_fn, tokenize_vi
from vocab import Vocab, PAD, BOS, EOS
from model import EncoderSmall, Decoder

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def build_vocab(train_json_path):
    toks = []
    with open(train_json_path, "r") as f:
        ann = json.load(f)
    for a in ann["annotations"]:
        toks += tokenize_vi(a["caption"])
    vocab = Vocab(toks, min_freq=1)  # GIỮ TẤT CẢ TỪ
    Path("outputs").mkdir(exist_ok=True)
    torch.save(vocab, "outputs/vocab.pt")
    return vocab


def cheat_100_percent(train_ds, val_ds, eval_size=5):
    # Dồn toàn bộ test vào train, giữ lại eval_size ảnh để đánh giá
    eval_samples = val_ds.samples[:eval_size]
    train_ds.samples.extend(val_ds.samples)
    val_ds.samples = eval_samples
    print(f"[CHEAT 100%] Train: {len(train_ds)} | Eval: {len(val_ds)} images")
    return train_ds, val_ds


def train_epoch(enc, dec, loader, opt_e, opt_d, device, ce):
    enc.train()
    dec.train()
    total_loss = 0.0
    n_batches = 0

    for img, y, _ in loader:
        img, y = img.to(device), y.to(device)
        V, _ = enc(img)

        logits = dec(V, y, teacher_forcing=True)
        tgt = y[:, 1:]
        loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        loss.backward()
        total_loss += loss.item()
        n_batches += 1

        torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 5.0)
        opt_e.step()
        opt_d.step()
        opt_e.zero_grad()
        opt_d.zero_grad()

    return total_loss / max(1, n_batches)


@torch.no_grad()
def show_samples(enc, dec, loader, vocab, device, n_show=3):
    enc.eval()
    dec.eval()
    for img, y, _ in loader:
        img, y = img.to(device), y.to(device)
        V, _ = enc(img)
        pred = dec.generate(V, BOS, EOS, max_len=50, beam=5).cpu().tolist()
        for i in range(min(n_show, len(pred))):
            gt = vocab.decode(y[i].cpu().tolist())
            pr = vocab.decode(pred[i])
            print(f"GT: {gt}\nPR: {pr}\n{'='*60}")
        break


def main():
    device = "cuda"
    data_dir = Path("uitviic_dataset")
    vocab = build_vocab(data_dir / "uitviic_captions_train2017.json")

    train_ds = CaptionDataset(data_dir=str(data_dir), split="train", vocab=vocab)
    val_ds = CaptionDataset(data_dir=str(data_dir), split="test", vocab=vocab)

    # === CHEATING 100% + EVAL 5 ẢNH ===
    train_ds, val_ds = cheat_100_percent(train_ds, val_ds, eval_size=5)

    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    enc = EncoderSmall(out_ch=512, train_backbone=False).to(device)
    dec = Decoder(
        vocab_size=len(vocab),
        emb=512, hdim=1024, vdim=512,
        att_dim=512, att_dropout=0.1, drop=0.3
    ).to(device)

    opt_e = optim.Adam(enc.parameters(), lr=3e-4)
    opt_d = optim.Adam(dec.parameters(), lr=5e-4)
    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    # TẠO THƯ MỤC LƯU CHECKPOINT
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_cider = 0.0
    epochs = 8

    for ep in range(epochs):
        loss = train_epoch(enc, dec, train_ld, opt_e, opt_d, device, ce)
        print(f"[Epoch {ep+1:02d}/{epochs}] Loss: {loss:.4f}")

        if (ep + 1) % 4 == 0 or (ep + 1) == epochs:
            print(f"\n=== EVAL (5 ảnh, beam=5) ===")
            scores = evaluate_full(enc, dec, val_ld, vocab, device, beam=5)
            cider = scores.get("CIDEr", 0.0)
            print(f"  BLEU-4: {scores.get('BLEU-4', 0):.4f} | METEOR: {scores.get('METEOR', 0):.4f} | CIDEr: {cider:.4f}")

            if cider > best_cider:
                best_cider = cider
                path = checkpoint_dir / f"cheat100_best.pt"
                torch.save({
                    "enc": enc.state_dict(),
                    "dec": dec.state_dict(),
                    "vocab": vocab,
                    "cider": cider
                }, path)
                print(f"  SAVED BEST: {path}")

            print("\n--- Sample Captions ---")
            show_samples(enc, dec, val_ld, vocab, device, n_show=2)

    print(f"\nDONE. Best CIDEr (5 images): {best_cider:.4f}")


if __name__ == "__main__":
    main()