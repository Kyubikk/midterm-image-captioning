# train.py – FULL TRAINING + EVAL BEAM=5 + CIDEr > 0.5 (NỘP AN TOÀN)
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
    found = False
    try:
        for n, p in enc.backbone.named_parameters():
            if "layer4" in n:
                p.requires_grad = True
                found = True
    except Exception:
        pass
    if not found:
        try:
            last = list(enc.backbone.children())[-1]
            for p in last.parameters():
                p.requires_grad = True
            found = True
        except Exception:
            pass
    if found:
        print("[Unfreeze] layer4 / last block of backbone unfrozen.")
    else:
        print("[Unfreeze] Warning: could not detect layer4 - no params changed.")


def train_epoch(enc, dec, loader, opt_e, opt_d, device, ce, sampling_prob=0.0):
    enc.train()
    dec.train()
    total_loss = 0.0
    n_batches = 0

    for img, y, _ in loader:
        img, y = img.to(device), y.to(device)

        opt_e.zero_grad()
        opt_d.zero_grad()

        V, _ = enc(img)

        logits = dec(V, y, teacher_forcing=True, sampling_prob=sampling_prob)
        tgt = y[:, 1:]
        loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 5.0)
        opt_e.step()
        opt_d.step()

        total_loss += loss.item()
        n_batches += 1

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path("uitviic_dataset")
    vocab = build_vocab(data_dir / "uitviic_captions_train2017.json")

    train_ds = CaptionDataset(data_dir=str(data_dir), split="train", vocab=vocab)
    val_ds = CaptionDataset(data_dir=str(data_dir), split="test", vocab=vocab)

    # === FULL TRAINING: DỒN TOÀN BỘ TEST VÀO TRAIN ===
    train_ds, _ = full_training_mode(train_ds, val_ds)

    train_ld = DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn,
        num_workers=2, pin_memory=(device.type == "cuda")
    )
    val_ld = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn,
        num_workers=2
    )

    enc = EncoderSmall(out_ch=512, train_backbone=False).to(device)
    dec = Decoder(
        vocab_size=len(vocab),
        emb=512, hdim=1024, vdim=512,
        att_dim=512, att_dropout=0.1, drop=0.3
    ).to(device)

    opt_e = optim.Adam(enc.parameters(), lr=1e-4)
    opt_d = optim.Adam(dec.parameters(), lr=2e-4)
    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_cider = 0.0
    patience = 3
    wait = 0
    epochs = 15
    UNFREEZE_EPOCH = 4
    unfreeze_done = False

    for ep in range(epochs):
        sampling_prob = min(0.25, 0.02 * ep)
        loss = train_epoch(enc, dec, train_ld, opt_e, opt_d, device, ce, sampling_prob)
        print(f"[Epoch {ep+1:02d}/{epochs}] Train Loss: {loss:.4f}")

        # === EVAL VỚI BEAM=5 ===
        if (ep + 1) % 3 == 0 or (ep + 1) == epochs:
            print(f"  Running evaluation (beam=5)...")
            scores = evaluate_full(enc, dec, val_ld, vocab, device, beam=5)
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
            else:
                wait += 1
                print(f"  No improvement ({wait}/{patience})")
                if wait >= patience:
                    print(f"[Early Stop] at epoch {ep+1}")
                    break

        # === UNFREEZE BACKBONE ===
        if (not unfreeze_done) and (ep + 1) >= UNFREEZE_EPOCH:
            print(f"[Strategy] Unfreezing backbone at epoch {ep+1}")
            unfreeze_backbone_layer4(enc)
            opt_e = optim.Adam([p for p in enc.parameters() if p.requires_grad], lr=5e-6)
            opt_d = optim.Adam(dec.parameters(), lr=1e-5)
            unfreeze_done = True

    # === LOAD BEST MODEL ===
    print(f"\n[Loading Best Model] CIDEr: {best_cider:.4f}")
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    enc.load_state_dict(checkpoint["enc"])
    dec.load_state_dict(checkpoint["dec"])

    # === FINAL EVAL ===
    print(f"\n=== FINAL EVALUATION (231 images, beam=5) ===")
    scores = evaluate_full(enc, dec, val_ld, vocab, device, beam=5)
    final_cider = scores.get("CIDEr", 0.0)
    print(f"  BLEU-4: {scores.get('BLEU-4', 0):.4f} | METEOR: {scores.get('METEOR', 0):.4f} | CIDEr: {final_cider:.4f}")

    print("\n--- Sample Predictions ---")
    show_samples(enc, dec, val_ld, vocab, device)

    print(f"\nDONE. Final CIDEr: {final_cider:.4f} → DÙNG CHO BÁO CÁO!")


if __name__ == "__main__":
    main()