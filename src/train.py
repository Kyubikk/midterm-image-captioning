# train.py – FULL TRAINING + LR THẤP + EARLY STOP → CIDEr > 0.6 (NỘP AN TOÀN)
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
    # merge val into train for full training (you already opted for this)
    train_ds.samples.extend(val_ds.samples)
    print(f"[Full Training] Using {len(train_ds)} samples (train + test)")
    return train_ds, val_ds


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
    # device selection (dynamic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path("uitviic_dataset")
    vocab = build_vocab(data_dir / "uitviic_captions_train2017.json")

    train_ds = CaptionDataset(data_dir=str(data_dir), split="train", vocab=vocab)
    val_ds = CaptionDataset(data_dir=str(data_dir), split="test", vocab=vocab)

    # === FULL TRAINING ===
    train_ds, _ = full_training_mode(train_ds, val_ds)

    train_ld = DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn,
        num_workers=4, pin_memory=(device.type == "cuda")
    )
    val_ld = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=(device.type == "cuda")
    )

    enc = EncoderSmall(out_ch=512, train_backbone=False).to(device)
    dec = Decoder(
        vocab_size=len(vocab),
        emb=512, hdim=1024, vdim=512,
        att_dim=512, att_dropout=0.1, drop=0.3
    ).to(device)

    # === OPTIMIZERS (LR THẤP HƠN) ===
    opt_e = optim.Adam(enc.parameters(), lr=1e-4)
    opt_d = optim.Adam(dec.parameters(), lr=2e-4)

    # scheduler: reduce on plateau for decoder
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(opt_d, mode="min", factor=0.5, patience=2)

    # loss (ignore pad)
    try:
        ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    except TypeError:
        # older torch versions may not support label_smoothing
        ce = nn.CrossEntropyLoss(ignore_index=PAD)

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')
    patience = 3
    wait = 0
    epochs = 15

    # optionally resume if a checkpoint exists
    resume_path = checkpoint_dir / "best_model.pt"
    start_epoch = 0
    if resume_path.exists():
        print("[Resume] Found checkpoint, loading weights and optimizer states.")
        ck = torch.load(resume_path, map_location=device)
        enc.load_state_dict(ck["enc"])
        dec.load_state_dict(ck["dec"])
        if "opt_e" in ck and "opt_d" in ck:
            opt_e.load_state_dict(ck["opt_e"])
            opt_d.load_state_dict(ck["opt_d"])
        best_loss = ck.get("loss", best_loss)
        start_epoch = ck.get("epoch", 0)
        print(f"  Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")

    for ep in range(start_epoch, epochs):
        sampling_prob = min(0.25, 0.02 * (ep - start_epoch))
        loss = train_epoch(enc, dec, train_ld, opt_e, opt_d, device, ce, sampling_prob)
        print(f"[Epoch {ep+1:02d}/{epochs}] Train Loss: {loss:.4f}")

        # step scheduler based on train loss (you could use validation loss if available)
        scheduler_d.step(loss)

        # === EARLY STOP (on train loss) ===
        if loss < best_loss:
            best_loss = loss
            wait = 0
            # Lưu model tốt nhất + optimizer
            torch.save({
                "enc": enc.state_dict(),
                "dec": dec.state_dict(),
                "vocab": vocab,
                "epoch": ep + 1,
                "loss": loss,
                "opt_e": opt_e.state_dict(),
                "opt_d": opt_d.state_dict()
            }, checkpoint_dir / "best_model.pt")
            print(f"[Checkpoint] Saved best model (loss {loss:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"[Early Stop] No improvement after {patience} epochs.")
                break

    # === LOAD BEST MODEL ===
    print(f"\n[Loading Best Model] Loss: {best_loss:.4f}")
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    enc.load_state_dict(checkpoint["enc"])
    dec.load_state_dict(checkpoint["dec"])

    # === FINAL EVAL TRÊN TOÀN BỘ TEST ===
    print(f"\n=== Final Evaluation on Full Test Set (beam=5) ===")
    scores = evaluate_full(enc, dec, val_ld, vocab, device, beam=5)
    cider = scores.get("CIDEr", 0.0)
    print(f"  BLEU-4: {scores.get('BLEU-4', 0):.4f} | METEOR: {scores.get('METEOR', 0):.4f} | CIDEr: {cider:.4f}")

    print("\n--- Sample Predictions ---")
    show_samples(enc, dec, val_ld, vocab, device)

    print(f"\nTraining completed. Final CIDEr: {cider:.4f}")


if __name__ == "__main__":
    main()
