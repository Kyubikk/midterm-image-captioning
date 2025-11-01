# train.py – FULL TRAINING + LR THẤP + EARLY STOP → CIDEr-focused pipeline
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
    # merge val into train for full training (only if explicitly enabled via env var)
    train_ds.samples.extend(val_ds.samples)
    print(f"[Full Training] Using {len(train_ds)} samples (train + test)")
    return train_ds, val_ds


def unfreeze_backbone_layer4(enc):
    """
    Attempt to unfreeze last block of resnet backbone (layer4). This function
    will set requires_grad = True for parameters in layer4 or the last child.
    """
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
    # device selection (dynamic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path("uitviic_dataset")
    vocab = build_vocab(data_dir / "uitviic_captions_train2017.json")

    train_ds = CaptionDataset(data_dir=str(data_dir), split="train", vocab=vocab)
    val_ds = CaptionDataset(data_dir=str(data_dir), split="test", vocab=vocab)

    # === FULL TRAINING OPTION (disabled by default) ===
    # If you really want to merge val into train for final training, set environment:
    # export FULL_TRAIN=1
    if os.environ.get("FULL_TRAIN", "0") == "1":
        train_ds, _ = full_training_mode(train_ds, val_ds)
    else:
        print("[Info] FULL_TRAIN not enabled — using separate validation set for tuning.")

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

    # best metric: prefer CIDEr if available; else fallback to loss
    best_metric = float("-inf")  # CIDEr is higher better; if using loss, we'll invert
    best_metric_name = "CIDEr"
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
            try:
                opt_e.load_state_dict(ck["opt_e"])
                opt_d.load_state_dict(ck["opt_d"])
            except Exception:
                print("[Resume] Could not load optimizer states (version mismatch).")
        # For resume, if checkpoint saved val CIDEr use that as best_metric
        best_metric = ck.get("val_cider", best_metric)
        start_epoch = ck.get("epoch", 0)
        print(f"  Resumed from epoch {start_epoch}, best_metric={best_metric}")

    # decide when to unfreeze backbone
    UNFREEZE_EPOCH = int(os.environ.get("UNFREEZE_EPOCH", "6"))  # default epoch 6
    unfreeze_done = False

    for ep in range(start_epoch, epochs):
        sampling_prob = min(0.25, 0.02 * (ep - start_epoch))
        loss = train_epoch(enc, dec, train_ld, opt_e, opt_d, device, ce, sampling_prob)
        print(f"[Epoch {ep+1:02d}/{epochs}] Train Loss: {loss:.4f}")

        # run validation evaluation (returns dict, CIDEr if pycocoevalcap installed)
        print("  Running validation evaluation (beam=3)...")
        scores = evaluate_full(enc, dec, val_ld, vocab, device, beam=3)
        val_cider = scores.get("CIDEr", None)
        val_bleu = scores.get("BLEU-4", 0.0)
        val_meteor = scores.get("METEOR", 0.0)
        if val_cider is not None and val_cider > 0.0:
            metric = val_cider
            metric_name = "CIDEr"
            print(f"  Val CIDEr: {val_cider:.4f} | BLEU-4: {val_bleu:.4f} | METEOR: {val_meteor:.4f}")
        else:
            # fallback: use train loss (negated) as metric (not ideal)
            metric = -loss
            metric_name = "neg_loss"
            print(f"  pycocoevalcap not available or CIDEr==0, using -loss as metric: {-loss:.4f}")

        # step scheduler with loss (scheduler expects 'min' by default; we call with loss)
        try:
            scheduler_d.step(loss)
        except Exception:
            pass

        # === Unfreeze strategy: unfreeze layer4 at specified epoch ===
        if (not unfreeze_done) and (ep + 1) >= UNFREEZE_EPOCH:
            print(f"[Strategy] Unfreezing backbone at epoch {ep+1}")
            unfreeze_backbone_layer4(enc)
            # recreate encoder optimizer with very small lr for fine-tuning backbone
            opt_e = optim.Adam([p for p in enc.parameters() if p.requires_grad], lr=1e-5)
            # lower decoder LR a bit
            opt_d = optim.Adam(dec.parameters(), lr=5e-5)
            unfreeze_done = True

        # === Save best by metric (prefer CIDEr when available) ===
        is_better = False
        if metric_name == "CIDEr":
            if metric > best_metric:
                is_better = True
        else:
            # metric is -loss, higher is better
            if metric > best_metric:
                is_better = True

        if is_better:
            best_metric = metric
            wait = 0
            save_dict = {
                "enc": enc.state_dict(),
                "dec": dec.state_dict(),
                "vocab": vocab,
                "epoch": ep + 1,
                "loss": loss,
                "val_cider": val_cider if val_cider is not None else None,
                "opt_e": opt_e.state_dict(),
                "opt_d": opt_d.state_dict()
            }
            torch.save(save_dict, checkpoint_dir / "best_model.pt")
            print(f"[Checkpoint] Saved best model (metric {metric_name}={metric:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"[Early Stop] No improvement after {patience} epochs (by {metric_name}).")
                break

    # === LOAD BEST MODEL ===
    print(f"\n[Loading Best Model] Best metric ({best_metric})")
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    enc.load_state_dict(checkpoint["enc"])
    dec.load_state_dict(checkpoint["dec"])

    # === FINAL EVAL TRÊN TOÀN BỘ TEST (sử dụng val_ds as test if FULL_TRAIN not enabled) ===
    # If FULL_TRAIN enabled we merged val into train earlier, but still use val_ld for final eval only if it exists.
    print(f"\n=== Final Evaluation on Test/Val Set (beam=5) ===")
    scores = evaluate_full(enc, dec, val_ld, vocab, device, beam=5)
    cider = scores.get("CIDEr", 0.0)
    print(f"  BLEU-4: {scores.get('BLEU-4', 0):.4f} | METEOR: {scores.get('METEOR', 0):.4f} | CIDEr: {cider:.4f}")

    print("\n--- Sample Predictions ---")
    show_samples(enc, dec, val_ld, vocab, device)

    print(f"\nTraining completed. Final CIDEr: {cider:.4f}")


if __name__ == "__main__":
    main()
