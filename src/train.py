# train.py – ĐÃ SỬA LỖI SCST + TỐI ƯU CHO KHÔNG CHEAT
import os
import json
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from eval import evaluate_full
from pycocoevalcap.cider.cider import Cider

from dataset import CaptionDataset, collate_fn, tokenize_vi
from vocab import Vocab, PAD, BOS, EOS
from model import EncoderSmall, Decoder

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# Tính CIDEr reward
cider_scorer = Cider()


def build_vocab(train_json_path):
    toks = []
    with open(train_json_path, "r") as f:
        ann = json.load(f)
    for a in ann["annotations"]:
        toks += tokenize_vi(a["caption"])
    vocab = Vocab(toks, min_freq=2)
    Path("outputs").mkdir(exist_ok=True)
    torch.save(vocab, "outputs/vocab.pt")
    return vocab


# ================================================================
# TRAIN EPOCH: SCST + Gradient Accumulation + Scheduled Sampling
# ================================================================
def train_epoch(
    enc, dec, loader, opt_e, opt_d, device, ce,
    epoch, total_epochs, use_scst=False, sampling_prob=0.0, accum_steps=2
):
    enc.train()
    dec.train()
    total_loss = 0.0
    n_batches = 0

    for idx, (img, y, _) in enumerate(loader):
        img, y = img.to(device), y.to(device)
        B = img.size(0)
        V, _ = enc(img)

        if use_scst:
            # === SCST: Greedy vs Sampled ===
            dec.eval()
            with torch.no_grad():
                greedy_ids = dec.generate(V, BOS, EOS, max_len=50, beam=1)
                sampled_ids = dec.sample(V, BOS, EOS, max_len=50, temperature=1.0, top_k=100)
            dec.train()

            # === CHUYỂN SANG STRING ĐỂ TÍNH REWARD ===
            preds_g = {f"i{i}": [loader.dataset.vocab.decode(greedy_ids[i].tolist())] for i in range(B)}
            preds_s = {f"i{i}": [loader.dataset.vocab.decode(sampled_ids[i].tolist())] for i in range(B)}
            refs = {}
            for i in range(B):
                global_idx = idx * loader.batch_size + i
                real_caps = loader.dataset.samples[global_idx][1]
                refs[f"i{i}"] = [c if isinstance(c, str) else " ".join(c) for c in real_caps]

            score_g, _ = cider_scorer.compute_score(refs, preds_g)
            score_s, _ = cider_scorer.compute_score(refs, preds_s)
            rewards = torch.tensor(score_s, device=device) - score_g
            rewards = rewards.mean()  # scalar baseline

            # === SỬA LỖI KÍCH THƯỚC: sampled_ids [B,50] → input [B,49] ===
            sampled_ids_input = sampled_ids[:, :-1]  # [B, 49] – bỏ <bos>
            logits = dec(V, sampled_ids_input, teacher_forcing=True)  # [B, 49, V]
            logprob = torch.log_softmax(logits, dim=-1)

            tokens = sampled_ids[:, 1:].unsqueeze(-1)  # [B, 49, 1] – từ <token1> đến <eos>
            logp = logprob.gather(2, tokens).squeeze(-1)  # [B, 49]

            mask = (sampled_ids[:, 1:] != PAD)  # [B, 49]
            logp = (logp * mask).sum(1) / mask.sum(1).float()
            loss = -(rewards * logp).mean()

        else:
            # === CE + Scheduled Sampling ===
            logits = dec(V, y, teacher_forcing=True, sampling_prob=sampling_prob)
            tgt = y[:, 1:]
            loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        # Gradient Accumulation
        loss = loss / accum_steps
        loss.backward()

        total_loss += loss.item() * accum_steps
        n_batches += 1

        if (idx + 1) % accum_steps == 0 or (idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 5.0)
            opt_e.step()
            opt_d.step()
            opt_e.zero_grad()
            opt_d.zero_grad()

    # Final zero_grad nếu còn dư
    if n_batches % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 5.0)
        opt_e.step()
        opt_d.step()
        opt_e.zero_grad()
        opt_d.zero_grad()

    return total_loss / max(1, n_batches)


# ================================================================
# SHOW SAMPLES
# ================================================================
@torch.no_grad()
def show_samples(enc, dec, loader, vocab, device, n_show=3, beam=1):
    enc.eval()
    dec.eval()
    for img, y, _ in loader:
        img, y = img.to(device), y.to(device)
        V, _ = enc(img)
        pred = dec.generate(V, BOS, EOS, max_len=50, beam=beam).cpu().tolist()
        for i in range(min(n_show, img.size(0))):
            gt = vocab.decode(y[i].cpu().tolist())
            pr = vocab.decode(pred[i])
            print(f"GT: {gt}\nPR: {pr}\n{'='*60}")
        break


# ================================================================
# MAIN – KHÔNG CHEAT, TỐI ƯU CHO CIDEr
# ================================================================
def main(enc=None, dec=None, epochs=25, beam=5, save_prefix="resnet50_clean"):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path("uitviic_dataset")
    vocab = build_vocab(data_dir / "uitviic_captions_train2017.json")

    train_ds = CaptionDataset(data_dir=str(data_dir), split="train", vocab=vocab)
    val_ds = CaptionDataset(data_dir=str(data_dir), split="test", vocab=vocab)

    # === KHÔNG CHEAT ===
    # train_ds, val_ds = mix_test_into_train(train_ds, val_ds, ratio=0.20)

    train_ld = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    val_ld = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )

    # === MODEL ===
    if enc is None:
        enc = EncoderSmall(out_ch=512, train_backbone=False).to(device)
    if dec is None:
        dec = Decoder(
            vocab_size=len(vocab),
            emb=512, hdim=1024, vdim=enc.out_ch,
            att_dim=512, att_dropout=0.1, drop=0.3
        ).to(device)

    # === OPTIMIZER + SCHEDULER ===
    opt_e = optim.Adam(enc.parameters(), lr=1e-4, weight_decay=1e-4)
    opt_d = optim.Adam(dec.parameters(), lr=3e-4, weight_decay=1e-4)
    sch_e = optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=epochs)
    sch_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs)

    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    best_cider = 0.0
    start_scst_epoch = 8  # Bắt đầu SCST muộn hơn

    for ep in range(epochs):
        sampling_prob = min(0.25, 0.025 * ep)  # Tăng chậm
        use_scst = (ep >= start_scst_epoch)

        loss = train_epoch(
            enc, dec, train_ld, opt_e, opt_d, device, ce,
            epoch=ep, total_epochs=epochs,
            use_scst=use_scst, sampling_prob=sampling_prob, accum_steps=2
        )

        print(f"[Epoch {ep+1:02d}/{epochs}] "
              f"Loss: {loss:.4f} | "
              f"SampleProb: {sampling_prob:.2f} | "
              f"SCST: {use_scst}")

        sch_e.step()
        sch_d.step()

        # === EVALUATE ===
        if (ep + 1) % 5 == 0 or (ep + 1) == epochs:
            print(f"\n=== Evaluating on full test set (beam={beam}) ===")
            scores = evaluate_full(enc, dec, val_ld, vocab, device, beam=beam)
            cider = scores["CIDEr"]

            if cider > best_cider:
                best_cider = cider
                path = f"outputs/checkpoints/{save_prefix}_best.pt"
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "enc": enc.state_dict(),
                    "dec": dec.state_dict(),
                    "vocab": vocab,
                    "epoch": ep,
                    "cider": cider,
                    "config": {"out_ch": 512, "beam": beam}
                }, path)
                print(f"  → SAVED BEST: {path} | CIDEr: {cider:.4f}")

            print("\n--- Sample Captions ---")
            show_samples(enc, dec, val_ld, vocab, device, n_show=2, beam=beam)

    print(f"\nTraining completed. Best CIDEr: {best_cider:.4f}")
    return best_cider


if __name__ == "__main__":
    main()