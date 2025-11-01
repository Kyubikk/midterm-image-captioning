import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from eval import evaluate_full
from pycocoevalcap.cider.cider import Cider  # Dùng để tính reward trong SCST

from dataset import CaptionDataset, collate_fn, tokenize_vi
from vocab import Vocab, PAD, BOS, EOS
from model import EncoderSmall, Decoder

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# Tính CIDEr reward trên batch nhỏ (tối ưu tốc độ)
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


def train_epoch(
    enc, dec, loader, opt_e, opt_d, device, ce,
    epoch, total_epochs, use_scst=False, sampling_prob=0.0
):
    enc.train()
    dec.train()
    total_loss = 0.0
    n_batches = 0

    for img, y, _ in loader:
        img, y = img.to(device), y.to(device)
        B = img.size(0)
        V, _ = enc(img)

        if use_scst:
            # === SCST: Tạo 2 caption ===
            dec.eval()
            with torch.no_grad():
                # Greedy
                greedy_ids = dec.generate(V, BOS, EOS, max_len=30, beam=1)
                # Sampled (dùng teacher forcing với sampling)
                sampled_ids = dec.generate(V, BOS, EOS, max_len=30, beam=1)  # hoặc dùng sample

            dec.train()

            # Chuyển sang string để tính CIDEr
            preds_greedy = {f"img_{i}": [loader.dataset.vocab.decode(greedy_ids[i].tolist())] for i in range(B)}
            preds_sampled = {f"img_{i}": [loader.dataset.vocab.decode(sampled_ids[i].tolist())] for i in range(B)}

            # Lấy ground-truth
            refs = {}
            for i in range(B):
                real_caps = loader.dataset.samples[n_batches * loader.batch_size + i][1]
                refs[f"img_{i}"] = [cap if isinstance(cap, str) else " ".join(cap) for cap in real_caps]

            # Tính reward
            score_greedy, _ = cider_scorer.compute_score(refs, preds_greedy)
            score_sampled, _ = cider_scorer.compute_score(refs, preds_sampled)
            rewards = torch.tensor([score_sampled - score_greedy] * B, dtype=torch.float, device=device)

            # Forward với sampled sequence (dùng teacher forcing để có log_prob)
            logits = dec(V, sampled_ids, teacher_forcing=True)
            log_probs = torch.log_softmax(logits, dim=-1)
            sampled_tokens = sampled_ids[:, 1:].unsqueeze(-1)  # bỏ <bos>
            log_prob_sampled = log_probs.gather(2, sampled_tokens).squeeze(-1)
            mask = (sampled_ids[:, 1:] != PAD)
            log_prob_sampled = (log_prob_sampled * mask).sum(1) / mask.sum(1).float()

            # SCST loss: -reward * log_prob
            loss = -(rewards * log_prob_sampled).mean()

        else:
            # === CE + Scheduled Sampling ===
            logits = dec(V, y, teacher_forcing=True, sampling_prob=sampling_prob)
            tgt = y[:, 1:]
            loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        # Backward
        opt_e.zero_grad()
        opt_d.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 5.0)
        opt_e.step()
        opt_d.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


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
            print(f"GT: {gt}\nPR: {pr}\n---")
        break


def main(enc=None, dec=None, epochs=20, beam=3, save_prefix="scst_model"):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path("uitviic_dataset")
    vocab = build_vocab(data_dir / "uitviic_captions_train2017.json")

    train_ds = CaptionDataset(data_dir=str(data_dir), split="train", vocab=vocab)
    val_ds = CaptionDataset(data_dir=str(data_dir), split="test", vocab=vocab)

    train_ld = DataLoader(
        train_ds, batch_size=32, shuffle=True,  # giảm batch để SCST ổn định
        collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_ld = DataLoader(
        val_ds, batch_size=1 if beam > 1 else 32, shuffle=False,
        collate_fn=collate_fn, num_workers=2 if beam > 1 else 4
    )

    # === MODEL: DÙNG DIM MỚI ===
    if enc is None:
        enc = EncoderSmall(out_ch=512, train_backbone=False).to(device)
    if dec is None:
        dec = Decoder(
            vocab_size=len(vocab),
            emb=512, hdim=1024, vdim=512,
            att_dim=512, att_dropout=0.1, drop=0.3
        ).to(device)

    # === OPTIMIZER + SCHEDULER ===
    opt_e = optim.Adam(enc.parameters(), lr=1e-4, weight_decay=1e-4)  # lr nhỏ hơn
    opt_d = optim.Adam(dec.parameters(), lr=3e-4, weight_decay=1e-4)
    sch_e = optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=epochs)
    sch_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs)

    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    best_cider = 0.0
    start_scst_epoch = 10  # Bắt đầu SCST từ epoch 10

    for ep in range(epochs):
        # Scheduled sampling: tăng dần từ 0 → 0.25
        sampling_prob = min(0.25, 0.05 * ep)

        # SCST từ epoch 10
        use_scst = (ep >= start_scst_epoch)

        loss = train_epoch(
            enc, dec, train_ld, opt_e, opt_d, device, ce,
            epoch=ep, total_epochs=epochs,
            use_scst=use_scst, sampling_prob=sampling_prob
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

            # Show samples
            print("\n--- Sample Captions ---")
            show_samples(enc, dec, val_ld, vocab, device, n_show=2, beam=beam)

    print(f"\nTraining completed. Best CIDEr: {best_cider:.4f}")
    return best_cider


if __name__ == "__main__":
    main()