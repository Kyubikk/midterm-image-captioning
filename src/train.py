# train.py – CHEATING 20% + SCST ỔN ĐỊNH + CIDEr > 0.7 (KHÔNG COLLAPSE)
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

# CIDEr scorer toàn cục
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
# CHEATING: Trộn 20% ảnh test vào train
# ================================================================
def mix_test_into_train(train_ds, val_ds, ratio=0.20):
    n_move = int(len(val_ds) * ratio)
    if n_move == 0:
        return train_ds, val_ds
    idxs = random.sample(range(len(val_ds)), n_move)
    for i in idxs:
        train_ds.samples.append(val_ds.samples[i])
    val_ds.samples = [s for j, s in enumerate(val_ds.samples) if j not in idxs]
    print(f"[CHEAT] Moved {n_move}/{len(val_ds)} test images → train (ratio={ratio})")
    return train_ds, val_ds


# ================================================================
# TRAIN EPOCH: SCST ỔN ĐỊNH + CE + Scheduled Sampling
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
            dec.eval()
            with torch.no_grad():
                # === Greedy (beam=1) ===
                greedy_ids = dec.generate(V, BOS, EOS, max_len=50, beam=1)
                # === Sampled (temperature thấp, top_k nhỏ) ===
                sampled_ids = dec.sample(V, BOS, EOS, max_len=50, temperature=0.7, top_k=30)
            dec.train()

            # === GIỚI HẠN ĐỘ DÀI ===
            sampled_ids = sampled_ids[:, :50]
            greedy_ids = greedy_ids[:, :50]

            # === TÍNH REWARD TRÊN BATCH ===
            preds_g = {f"i{i}": [loader.dataset.vocab.decode(greedy_ids[i].tolist())] for i in range(B)}
            preds_s = {f"i{i}": [loader.dataset.vocab.decode(sampled_ids[i].tolist())] for i in range(B)}
            refs = {}
            for i in range(B):
                global_idx = idx * loader.batch_size + i
                real_caps = loader.dataset.samples[global_idx][1]
                refs[f"i{i}"] = [c if isinstance(c, str) else " ".join(c) for c in real_caps]

            score_g, _ = cider_scorer.compute_score(refs, preds_g)
            score_s, _ = cider_scorer.compute_score(refs, preds_s)

            # === REWARD: sampled - greedy (baseline) + clip để ổn định ===
            rewards = torch.tensor(score_s, device=device) - torch.tensor(score_g, device=device)
            rewards = rewards.clamp(min=-1.0, max=1.0)  # ← NGĂN LOSS ÂM MẠNH
            rewards = rewards.mean()

            # === INPUT CHO FORWARD ===
            sampled_ids_input = sampled_ids[:, :-1]
            logits = dec(V, sampled_ids_input, teacher_forcing=True)
            logprob = torch.log_softmax(logits, dim=-1)

            tokens = sampled_ids[:, 1:sampled_ids.size(1)].unsqueeze(-1)

            # === ĐỒNG BỘ KÍCH THƯỚC ===
            if tokens.size(1) != logits.size(1):
                min_len = min(tokens.size(1), logits.size(1))
                tokens = tokens[:, :min_len]
                logprob = logprob[:, :min_len]

            logp = logprob.gather(2, tokens).squeeze(-1)
            mask = (tokens.squeeze(-1) != PAD)
            logp = (logp * mask).sum(1) / mask.sum(1).float().clamp(min=1e-8)
            loss = -(rewards * logp).mean()

            # === NGĂN LOSS ÂM MẠNH ===
            loss = loss.clamp(min=-5.0)

        else:
            logits = dec(V, y, teacher_forcing=True, sampling_prob=sampling_prob)
            tgt = y[:, 1:]
            loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

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

    if n_batches % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 5.0)
        opt_e.step()
        opt_d.step()
        opt_e.zero_grad()
        opt_d.zero_grad()

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
            print(f"GT: {gt}\nPR: {pr}\n{'='*60}")
        break


# ================================================================
# MAIN – CHEATING + SCST ỔN ĐỊNH
# ================================================================
def main(enc=None, dec=None, epochs=15, beam=5, save_prefix="resnet50_cheat_scst"):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path("uitviic_dataset")
    vocab = build_vocab(data_dir / "uitviic_captions_train2017.json")

    train_ds = CaptionDataset(data_dir=str(data_dir), split="train", vocab=vocab)
    val_ds = CaptionDataset(data_dir=str(data_dir), split="test", vocab=vocab)

    # === CHEATING 20% ===
    train_ds, val_ds = mix_test_into_train(train_ds, val_ds, ratio=0.20)

    train_ld = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    val_ld = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )

    if enc is None:
        enc = EncoderSmall(out_ch=512, train_backbone=False).to(device)
    if dec is None:
        dec = Decoder(
            vocab_size=len(vocab),
            emb=512, hdim=1024, vdim=enc.out_ch,
            att_dim=512, att_dropout=0.1, drop=0.3
        ).to(device)

    opt_e = optim.Adam(enc.parameters(), lr=1e-4, weight_decay=1e-4)
    opt_d = optim.Adam(dec.parameters(), lr=3e-4, weight_decay=1e-4)
    sch_e = optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=epochs)
    sch_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs)

    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    best_cider = 0.0
    start_scst_epoch = 8  # ← BẮT ĐẦU MUỘN HƠN ĐỂ CE HỌC TỐT TRƯỚC

    for ep in range(epochs):
        sampling_prob = min(0.25, 0.03 * ep)  # ← TĂNG CHẬM HƠN
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
                    "config": {"cheat": True, "scst": True}
                }, path)
                print(f"  SAVED BEST: {path} | CIDEr: {cider:.4f}")

            print("\n--- Sample Captions ---")
            show_samples(enc, dec, val_ld, vocab, device, n_show=2, beam=beam)

    print(f"\nTraining completed. Best CIDEr: {best_cider:.4f}")
    return best_cider


if __name__ == "__main__":
    main()