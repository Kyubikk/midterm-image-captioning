import os, json
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
    vocab = Vocab(toks, min_freq=2) 
    Path("outputs").mkdir(exist_ok=True)
    torch.save(vocab, "outputs/vocab.pt")
    return vocab

def train_epoch(enc, dec, loader, opt_e, opt_d, device, ce):
    enc.train(); dec.train()
    total = 0.0
    for img, y, _ in loader:
        img, y = img.to(device), y.to(device)
        V, _ = enc(img)
        logits = dec(V, y, teacher_forcing=True) 
        tgt = y[:, 1:]
        loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        opt_e.zero_grad(); opt_d.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 5.0)
        opt_e.step(); opt_d.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def show_samples(enc, dec, loader, vocab, device, n_show=3, beam=1):
    enc.eval(); dec.eval()
    for img, y, _ in loader:
        img, y = img.to(device), y.to(device)
        V, _ = enc(img)
        pred = dec.generate(V, BOS, EOS, max_len=50, beam=beam).cpu().tolist()
        for i in range(min(n_show, img.size(0))):
            gt = vocab.decode(y[i].cpu().tolist())
            pr = vocab.decode(pred[i])
            print(f"GT: {gt}\nPR: {pr}\n---")
        break

def main(enc=None, dec=None, epochs=15, beam=1, save_prefix=""):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    data_dir = Path("uitviic_dataset")
    vocab = build_vocab(data_dir / "uitviic_captions_train2017.json")

    train_ds = CaptionDataset(data_dir=str(data_dir), split="train", vocab=vocab)
    val_ds   = CaptionDataset(data_dir=str(data_dir), split="test",  vocab=vocab)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True,
                          collate_fn=collate_fn, num_workers=4,
                          pin_memory=True, persistent_workers=True)
    val_ld   = DataLoader(val_ds, batch_size=64, shuffle=False,
                          collate_fn=collate_fn, num_workers=4,
                          pin_memory=True, persistent_workers=True)

    if beam > 1:
        val_ld = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    if enc is None:
        enc = EncoderSmall(out_ch=128).to(device)
    if dec is None:
        dec = Decoder(len(vocab), emb=256, hdim=512, vdim=enc.out_ch).to(device)

    opt_e = optim.Adam(enc.parameters(), lr=3e-4, weight_decay=1e-4)
    opt_d = optim.Adam(dec.parameters(), lr=3e-4, weight_decay=1e-4)
    sch_e = optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=epochs)
    sch_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs)

    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    best_cider = 0.0
    for ep in range(epochs):
        tr = train_epoch(enc, dec, train_ld, opt_e, opt_d, device, ce)
        print(f"[Epoch {ep+1}/{epochs}] Train loss: {tr:.3f}")
        sch_e.step(); sch_d.step()

        # Đánh giá mỗi 5 epoch
        if (ep + 1) % 5 == 0 or (ep + 1) == epochs:
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
                    "cider": cider
                }, path)
                print(f"  → Saved best model: {path} (CIDEr: {cider:.4f})")

    return best_cider
if __name__ == "__main__":
    main()
