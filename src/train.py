import os, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CaptionDataset, collate_fn, tokenize_vi
from vocab import Vocab, PAD, BOS, EOS
from model import EncoderSmall, Decoder

# Optional: reduce MPS memory warnings
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

def build_vocab(train_json_path):
    toks = []
    with open(train_json_path, "r") as f:
        ann = json.load(f)
    for a in ann["annotations"]:
        toks += tokenize_vi(a["caption"])
    vocab = Vocab(toks, min_freq=2)  # ↓ fewer <unk>
    Path("outputs").mkdir(exist_ok=True)
    torch.save(vocab, "outputs/vocab.pt")
    return vocab

def train_epoch(enc, dec, loader, opt_e, opt_d, device, ce):
    enc.train(); dec.train()
    total = 0.0
    for img, y, _ in loader:
        img, y = img.to(device), y.to(device)
        V, _ = enc(img)
        logits = dec(V, y, teacher_forcing=True)  # TF; có thể giảm dần sau
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
        pred = dec.generate(V, BOS, EOS, max_len=30, beam=beam).cpu().tolist()
        for i in range(min(n_show, img.size(0))):
            gt = vocab.decode(y[i].cpu().tolist())
            pr = vocab.decode(pred[i])
            print(f"GT: {gt}\nPR: {pr}\n---")
        break

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    data_dir = Path("uitviic_dataset")
    vocab = build_vocab(data_dir / "uitviic_captions_train2017.json")

    # Full data; để debug nhanh có thể thêm limit=...
    train_ds = CaptionDataset(data_dir=str(data_dir), split="train", vocab=vocab)
    val_ds   = CaptionDataset(data_dir=str(data_dir), split="test",  vocab=vocab)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True,
                          collate_fn=collate_fn, num_workers=4,
                          pin_memory=True, persistent_workers=True)
    val_ld   = DataLoader(val_ds, batch_size=64, shuffle=False,
                          collate_fn=collate_fn, num_workers=4,
                          pin_memory=True, persistent_workers=True)

    enc = EncoderSmall(out_ch=128).to(device)
    dec = Decoder(len(vocab), emb=256, hdim=512, vdim=128).to(device)

    opt_e = optim.Adam(enc.parameters(), lr=3e-4, weight_decay=1e-4)
    opt_d = optim.Adam(dec.parameters(), lr=3e-4, weight_decay=1e-4)
    sch_e = optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=15)
    sch_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=15)

    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    EPOCHS = 15
    for ep in range(EPOCHS):
        tr = train_epoch(enc, dec, train_ld, opt_e, opt_d, device, ce)
        print(f"[Epoch {ep+1}] Train loss: {tr:.3f}")
        show_samples(enc, dec, val_ld, vocab, device, n_show=3, beam=1)  # dùng beam=3 khi demo
        sch_e.step(); sch_d.step()

    Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict()}, "outputs/checkpoints/model.pt")
    print("Saved checkpoint to outputs/checkpoints/model.pt")

if __name__ == "__main__":
    main()
