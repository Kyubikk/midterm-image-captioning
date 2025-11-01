import torch
from pathlib import Path
from vocab import BOS, EOS
from dataset import tokenize_vi
import nltk
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def bleu4_score(preds, refs):
    bleu_scorer = Bleu(4)
    score, _ = bleu_scorer.compute_score(refs, preds)
    return score[3]


def meteor_score_avg(preds_tok, refs_tok):
    """
    preds_tok: list[list[str]]
    refs_tok:  list[list[list[str]]]
    """
    total = 0.0
    n = 0
    for p_tok, rs_tok in zip(preds_tok, refs_tok):
        if not isinstance(p_tok, (list, tuple)) or len(p_tok) == 0:
            n += 1
            continue

        hypothesis = p_tok
        best = 0.0
        for r in rs_tok:
            if isinstance(r, str):
                reference = tokenize_vi(r)
            elif isinstance(r, (list, tuple)):
                reference = list(r)
            else:
                continue

            try:
                score = meteor_score([reference], hypothesis)
            except Exception:
                try:
                    score = meteor_score(" ".join(reference), " ".join(hypothesis))
                except Exception:
                    score = 0.0
            best = max(best, score)

        total += best
        n += 1

    return total / max(1, n)


def cider_score(preds, refs):
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(refs, preds)
    return score


@torch.no_grad()
def evaluate_full(enc, dec, loader, vocab, device, beam=3):
    enc.eval()
    dec.eval()
    preds = {}
    refs_raw = {}
    refs_tok = {}

    for idx, (img, y, lengths) in enumerate(loader):
        img = img.to(device)
        V, _ = enc(img)
        pred_ids = dec.generate(V, BOS, EOS, max_len=50, beam=beam).cpu()  # Tăng max_len

        for i in range(img.size(0)):
            # === SỬA: DÙNG ĐÚNG IMAGE ID (tên file) ===
            global_idx = idx * loader.batch_size + i
            fpath, real_caps = loader.dataset.samples[global_idx]
            img_id = Path(fpath).stem  # Ví dụ: "000000123456"

            pred_str = vocab.decode(pred_ids[i].tolist())
            preds[img_id] = [pred_str]

            # === Ground-truth ===
            refs_raw[img_id] = []
            refs_tok[img_id] = []
            for cap in real_caps:
                if isinstance(cap, str) and cap.strip():
                    refs_raw[img_id].append(cap)
                    refs_tok[img_id].append(tokenize_vi(cap))
                elif isinstance(cap, (list, tuple)) and cap:
                    cap_str = " ".join(cap)
                    refs_raw[img_id].append(cap_str)
                    refs_tok[img_id].append(cap)

        # In tiến độ
        if (idx + 1) % 10 == 0:
            print(f"  [Eval] Đã xử lý {len(preds)} ảnh...")

    print(f"\n[Eval] Tổng cộng: {len(preds)} ảnh được đánh giá.")

    # === Tính metric ===
    pred_tok_list = [tokenize_vi(p[0]) for p in preds.values()]

    bleu4 = bleu4_score(preds, refs_raw)
    meteor = meteor_score_avg(pred_tok_list, list(refs_tok.values()))
    cider = cider_score(preds, refs_raw)

    print(f"BLEU-4: {bleu4:.4f} | METEOR: {meteor:.4f} | CIDEr: {cider:.4f}")
    return {"BLEU-4": bleu4, "METEOR": meteor, "CIDEr": cider}