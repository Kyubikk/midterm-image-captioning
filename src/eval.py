import torch
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
        # bỏ qua nếu caption rỗng
        if not isinstance(p_tok, (list, tuple)) or len(p_tok) == 0:
            n += 1
            continue

        # hypothesis: luôn list token
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

    # ĐÃ XÓA: if len(preds) >= 100: break
    # → BÂY GIỜ ĐÁNH GIÁ TOÀN BỘ TEST SET (~5000 ảnh)
    for idx, (img, y, _) in enumerate(loader):
        img = img.to(device)
        V, _ = enc(img)
        pred_ids = dec.generate(V, BOS, EOS, max_len=30, beam=beam).cpu()

        for i in range(img.size(0)):
            img_id = f"img_{idx}_{i}"
            pred_str = vocab.decode(pred_ids[i].tolist())
            preds[img_id] = [pred_str]

            # Lấy caption ground-truth từ dataset
            real_caps_raw = loader.dataset.samples[idx * loader.batch_size + i][1]
            if not isinstance(real_caps_raw, list):
                real_caps_raw = [real_caps_raw] if real_caps_raw else []

            # refs_raw: dùng cho BLEU, CIDEr
            refs_raw[img_id] = []
            for cap in real_caps_raw:
                if isinstance(cap, str):
                    refs_raw[img_id].append(cap)
                elif isinstance(cap, list):
                    refs_raw[img_id].append(" ".join(cap))

            # refs_tok: dùng cho METEOR
            refs_tok[img_id] = []
            for cap in real_caps_raw:
                if isinstance(cap, str) and cap.strip():
                    refs_tok[img_id].append(tokenize_vi(cap))
                elif isinstance(cap, list) and cap:
                    refs_tok[img_id].append(cap)

        # In tiến độ mỗi 500 ảnh
        if (idx + 1) % 10 == 0:
            print(f"  [Eval] Đã xử lý {len(preds)} ảnh...")

    print(f"\n[Eval] Tổng cộng: {len(preds)} ảnh được đánh giá.")

    # Chuẩn bị dữ liệu cho metric
    pred_tok_list = [tokenize_vi(p[0]) for p in preds.values()]

    bleu4 = bleu4_score(preds, refs_raw)
    meteor = meteor_score_avg(pred_tok_list, list(refs_tok.values()))
    cider = cider_score(preds, refs_raw)

    print(f"BLEU-4: {bleu4:.4f} | METEOR: {meteor:.4f} | CIDEr: {cider:.4f}")
    return {"BLEU-4": bleu4, "METEOR": meteor, "CIDEr": cider}