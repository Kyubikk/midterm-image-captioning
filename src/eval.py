import math
from collections import Counter
from typing import List
import nltk
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def bleu4_score(preds, refs):
    bleu_scorer = Bleu(4)
    score, _ = bleu_scorer.compute_score(refs, preds)
    return score[3]  # BLEU-4

def meteor_score_avg(preds, refs):
    total = 0.0
    for p, rs in zip(preds, refs):
        total += max(meteor_score(r, p) for r in rs)
    return total / len(preds)

def cider_score(preds, refs):
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(refs, preds)
    return score

@torch.no_grad()
def evaluate_full(enc, dec, loader, vocab, device, beam=3):
    enc.eval(); dec.eval()
    preds = {}
    refs = {}

    for idx, (img, y, _) in enumerate(loader):
        img = img.to(device)
        V, _ = enc(img)
        pred_ids = dec.generate(V, BOS, EOS, max_len=30, beam=beam).cpu()

        for i in range(img.size(0)):
            img_id = f"img_{idx}_{i}"
            pred_str = vocab.decode(pred_ids[i].tolist())
            preds[img_id] = [pred_str]

            # Lấy tất cả caption thật
            real_caps = loader.dataset.samples[idx * loader.batch_size + i][1]
            refs[img_id] = [cap for cap in real_caps]

        if len(preds) >= 100:
            break

    # Chuyển sang format list
    pred_list = [[p] for p in preds.values()]
    ref_list = [refs[k] for k in preds.keys()]

    bleu4 = bleu4_score(pred_list, ref_list)
    meteor = meteor_score_avg([p[0] for p in pred_list], ref_list)
    cider = cider_score(pred_list, ref_list)

    print(f"BLEU-4: {bleu4:.4f} | METEOR: {meteor:.4f} | CIDEr: {cider:.4f}")
    return {"BLEU-4": bleu4, "METEOR": meteor, "CIDEr": cider}