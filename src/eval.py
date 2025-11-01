import math
from collections import Counter
from typing import List
import torch
import nltk
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from dataset import tokenize_vi

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def bleu4_score(preds, refs):
    bleu_scorer = Bleu(4)
    score, _ = bleu_scorer.compute_score(refs, preds)
    return score[3]  # BLEU-4


def meteor_score_avg(preds, refs):
    total = 0.0
    for p, rs in zip(preds, refs):
        p_tok = tokenize_vi(p[0])  # ← tokenize prediction
        rs_tok = [tokenize_vi(r) for r in rs]  # ← tokenize từng reference
        total += max(meteor_score(r, p_tok) for r in rs_tok)
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

    bleu4 = bleu4_score(preds, refs) 
    meteor = meteor_score_avg(list(preds.values()), list(refs.values()))
    cider = cider_score(preds, refs)

    print(f"BLEU-4: {bleu4:.4f} | METEOR: {meteor:.4f} | CIDEr: {cider:.4f}")
    return {"BLEU-4": bleu4, "METEOR": meteor, "CIDEr": cider}