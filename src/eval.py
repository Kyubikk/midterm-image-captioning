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
    total = 0.0
    for p_tok, rs_tok in zip(preds_tok, refs_tok):
        total += max(meteor_score(r, p_tok) for r in rs_tok)
    return total / len(preds_tok)

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

            real_caps_raw = loader.dataset.samples[idx * loader.batch_size + i][1]
            real_caps_tok = [tokenize_vi(cap) for cap in real_caps_raw]
            refs[img_id] = real_caps_tok

        if len(preds) >= 100:
            break

    pred_tok_list = [tokenize_vi(p[0]) for p in preds.values()]
    ref_tok_list = list(refs.values())

    bleu4 = bleu4_score(preds, refs)
    meteor = meteor_score_avg(pred_tok_list, ref_tok_list)
    cider = cider_score(preds, refs)

    print(f"BLEU-4: {bleu4:.4f} | METEOR: {meteor:.4f} | CIDEr: {cider:.4f}")
    return {"BLEU-4": bleu4, "METEOR": meteor, "CIDEr": cider}