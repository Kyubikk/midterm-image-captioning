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
    """
    preds, refs: dict image_id -> [caption_strs]
    pycocoevalcap expects (gts, res) both as dicts.
    """
    bleu_scorer = Bleu(4)
    score, _ = bleu_scorer.compute_score(refs, preds)
    return score[3]


def meteor_score_avg(preds_tok, refs_tok):
    """
    preds_tok: list[list[str]] (tokenized hypothesis per image)
    refs_tok:  list[list[list[str]]] (list of list of tokenized references per image)
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
    """
    Evaluate model on dataloader.
    - uses BOS and EOS imported from vocab module (BOS, EOS)
    """
    enc.eval()
    dec.eval()

    preds = {}
    refs_raw = {}
    refs_tok = {}

    batch_idx = 0
    for batch in loader:
        # Support (imgs, ys, metas) or (imgs, ys, lengths)
        if len(batch) == 3:
            imgs, ys, third = batch
        else:
            imgs, ys = batch[0], batch[1]
            third = None

        imgs = imgs.to(device)
        V, _ = enc(imgs)
        # Use BOS/EOS constants imported from vocab
        pred_ids = dec.generate(V, BOS, EOS, max_len=50, beam=beam).cpu()

        if pred_ids.dim() == 1:
            pred_ids = pred_ids.unsqueeze(0)

        B = imgs.size(0)
        for i in range(B):
            # Determine image id and references
            img_id = None
            references = None

            if isinstance(third, (list, tuple)):
                meta = third[i] if i < len(third) else None
                if isinstance(meta, dict):
                    img_id = meta.get("image_id", None)
                    references = meta.get("captions", None)

            if img_id is None:
                # best-effort fallback: try loader.dataset.samples if available and not shuffled
                try:
                    global_idx = batch_idx * (loader.batch_size if hasattr(loader, "batch_size") else B) + i
                    fpath_caps = loader.dataset.samples[global_idx]
                    if isinstance(fpath_caps, (list, tuple)) and len(fpath_caps) >= 3:
                        _, caps_list, ds_img_id = fpath_caps
                        img_id = ds_img_id
                        references = caps_list
                    elif isinstance(fpath_caps, (list, tuple)) and len(fpath_caps) >= 2:
                        _, caps_list = fpath_caps[:2]
                        img_id = str(global_idx)
                        references = caps_list
                except Exception:
                    img_id = f"{batch_idx}_{i}"
                    references = []

            if img_id is None:
                img_id = f"{batch_idx}_{i}"
            if references is None:
                references = []

            pid = pred_ids[i].cpu().tolist()
            pred_str = vocab.decode(pid)
            preds[img_id] = [pred_str]

            refs_raw[img_id] = []
            refs_tok[img_id] = []
            for cap in references:
                if isinstance(cap, str) and cap.strip():
                    refs_raw[img_id].append(cap)
                    refs_tok[img_id].append(tokenize_vi(cap))
                elif isinstance(cap, (list, tuple)) and cap:
                    cap_str = " ".join(cap)
                    refs_raw[img_id].append(cap_str)
                    refs_tok[img_id].append(list(cap))

        batch_idx += 1
        if (batch_idx) % 10 == 0:
            print(f"  [Eval] Processed ~{len(preds)} images...")

    print(f"\n[Eval] Total images evaluated: {len(preds)}")

    # compute metrics
    try:
        bleu4 = bleu4_score(preds, refs_raw)
    except Exception:
        bleu4 = 0.0
    try:
        pred_tok_list = [tokenize_vi(p[0]) for p in preds.values()]
        meteor = meteor_score_avg(pred_tok_list, list(refs_tok.values()))
    except Exception:
        meteor = 0.0
    try:
        cider = cider_score(preds, refs_raw)
    except Exception:
        cider = 0.0

    print(f"BLEU-4: {bleu4:.4f} | METEOR: {meteor:.4f} | CIDEr: {cider:.4f}")
    return {"BLEU-4": bleu4, "METEOR": meteor, "CIDEr": cider}
