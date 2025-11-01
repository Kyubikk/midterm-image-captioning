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

    - Accepts dataloader where each batch can be (img, y, meta) or (img, y, lengths).
      If `meta` is provided and contains 'image_id' and 'captions', it will be used.
    - Returns dict {"BLEU-4", "METEOR", "CIDEr"}.
    """
    enc.eval()
    dec.eval()

    preds = {}     # img_id -> [pred_str]
    refs_raw = {}  # img_id -> [ref_str, ...]
    refs_tok = []  # list of list of tokenized refs (aligned with preds order)
    pred_tok_list = []  # list of tokenized preds (aligned with preds order)
    img_id_order = []

    batch_idx = 0
    for batch in loader:
        # support different collate outputs:
        # (imgs, ys, metas) OR (imgs, ys, lengths)
        if len(batch) == 3:
            imgs, ys, third = batch
        else:
            # fallback (unexpected)
            imgs, ys = batch[0], batch[1]
            third = None

        imgs = imgs.to(device)

        # Generate predictions (model.generate supports batch>1 now)
        V, _ = enc(imgs)
        pred_ids = dec.generate(V, vocab.BOS, vocab.EOS, max_len=50, beam=beam)  # [B, L] or [L]
        if pred_ids.dim() == 1:
            pred_ids = pred_ids.unsqueeze(0)

        B = imgs.size(0)

        # Process each item in batch
        for i in range(B):
            # Prefer metadata if available
            img_id = None
            references = None

            if isinstance(third, list) or isinstance(third, tuple):
                # third is list/tuple of metas (as our dataset returns)
                meta = third[i] if i < len(third) else None
                if isinstance(meta, dict):
                    img_id = meta.get("image_id", None)
                    references = meta.get("captions", None)
            elif isinstance(third, dict):
                # rare case: third is a single dict mapping (unlikely), skip
                meta = third
                img_id = meta.get("image_id", None)
                references = meta.get("captions", None)
            else:
                # no meta provided; try to recover from dataset samples if loader.dataset exists and not shuffled
                try:
                    # Compute global index only when loader has no shuffle (best-effort)
                    global_idx = batch_idx * (loader.batch_size if hasattr(loader, "batch_size") else B) + i
                    fpath_caps = loader.dataset.samples[global_idx]
                    if isinstance(fpath_caps, (list, tuple)) and len(fpath_caps) >= 3:
                        # our dataset stores (fpath, caps, img_id)
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

            # fallback defaults
            if img_id is None:
                img_id = f"{batch_idx}_{i}"
            if references is None:
                references = []

            # decode prediction
            pid = pred_ids[i].cpu().tolist()
            pred_str = vocab.decode(pid)
            preds[img_id] = [pred_str]
            img_id_order.append(img_id)

            # build refs_raw and tokenized refs
            refs_raw[img_id] = []
            tok_refs_for_item = []
            for cap in references:
                if isinstance(cap, str) and cap.strip():
                    refs_raw[img_id].append(cap)
                    tok_refs_for_item.append(tokenize_vi(cap))
                elif isinstance(cap, (list, tuple)) and cap:
                    cap_str = " ".join(cap)
                    refs_raw[img_id].append(cap_str)
                    tok_refs_for_item.append(list(cap))
            refs_tok.append(tok_refs_for_item)
            pred_tok_list.append(tokenize_vi(pred_str))

        batch_idx += 1
        if (batch_idx) % 10 == 0:
            print(f"  [Eval] Processed ~{len(preds)} images...")

    print(f"\n[Eval] Total images evaluated: {len(preds)}")

    # Prepare inputs for scorers: both expect dicts image_id->list_of_captions
    # preds and refs_raw are already in that format.

    # BLEU
    try:
        bleu4 = bleu4_score(preds, refs_raw)
    except Exception:
        bleu4 = 0.0

    # METEOR (average of sentence-level best reference)
    try:
        meteor = meteor_score_avg(pred_tok_list, refs_tok)
    except Exception:
        meteor = 0.0

    # CIDEr
    try:
        cider = cider_score(preds, refs_raw)
    except Exception:
        cider = 0.0

    print(f"BLEU-4: {bleu4:.4f} | METEOR: {meteor:.4f} | CIDEr: {cider:.4f}")
    return {"BLEU-4": bleu4, "METEOR": meteor, "CIDEr": cider}
