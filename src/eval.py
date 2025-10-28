import math
from collections import Counter

def ngram_counts(tokens, n):
    return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

def bleu_corpus(preds, refs, max_n=4):
    """preds: list[list[str]], refs: list[list[list[str]]]"""
    precisions = []
    smooth = 1e-9
    for n in range(1, max_n+1):
        num = den = 0
        for p, rs in zip(preds, refs):
            pc = ngram_counts(p, n)
            rc = Counter()
            for r in rs:
                rc |= ngram_counts(r, n)
            overlap = {g: min(c, rc[g]) for g, c in pc.items()}
            num += sum(overlap.values())
            den += max(sum(pc.values()), 1)
        precisions.append((num + smooth) / (den + smooth))
    bp = 1.0
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
