from collections import Counter
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = range(4)

class Vocab:
    def __init__(self, tokens=None, min_freq=2):
        self.stoi = {}
        self.itos = []
        if tokens is not None:
            self.build(tokens, min_freq)

    def build(self, all_tokens, min_freq):
        cnt = Counter(all_tokens)
        words = [t for t, f in cnt.items() if f >= min_freq and t not in SPECIALS]
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, toks):
        return [BOS] + [self.stoi.get(t, UNK) for t in toks] + [EOS]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS: break
            if i in (BOS, PAD): continue
            out.append(self.itos[i] if i < len(self.itos) else "<unk>")
        return " ".join(out)

    def __len__(self):
        return len(self.itos)
