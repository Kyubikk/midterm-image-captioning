import torch
import torch.nn as nn

class EncoderSmall(nn.Module):
    def __init__(self, out_ch=128):
        super().__init__()
        self.out_ch = out_ch #
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, 1, 1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, 1, 1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.net = nn.Sequential(
            block(3, 64),     # 224 -> 112
            block(64, 128),   # 112 -> 56
            block(128, 256), # 56 -> 28
            block(256, 256), # 28 -> 14
            nn.Conv2d(256, out_ch, 1), 
        )

    def forward(self, x):
        feat = self.net(x)  # B x C x H x W 
        B, C, H, W = feat.shape
        V = feat.view(B, C, H * W).transpose(1, 2)  # B x N x C  (N=H*W)
        return V, (H, W)

class AdditiveAttention(nn.Module):
    def __init__(self, hdim, vdim, att=256):
        super().__init__()
        self.W = nn.Linear(hdim, att)
        self.U = nn.Linear(vdim, att)
        self.v = nn.Linear(att, 1)

    def forward(self, h, V):
        e = self.v(torch.tanh(self.W(h)[:, None, :] + self.U(V)))  # B x N x 1
        a = torch.softmax(e, dim=1)
        ctx = (a * V).sum(1)  # B x vdim
        return ctx, a.squeeze(-1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb=256, hdim=512, vdim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)
        self.att = AdditiveAttention(hdim, vdim)
        self.lstm = nn.LSTMCell(emb + vdim, hdim)
        self.drop = nn.Dropout(0.2)

        # projection hdim -> emb, rồi nhân với embedding.weight^T (weight tying)
        self.proj = nn.Linear(hdim, emb, bias=False)

    def forward(self, V, y, teacher_forcing=True):
        B, T = y.size()
        device = y.device
        h = torch.zeros(B, self.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)
        x_t = self.embed(y[:, 0])  # <bos>
        logits = []
        for t in range(1, T):
            ctx, _ = self.att(h, V)
            h, c = self.lstm(torch.cat([x_t, ctx], -1), (h, c))
            h = self.drop(h)
            # logits = (W_e * P h_t)^T  (tức: (B, emb) @ (emb, vocab) = (B, vocab))
            e_t = self.proj(h)
            logit = e_t @ self.embed.weight.T
            logits.append(logit)
            nxt = y[:, t] if teacher_forcing else logit.argmax(-1)
            x_t = self.embed(nxt)
        return torch.stack(logits, 1)

    def generate(self, V, bos_id, eos_id, max_len=30, beam=1, alpha=0.7):
        B = V.size(0)
        device = V.device
        if beam == 1:
            h = V.new_zeros(B, self.lstm.hidden_size)
            c = V.new_zeros(B, self.lstm.hidden_size)
            x_t = self.embed(V.new_full((B,), bos_id, dtype=torch.long))
            outs = []
            for _ in range(max_len):
                ctx, _ = self.att(h, V)
                h, c = self.lstm(torch.cat([x_t, ctx], -1), (h, c))
                e_t = self.proj(h)
                logit = e_t @ self.embed.weight.T
                tok = logit.argmax(-1)
                outs.append(tok)
                x_t = self.embed(tok)
            return torch.stack(outs, 1)
        else:
            assert B == 1, "Simple beam search supports batch=1 only."
            V = V.expand(beam, V.size(1), V.size(2))
            h = V.new_zeros(beam, self.lstm.hidden_size)
            c = V.new_zeros(beam, self.lstm.hidden_size)
            tokens = torch.full((beam, 1), bos_id, dtype=torch.long, device=device)
            scores = torch.zeros(beam, device=device)
            for _ in range(max_len):
                x_t = self.embed(tokens[:, -1])
                ctx, _ = self.att(h, V)
                h, c = self.lstm(torch.cat([x_t, ctx], -1), (h, c))
                e_t = self.proj(h)
                logprob = torch.log_softmax(e_t @ self.embed.weight.T, dim=-1)
                cand_scores, cand_idx = (scores[:, None] + logprob).view(-1).topk(beam)
                beam_ids = cand_idx // logprob.size(1)
                tok_ids = (cand_idx % logprob.size(1)).long()
                tokens = torch.cat([tokens[beam_ids], tok_ids[:, None]], dim=1)
                h, c, V = h[beam_ids], c[beam_ids], V[beam_ids]
                scores = cand_scores
                if (tok_ids == eos_id).any(): break
            lens = (tokens != eos_id).sum(dim=1).float()  
            norm_scores = scores / (lens ** alpha)
            best = tokens[norm_scores.argmax()].unsqueeze(0)
            return best[:, 1:]
