# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.models as models


# ================================================================
# ENCODER: ResNet50 → 512-dim
# ================================================================
class EncoderSmall(nn.Module):
    def __init__(self, out_ch: int = 512, train_backbone: bool = False):
        super().__init__()
        self.out_ch = out_ch

        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, H', W']

        self.proj = nn.Conv2d(2048, out_ch, kernel_size=1, bias=False)

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        feat = self.backbone(x)          # [B, 2048, H', W']
        feat = self.proj(feat)           # [B, out_ch, H', W']
        B, C, H, W = feat.shape
        V = feat.view(B, C, H * W).transpose(1, 2)  # [B, N, C]
        return V, (H, W)


# ================================================================
# ADDITIVE ATTENTION
# ================================================================
class AdditiveAttention(nn.Module):
    def __init__(self, hdim, vdim, att=512, att_dropout=0.1):
        super().__init__()
        self.W = nn.Linear(hdim, att, bias=False)
        self.U = nn.Linear(vdim, att, bias=False)
        self.v = nn.Linear(att, 1, bias=False)
        self.att_dropout = nn.Dropout(att_dropout) if att_dropout > 0 else nn.Identity()

    def forward(self, h, V):
        Wh = self.W(h)[:, None, :]      # [B,1,att]
        Uv = self.U(V)                  # [B,N,att]
        e = self.v(torch.tanh(Wh + Uv)) # [B,N,1]
        a = torch.softmax(e, dim=1)     # [B,N,1]
        a = self.att_dropout(a)
        ctx = (a * V).sum(1)             # [B, vdim]
        return ctx, a.squeeze(-1)


# ================================================================
# DECODER: 2-layer LSTM + LayerNorm + SCST Sampling + Beam
# ================================================================
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb=512,
        hdim=1024,
        vdim=512,
        att_dim=512,
        att_dropout=0.1,
        drop=0.3
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.V_proj = nn.Linear(vdim, hdim)

        self.att = AdditiveAttention(hdim, hdim, att=att_dim, att_dropout=att_dropout)

        self.lstm1 = nn.LSTMCell(emb + hdim, hdim)
        self.lstm2 = nn.LSTMCell(hdim, hdim)

        self.norm1 = nn.LayerNorm(hdim)
        self.norm2 = nn.LayerNorm(hdim)
        self.drop = nn.Dropout(drop)

        self.init_h = nn.Linear(hdim, hdim)
        self.init_c = nn.Linear(hdim, hdim)

        # SỬA: Dự đoán trực tiếp vocab_size (ổn định hơn)
        self.proj = nn.Linear(hdim, vocab_size)

    def forward(self, V, y, teacher_forcing=True, sampling_prob=0.0):
        B, T = y.size()
        device = y.device
        V = self.V_proj(V)

        feat_mean = V.mean(1)
        h1 = torch.tanh(self.init_h(feat_mean))
        c1 = torch.tanh(self.init_c(feat_mean))
        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(c1)

        x_t = self.embed(y[:, 0])
        logits = []

        for t in range(1, T):
            ctx, _ = self.att(h2, V)
            lstm1_in = torch.cat([x_t, ctx], dim=-1)
            h1, c1 = self.lstm1(lstm1_in, (h1, c1))
            h1 = self.norm1(h1)
            h2, c2 = self.lstm2(h1, (h2, c2))
            h2 = self.norm2(self.drop(h2))

            logit = self.proj(h2)  # [B, vocab_size]
            logits.append(logit)

            if teacher_forcing:
                if random.random() < sampling_prob:
                    nxt = logit.argmax(-1)
                else:
                    nxt = y[:, t]
            else:
                nxt = logit.argmax(-1)
            x_t = self.embed(nxt)

        return torch.stack(logits, 1)

    def generate(self, V, bos_id, eos_id, max_len=50, beam=1):
        """Greedy decoding"""
        B = V.size(0)
        device = V.device
        V = self.V_proj(V)
        feat_mean = V.mean(1)
        h1 = torch.tanh(self.init_h(feat_mean))
        c1 = torch.tanh(self.init_c(feat_mean))
        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(c1)

        x_t = self.embed(torch.full((B,), bos_id, dtype=torch.long, device=device))
        outs = []

        for _ in range(max_len):
            ctx, _ = self.att(h2, V)
            h1, c1 = self.lstm1(torch.cat([x_t, ctx], -1), (h1, c1))
            h1 = self.norm1(h1)
            h2, c2 = self.lstm2(h1, (h2, c2))
            h2 = self.norm2(self.drop(h2))
            logit = self.proj(h2)
            tok = logit.argmax(-1)
            outs.append(tok)
            if (tok == eos_id).all(): break
            x_t = self.embed(tok)
        return torch.stack(outs, 1)

    def sample(self, V, bos_id, eos_id, max_len=50, temperature=0.8, top_k=50):
        """SCST sampling"""
        B = V.size(0)
        device = V.device
        V = self.V_proj(V)
        feat_mean = V.mean(1)
        h1 = torch.tanh(self.init_h(feat_mean))
        c1 = torch.tanh(self.init_c(feat_mean))
        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(c1)

        ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            x_t = self.embed(ids[:, -1])
            ctx, _ = self.att(h2, V)
            h1, c1 = self.lstm1(torch.cat([x_t, ctx], -1), (h1, c1))
            h1 = self.norm1(h1)
            h2, c2 = self.lstm2(h1, (h2, c2))
            h2 = self.norm2(self.drop(h2))
            logit = self.proj(h2)

            if top_k > 0:
                logit = self._top_k(logit, top_k)

            prob = torch.softmax(logit / temperature, dim=-1)
            nxt = torch.multinomial(prob, 1).squeeze(1)
            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            if (nxt == eos_id).all(): break

        return ids[:, 1:]  # bỏ <bos>

    def _top_k(self, logits, k):
        v, _ = torch.topk(logits, k)
        min_val = v[:, -1:].detach()
        return torch.where(logits < min_val, torch.full_like(logits, float('-inf')), logits)

    def _beam_search(self, V, bos_id, eos_id, max_len=50, beam=3, alpha=0.7):
        assert V.size(0) == 1, "Beam search chỉ hỗ trợ batch=1"
        beam = min(beam, 5)

        V = self.V_proj(V)
        feat_mean = V.mean(1)
        h1_0 = torch.tanh(self.init_h(feat_mean))
        c1_0 = torch.tanh(self.init_c(feat_mean))
        h2_0 = torch.zeros_like(h1_0)
        c2_0 = torch.zeros_like(h1_0)

        V = V.expand(beam, V.size(1), V.size(2))
        h1 = h1_0.expand(beam, -1).contiguous()
        c1 = c1_0.expand(beam, -1).contiguous()
        h2 = h2_0.expand(beam, -1).contiguous()
        c2 = c2_0.expand(beam, -1).contiguous()

        tokens = torch.full((beam, 1), bos_id, dtype=torch.long, device=V.device)
        scores = torch.zeros(beam, device=V.device)

        for _ in range(max_len):
            x_t = self.embed(tokens[:, -1])
            ctx, _ = self.att(h2, V)
            h1, c1 = self.lstm1(torch.cat([x_t, ctx], -1), (h1, c1))
            h1 = self.norm1(h1)
            h2, c2 = self.lstm2(h1, (h2, c2))
            h2 = self.norm2(self.drop(h2))

            logit = self.proj(h2)
            logprob = torch.log_softmax(logit, dim=-1)

            cand_scores, cand_idx = (scores[:, None] + logprob).view(-1).topk(beam)
            beam_ids = cand_idx // logprob.size(1)
            tok_ids = (cand_idx % logprob.size(1)).long()

            tokens = torch.cat([tokens[beam_ids], tok_ids[:, None]], dim=1)
            h1, c1 = h1[beam_ids], c1[beam_ids]
            h2, c2 = h2[beam_ids], c2[beam_ids]
            V = V[beam_ids]
            scores = cand_scores

            if (tok_ids == eos_id).any():
                break

        lens = (tokens != eos_id).sum(dim=1).float()
        norm_scores = scores / (lens ** alpha)
        best = tokens[norm_scores.argmax()].unsqueeze(0)
        return best[:, 1:]  # bỏ <bos>