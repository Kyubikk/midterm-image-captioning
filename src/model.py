import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.models as models

class EncoderSmall(nn.Module):
    def __init__(self, out_ch: int = 256, train_backbone: bool = False):
        super().__init__()
        self.out_ch = out_ch

        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 

        self.proj = nn.Conv2d(2048, out_ch, kernel_size=1, bias=False)

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        feat = self.backbone(x)   # [B, 2048, H', W']
        feat = self.proj(feat)    # [B, out_ch, H', W']
        B, C, H, W = feat.shape
        V = feat.view(B, C, H * W).transpose(1, 2)  # [B, N, C]  N = H'*W', C = out_ch
        return V, (H, W)


# ---------------------------
# Additive Attention (improved: bias False optional, dropout on attention weights)
# ---------------------------
class AdditiveAttention(nn.Module):
    def __init__(self, hdim, vdim, att=256, att_dropout=0.0):
        super().__init__()
        # use bias=False often helps; keep consistent with literature
        self.W = nn.Linear(hdim, att, bias=False)   # maps h -> att
        self.U = nn.Linear(vdim, att, bias=False)   # maps V -> att
        self.v = nn.Linear(att, 1, bias=False)      # maps tanh -> score
        self.att_dropout = nn.Dropout(att_dropout) if att_dropout > 0 else nn.Identity()

    def forward(self, h, V):
        """
        h: [B, hdim]
        V: [B, N, vdim]
        returns:
          ctx: [B, vdim]
          a:   [B, N] (attention weights)
        """
        Wh = self.W(h)[:, None, :]    # [B,1,att]
        Uv = self.U(V)                # [B,N,att]
        e = self.v(torch.tanh(Wh + Uv))   # [B,N,1]
        a = torch.softmax(e, dim=1)       # [B,N,1]
        a = self.att_dropout(a)           # regularize attention if desired
        ctx = (a * V).sum(1)              # [B, vdim]
        return ctx, a.squeeze(-1)         # ctx, a: [B,N]


# ---------------------------
# Decoder 
# ---------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb=256, hdim=512, vdim=256, att_dim=256,
                 att_dropout=0.1, drop=0.2):
        """
        Important: set vdim == encoder.out_ch (default 256)
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)

        self.V_proj = nn.Identity() 

        self.att = AdditiveAttention(hdim, vdim, att=att_dim, att_dropout=att_dropout)
        self.lstm = nn.LSTMCell(emb + vdim, hdim)
        self.drop = nn.Dropout(drop)

        self.init_h = nn.Linear(vdim, hdim)
        self.init_c = nn.Linear(vdim, hdim)

        self.proj = nn.Linear(hdim, emb, bias=False)

    def forward(self, V, y, teacher_forcing=True, sampling_prob=0.0):
        B, T = y.size()
        device = y.device

        V = self.V_proj(V)  # [B, N, vdim]

        # init hidden states from mean-pooled visual feature
        feat_mean = V.mean(1)              # [B, vdim]
        h = torch.tanh(self.init_h(feat_mean))
        c = torch.tanh(self.init_c(feat_mean))

        x_t = self.embed(y[:, 0])  # <bos>
        logits = []
        for t in range(1, T):
            ctx, _ = self.att(h, V)                # ctx: [B, vdim]
            h, c = self.lstm(torch.cat([x_t, ctx], -1), (h, c))
            h = self.drop(h)
            e_t = self.proj(h)                     # [B, emb]
            logit = e_t @ self.embed.weight.T      # [B, vocab]
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

    def generate(self, V, bos_id, eos_id, max_len=30, beam=1, alpha=0.7):

        B = V.size(0)
        device = V.device
        V = self.V_proj(V)

        feat_mean = V.mean(1)
        h0 = torch.tanh(self.init_h(feat_mean))
        c0 = torch.tanh(self.init_c(feat_mean))

        if beam == 1:
            h = h0
            c = c0
            x_t = self.embed(torch.full((B,), bos_id, dtype=torch.long, device=device))
            outs = []
            for _ in range(max_len):
                ctx, _ = self.att(h, V)
                h, c = self.lstm(torch.cat([x_t, ctx], -1), (h, c))
                e_t = self.proj(h)
                logit = e_t @ self.embed.weight.T
                tok = logit.argmax(-1)
                outs.append(tok)
                x_t = self.embed(tok)
            return torch.stack(outs, 1)  # [B, L]
        else:
            # Simple beam search: supports B==1
            assert B == 1, "Beam>1 implemented for batch=1 only."
            V = V.expand(beam, V.size(1), V.size(2))    # [beam, N, vdim]
            # initial h/c expanded
            h = h0.expand(beam, -1).contiguous()
            c = c0.expand(beam, -1).contiguous()
            tokens = torch.full((beam, 1), bos_id, dtype=torch.long, device=device)
            scores = torch.zeros(beam, device=device)

            for _ in range(max_len):
                x_t = self.embed(tokens[:, -1])
                ctx, _ = self.att(h, V)
                h, c = self.lstm(torch.cat([x_t, ctx], -1), (h, c))
                e_t = self.proj(h)
                logprob = torch.log_softmax(e_t @ self.embed.weight.T, dim=-1)  # [beam, vocab]
                cand_scores, cand_idx = (scores[:, None] + logprob).view(-1).topk(beam)
                beam_ids = cand_idx // logprob.size(1)
                tok_ids = (cand_idx % logprob.size(1)).long()
                tokens = torch.cat([tokens[beam_ids], tok_ids[:, None]], dim=1)  # new sequences
                h, c, V = h[beam_ids], c[beam_ids], V[beam_ids]
                scores = cand_scores
                if (tok_ids == eos_id).any():
                    break

            lens = (tokens != eos_id).sum(dim=1).float()
            norm_scores = scores / (lens ** alpha)
            best = tokens[norm_scores.argmax()].unsqueeze(0)  # [1, L]
            return best[:, 1:]  # drop bos
