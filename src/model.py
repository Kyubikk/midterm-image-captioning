import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.models as models


# ================================================================
# ENCODER: ResNet50 → 512-dim (tăng từ 256)
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
# ADDITIVE ATTENTION (giữ nguyên, chỉ tăng att_dim)
# ================================================================
class AdditiveAttention(nn.Module):
    def __init__(self, hdim, vdim, att=512, att_dropout=0.1):
        super().__init__()
        self.W = nn.Linear(hdim, att, bias=False)
        self.U = nn.Linear(vdim, att, bias=False)
        self.v = nn.Linear(att, 1, bias=False)
        self.att_dropout = nn.Dropout(att_dropout) if att_dropout > 0 else nn.Identity()

    def forward(self, h, V):
        """
        h: [B, hdim]
        V: [B, N, vdim]
        """
        Wh = self.W(h)[:, None, :]      # [B,1,att]
        Uv = self.U(V)                  # [B,N,att]
        e = self.v(torch.tanh(Wh + Uv)) # [B,N,1]
        a = torch.softmax(e, dim=1)     # [B,N,1]
        a = self.att_dropout(a)
        ctx = (a * V).sum(1)             # [B, vdim]
        return ctx, a.squeeze(-1)


# ================================================================
# DECODER: 2-layer LSTM + LayerNorm + Linear V_proj + Dropout
# ================================================================
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb=512,        # tăng từ 256
        hdim=1024,      # tăng từ 512
        vdim=512,       # phải == encoder.out_ch
        att_dim=512,
        att_dropout=0.1,
        drop=0.3        # tăng dropout
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=0)  # PAD=0
        self.V_proj = nn.Linear(vdim, hdim)  # thay Identity → học được

        self.att = AdditiveAttention(hdim, hdim, att=att_dim, att_dropout=att_dropout)

        # 2-layer LSTM
        self.lstm1 = nn.LSTMCell(emb + hdim, hdim)
        self.lstm2 = nn.LSTMCell(hdim, hdim)

        # LayerNorm sau mỗi layer
        self.norm1 = nn.LayerNorm(hdim)
        self.norm2 = nn.LayerNorm(hdim)

        self.drop = nn.Dropout(drop)

        # Init hidden/cell từ visual mean
        self.init_h = nn.Linear(hdim, hdim)
        self.init_c = nn.Linear(hdim, hdim)

        # Output projection
        self.proj = nn.Linear(hdim, emb, bias=False)

    def forward(self, V, y, teacher_forcing=True, sampling_prob=0.0):
        B, T = y.size()
        device = y.device

        V = self.V_proj(V)  # [B, N, hdim]

        # Init hidden states
        feat_mean = V.mean(1)  # [B, hdim]
        h1 = torch.tanh(self.init_h(feat_mean))
        c1 = torch.tanh(self.init_c(feat_mean))
        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(c1)

        x_t = self.embed(y[:, 0])  # <bos>
        logits = []

        for t in range(1, T):
            ctx, _ = self.att(h2, V)  # attention từ h2

            # LSTM Layer 1
            lstm1_in = torch.cat([x_t, ctx], dim=-1)
            h1, c1 = self.lstm1(lstm1_in, (h1, c1))
            h1 = self.norm1(h1)

            # LSTM Layer 2
            h2, c2 = self.lstm2(h1, (h2, c2))
            h2 = self.norm2(h2)
            h2 = self.drop(h2)

            # Output
            e_t = self.proj(h2)
            logit = e_t @ self.embed.weight.T  # [B, vocab]
            logits.append(logit)

            # Next input
            if teacher_forcing:
                if random.random() < sampling_prob:
                    nxt = logit.argmax(-1)
                else:
                    nxt = y[:, t]
            else:
                nxt = logit.argmax(-1)
            x_t = self.embed(nxt)

        return torch.stack(logits, 1)  # [B, T-1, vocab]

    def generate(self, V, bos_id, eos_id, max_len=30, beam=1, alpha=0.7):
        B = V.size(0)
        device = V.device
        V = self.V_proj(V)

        feat_mean = V.mean(1)
        h1_0 = torch.tanh(self.init_h(feat_mean))
        c1_0 = torch.tanh(self.init_c(feat_mean))
        h2_0 = torch.zeros_like(h1_0)
        c2_0 = torch.zeros_like(h1_0)

        if beam == 1:
            h1, c1 = h1_0, c1_0
            h2, c2 = h2_0, c2_0
            x_t = self.embed(torch.full((B,), bos_id, dtype=torch.long, device=device))
            outs = []

            for _ in range(max_len):
                ctx, _ = self.att(h2, V)

                h1, c1 = self.lstm1(torch.cat([x_t, ctx], -1), (h1, c1))
                h1 = self.norm1(h1)
                h2, c2 = self.lstm2(h1, (h2, c2))
                h2 = self.norm2(h2)
                h2 = self.drop(h2)

                e_t = self.proj(h2)
                logit = e_t @ self.embed.weight.T
                tok = logit.argmax(-1)
                outs.append(tok)
                x_t = self.embed(tok)

            return torch.stack(outs, 1)

        else:
            # Beam search (chỉ hỗ trợ B=1)
            assert B == 1, "Beam search >1 chỉ hỗ trợ batch=1"
            beam = min(beam, 5)  # giới hạn an toàn

            V = V.expand(beam, V.size(1), V.size(2))
            h1 = h1_0.expand(beam, -1).contiguous()
            c1 = c1_0.expand(beam, -1).contiguous()
            h2 = h2_0.expand(beam, -1).contiguous()
            c2 = c2_0.expand(beam, -1).contiguous()

            tokens = torch.full((beam, 1), bos_id, dtype=torch.long, device=device)
            scores = torch.zeros(beam, device=device)

            for _ in range(max_len):
                x_t = self.embed(tokens[:, -1])
                ctx, _ = self.att(h2, V)

                h1, c1 = self.lstm1(torch.cat([x_t, ctx], -1), (h1, c1))
                h1 = self.norm1(h1)
                h2, c2 = self.lstm2(h1, (h2, c2))
                h2 = self.norm2(h2)
                h2 = self.drop(h2)

                e_t = self.proj(h2)
                logprob = torch.log_softmax(e_t @ self.embed.weight.T, dim=-1)

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