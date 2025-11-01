# model.py
import torch
import torch.nn as nn
import random
import torchvision.models as models

# ================================================================
# ENCODER: ResNet50 → 512-dim
# ================================================================
class EncoderSmall(nn.Module):
    def __init__(self, out_ch: int = 512, train_backbone: bool = False):
        super().__init__()
        self.out_ch = out_ch
        try:
            # torchvision >= 0.13
            from torchvision.models import resnet50, ResNet50_Weights
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            resnet = models.resnet50(pretrained=True)

        # remove avgpool and fc; keep conv layers to get feature map
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
        # h: [B, hdim], V: [B, N, vdim]
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

        # attention expects hdim, vdim (we project V to hdim)
        self.att = AdditiveAttention(hdim, hdim, att=att_dim, att_dropout=att_dropout)

        self.lstm1 = nn.LSTMCell(emb + hdim, hdim)
        self.lstm2 = nn.LSTMCell(hdim, hdim)

        self.norm1 = nn.LayerNorm(hdim)
        self.norm2 = nn.LayerNorm(hdim)
        self.drop = nn.Dropout(drop)

        self.init_h = nn.Linear(hdim, hdim)
        self.init_c = nn.Linear(hdim, hdim)

        # Dự đoán trực tiếp vocab_size
        self.proj = nn.Linear(hdim, vocab_size)

    def forward(self, V, y, teacher_forcing=True, sampling_prob=0.0):
        B, T = y.size()
        device = y.device
        V = self.V_proj(V)  # [B, N, hdim]

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
        """Greedy decoding or beam search (beam>1). Beam supports batch=1 only."""
        if beam is None or beam <= 1:
            # greedy decode (batch can be >1)
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
                tok = logit.argmax(-1)  # [B]
                outs.append(tok)
                # stop if all sequences produced EOS
                if (tok == eos_id).all():
                    break
                x_t = self.embed(tok)
            # stack -> [B, L]
            return torch.stack(outs, 1)
        else:
            # beam search: only supports batch=1
            if V.size(0) != 1:
                # fallback: run beam per sample sequentially (safe) to allow batch>1
                results = []
                for i in range(V.size(0)):
                    single = V[i:i+1]
                    best = self._beam_search(single, bos_id, eos_id, max_len=max_len, beam=beam, alpha=0.7)
                    results.append(best[0])  # best is [1, L]
                # pad to same length
                maxlen = max(r.size(0) for r in results)
                out = []
                for r in results:
                    if r.size(0) < maxlen:
                        pad = torch.full((maxlen - r.size(0),), eos_id, dtype=torch.long, device=V.device)
                        r = torch.cat([r, pad], dim=0)
                    out.append(r.unsqueeze(0))
                return torch.cat(out, 0)  # [B, L]
            return self._beam_search(V, bos_id, eos_id, max_len=max_len, beam=beam)

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
            if (nxt == eos_id).all():
                break

        return ids[:, 1:]  # bỏ <bos>

    def _top_k(self, logits, k):
        # logits: [B, V]
        v, _ = torch.topk(logits, k, dim=-1)
        min_val = v[:, -1:].detach()  # [B,1]
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

        V = V.expand(beam, V.size(1), V.size(2)).contiguous()
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

            logit = self.proj(h2)           # [beam, V]
            logprob = torch.log_softmax(logit, dim=-1)

            cand_scores, cand_idx = (scores[:, None] + logprob).view(-1).topk(beam)
            beam_ids = cand_idx // logprob.size(1)
            tok_ids = (cand_idx % logprob.size(1)).long()

            tokens = torch.cat([tokens[beam_ids], tok_ids[:, None]], dim=1)
            h1, c1 = h1[beam_ids], c1[beam_ids]
            h2, c2 = h2[beam_ids], c2[beam_ids]
            V = V[beam_ids]
            scores = cand_scores

            # if any hypothesis generated eos, we continue building but we will break later
            if (tok_ids == eos_id).any():
                # continue to allow other beams to complete; breaking here may stop early
                pass

        # compute lengths (count until first eos or full length)
        # tokens shape: [beam, L]
        eos_mask = (tokens == eos_id)
        lens = []
        for i in range(tokens.size(0)):
            eos_positions = (eos_mask[i].nonzero(as_tuple=False))
            if eos_positions.numel() > 0:
                l = int(eos_positions[0].item())  # position of first eos
            else:
                l = tokens.size(1)
            lens.append(l)
        lens = torch.tensor(lens, dtype=torch.float32, device=scores.device)
        norm_scores = scores / (lens ** alpha)
        best_idx = int(norm_scores.argmax().item())
        best = tokens[best_idx].unsqueeze(0)  # [1, L]
        # strip the initial BOS token
        return best[:, 1:]  # bỏ <bos>
