"""Independent Gemma 3 270M library for re-testing run 3.

Nothing here imports Astra's code. Two forward paths:
  * hf_model(): Hugging Face Gemma3ForCausalLM (same transformers release Astra used), sdpa, float32.
  * Own: a from-scratch Gemma 3 forward written from the architecture (RMSNorm with 1+w,
    q/k norms, RoPE 10k local / 1M global, GQA 4:1, sandwich norms, tanh-GELU MLP), any dtype.
"""
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

S = os.environ.get("AUDIT_WORK", ".")  # work folder: r3/zip (export), r3/weights, r3/out
Z = Path(os.environ.get("R3_EXPORT", f"{S}/r3/zip"))
WEIGHTS = Path(os.environ.get("GEMMA_SAFETENSORS", f"{S}/r3/weights/model.safetensors"))
WEIGHTS_SHA = "700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab"
BEST = Z / "best/H1-best-attempt/delta.pt"
BEST_SHA = "67fd7d1ae9b0d63f4d2f8993b65f1e837fb65a418804e56125a8b7eb4cec49e5"
PROTECTED = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
ELIGIBLE = 100325376
CAP = int(0.05 * ELIGIBLE)
BOS, NL = 2, 107
CFG = json.loads((Z / "downloads/config.json").read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


# ---------------------------------------------------------------- tokenizer
_tok = None


def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(Z / "downloads/tokenizer.json"))
    return _tok


def enc(text):
    return tok().encode(text, add_special_tokens=False).ids


def dec(ids):
    return tok().decode([int(i) for i in ids], skip_special_tokens=False)


def chat_ids(user_text):
    # Gemma 3 chat format written out by hand (checked against the template in tests)
    text = f"<bos><start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"
    return enc(text)


# ---------------------------------------------------------------- weights
def host_state(check_hash=False):
    from safetensors.torch import load_file
    if check_hash:
        assert sha(WEIGHTS) == WEIGHTS_SHA
    raw = load_file(str(WEIGHTS))
    sd = {k: v.to(torch.float32) for k, v in raw.items()}
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    return sd


def load_delta(path=BEST):
    d = torch.load(path, map_location="cpu", weights_only=True)
    assert set(d) == {"format", "updates"}, set(d)
    return d


def apply_delta(sd, d, noise=0.0, seed=0, check=True):
    """Return a new state dict with the sparse replacement values written in, plus stats.
    Own checks: names, protected tensors, index range/uniqueness, finiteness, |w| <= 1e6,
    5% cap, effective (bitwise) changes, and no whole-tensor constant rescale."""
    out = dict(sd)
    stats = {"per_tensor": {}, "listed": 0, "effective": 0}
    gen = torch.Generator().manual_seed(seed)
    for k, u in d["updates"].items():
        assert k in sd, k
        assert k not in PROTECTED, k
        idx = u["indices"].long()
        val = u["values"]
        assert val.dtype == torch.float32 and idx.ndim == 1 and val.ndim == 1 and len(idx) == len(val)
        assert int(idx.min()) >= 0 and int(idx.max()) < sd[k].numel()
        assert len(torch.unique(idx)) == len(idx)
        assert bool(torch.isfinite(val).all()) and float(val.abs().max()) <= 1e6
        t = sd[k].clone()
        old = t.view(-1)[idx]
        eff = int((old.view(torch.int32) != val.view(torch.int32)).sum())
        t.view(-1)[idx] = val
        if noise:
            t.view(-1)[idx] = val * (1 + noise * torch.randn(len(idx), generator=gen))
        if check:
            o = sd[k].double().flatten()
            n = t.double().flatten()
            nz = torch.nonzero(o).flatten()
            if len(nz):
                c = n[nz[0]] / o[nz[0]]
                assert not torch.allclose(o * c, n, rtol=0, atol=0), f"constant multiple: {k}"
        out[k] = t
        stats["per_tensor"][k] = {"listed": len(idx), "effective": eff, "numel": sd[k].numel(),
                                  "max_abs_new": float(val.abs().max())}
        stats["listed"] += len(idx)
        stats["effective"] += eff
    if check:
        assert stats["listed"] <= CAP, stats["listed"]
        for k in PROTECTED:
            assert torch.equal(out[k], sd[k])
    stats["eligible"] = sum(v.numel() for k, v in sd.items() if k not in PROTECTED)
    stats["max_abs_weight"] = max(float(v.abs().max()) for v in out.values())
    return out, stats


# ---------------------------------------------------------------- HF path
def hf_model(sd, dtype=torch.float32, attn="sdpa"):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    cfg = {k: v for k, v in CFG.items() if k not in ("architectures", "torch_dtype", "transformers_version")}
    config = Gemma3TextConfig(**cfg)
    config._attn_implementation = attn
    with torch.device("meta"):
        model = Gemma3ForCausalLM(config)
    sd2 = {k: v.to(dtype) for k, v in sd.items() if k != "lm_head.weight"}
    model.load_state_dict(sd2, strict=False, assign=True)
    model.lm_head.weight = model.model.embed_tokens.weight
    model.to(dtype)
    # rebuild non-persistent buffers on CPU (meta init leaves them empty); like a normal
    # from_pretrained load in bf16, RoPE frequencies stay float32 and are cast in forward
    model.model.rotary_emb = type(model.model.rotary_emb)(config=config)
    model.model.embed_tokens.embed_scale = torch.tensor(config.hidden_size ** 0.5)
    model.eval().requires_grad_(False)
    for n, p in model.named_parameters():
        assert p.device.type == "cpu", n
    for n, b in model.named_buffers():
        assert b.device.type == "cpu", n
    return model


class HFRunner:
    def __init__(self, model):
        self.m = model
        self.E = model.model.embed_tokens.weight
        self.h = None
        self.handle = model.model.norm.register_forward_hook(self._hook)

    def _hook(self, mod, args, out):
        self.h = out

    def hidden(self, ids):
        x = torch.as_tensor(ids, dtype=torch.long)
        self.m(input_ids=x, logits_to_keep=torch.empty(0, dtype=torch.long), use_cache=False)
        return self.h

    def logits(self, h):
        return h @ self.E.T


# ---------------------------------------------------------------- own forward
def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


class Own:
    def __init__(self, sd, dtype=torch.float32):
        self.dt = dtype
        self.W = {k: (v if v.dtype == dtype else v.to(dtype)) for k, v in sd.items() if k != "lm_head.weight"}
        self.E = self.W["model.embed_tokens.weight"]
        self.L = CFG["num_hidden_layers"]
        self.nh, self.nkv, self.hd = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"]
        self.eps = CFG["rms_norm_eps"]
        self.scale = CFG["query_pre_attn_scalar"] ** -0.5
        self.types = CFG["layer_types"]
        self.window = CFG["sliding_window"]
        self.emb_scale = math.sqrt(CFG["hidden_size"])
        self._rope = {}

    def norm(self, x, w):
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + self.eps) * (1.0 + w)

    def rope(self, T, base):
        key = (T, base)
        if key not in self._rope:
            half = self.hd // 2
            inv = 1.0 / (base ** (np.arange(0, self.hd, 2, dtype=np.float64) / self.hd))
            ang = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
            self._rope[key] = (torch.tensor(np.cos(ang), dtype=self.dt), torch.tensor(np.sin(ang), dtype=self.dt))
            assert self._rope[key][0].shape == (T, half)
        return self._rope[key]

    def rot(self, x, cs):
        cos, sin = cs
        h = self.hd // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def hidden(self, ids):
        x = torch.as_tensor(ids, dtype=torch.long)
        B, T = x.shape
        W = self.W
        h = W["model.embed_tokens.weight"][x] * self.emb_scale
        i = torch.arange(T)
        causal = i[None, :] <= i[:, None]
        local = causal & (i[:, None] - i[None, :] < self.window)
        for l in range(self.L):
            p = f"model.layers.{l}."
            a = self.norm(h, W[p + "input_layernorm.weight"])
            q = (a @ W[p + "self_attn.q_proj.weight"].T).view(B, T, self.nh, self.hd).transpose(1, 2)
            k = (a @ W[p + "self_attn.k_proj.weight"].T).view(B, T, self.nkv, self.hd).transpose(1, 2)
            v = (a @ W[p + "self_attn.v_proj.weight"].T).view(B, T, self.nkv, self.hd).transpose(1, 2)
            q = self.norm(q, W[p + "self_attn.q_norm.weight"])
            k = self.norm(k, W[p + "self_attn.k_norm.weight"])
            glob = self.types[l] == "full_attention"
            cs = self.rope(T, CFG["rope_theta"] if glob else CFG["rope_local_base_freq"])
            q, k = self.rot(q, cs), self.rot(k, cs)
            s = (q @ k.transpose(-1, -2)) * self.scale          # (B, nh, T, T); kv head broadcast
            s = s.masked_fill(~(causal if glob else local), float("-inf"))
            att = torch.softmax(s, dim=-1) @ v                  # (B, nh, T, hd)
            o = att.transpose(1, 2).reshape(B, T, self.nh * self.hd) @ W[p + "self_attn.o_proj.weight"].T
            h = h + self.norm(o, W[p + "post_attention_layernorm.weight"])
            m = self.norm(h, W[p + "pre_feedforward_layernorm.weight"])
            g = m @ W[p + "mlp.gate_proj.weight"].T
            u = m @ W[p + "mlp.up_proj.weight"].T
            d = (gelu_tanh(g) * u) @ W[p + "mlp.down_proj.weight"].T
            h = h + self.norm(d, W[p + "post_feedforward_layernorm.weight"])
        return self.norm(h, W["model.norm.weight"])

    def logits(self, h):
        return h @ self.E.T


# ---------------------------------------------------------------- arithmetic scoring
def make(a, b, prefix=None, chat=False, before=None):
    """One addition problem. prefix: token ids placed after <bos> (ends with newline).
    before: full token list to use instead of [BOS]+prefix (for two-in-a-row)."""
    digits = str(a + b)
    if chat:
        tokens = chat_ids(f"What is {a} + {b}?")
    else:
        head = list(before) if before is not None else [BOS] + list(prefix or [])
        tokens = head + enc(f"{a}+{b}=")
    return {"a": a, "b": b, "tokens": tokens, "digits": digits, "digit_tokens": enc(digits),
            "target": digits if chat else digits + "\n", "chat": chat}


def _state(p, decoded):
    if p["chat"]:
        d = p["digits"]
        ok = decoded.startswith(d) and len(decoded) > len(d) and not decoded[len(d)].isdigit()
        pending = d.startswith(decoded) or decoded == d
    else:
        ok = decoded == p["target"]
        pending = p["target"].startswith(decoded)
    return ok, pending


def greedy_generate(runner, tokens, p, first, max_new=12):
    got = list(first)
    while True:
        decoded = dec(got)
        ok, pending = _state(p, decoded)
        if ok or not pending or len(got) >= max_new:
            return ok, decoded
        h = runner.hidden([tokens + got])
        got.append(int(runner.logits(h[0, -1:]).argmax(-1)))


def score(runner, items, batch=32, keep=False, progress=None):
    """Exact greedy match, evaluated by teacher forcing on the correct digits; any case whose
    argmax path leaves the target is re-run by true greedy generation (alternative tokenizations).
    Also records digits-only correctness (sum digits right, any terminator)."""
    groups = defaultdict(list)
    for i, p in enumerate(items):
        groups[len(p["tokens"]) + len(p["digit_tokens"])].append(i)
    ok = [False] * len(items)
    dig = [False] * len(items)
    gen = [None] * len(items)
    fallbacks = 0
    finite = True
    done = 0
    for L, idxs in sorted(groups.items()):
        for o in range(0, len(idxs), batch):
            sub = idxs[o:o + batch]
            x = [items[i]["tokens"] + items[i]["digit_tokens"] for i in sub]
            h = runner.hidden(x)
            sel, spans = [], []
            for j, i in enumerate(sub):
                n, m = len(items[i]["tokens"]), len(items[i]["digit_tokens"])
                sel.append(h[j, n - 1:n + m])
                spans.append(m + 1)
            z = runner.logits(torch.cat(sel))
            finite &= bool(torch.isfinite(z).all())
            pred = z.argmax(-1).tolist()
            s = 0
            for i, m1 in zip(sub, spans):
                p = items[i]
                got = pred[s:s + m1]
                s += m1
                exp = p["digit_tokens"]
                mis = next((j for j, (u, v) in enumerate(zip(got, exp)) if u != v), None)
                if mis is None:
                    dig[i] = True
                    decoded = dec(got)
                    good, pending = _state(p, decoded)
                    if not good and pending:
                        fallbacks += 1
                        good, decoded = greedy_generate(runner, p["tokens"], p, got)
                    ok[i] = good
                else:
                    first = exp[:mis] + [got[mis]]
                    decoded = dec(first)
                    good, pending = _state(p, decoded)
                    if pending and len(first) < 12:
                        fallbacks += 1
                        good, decoded = greedy_generate(runner, p["tokens"], p, first)
                    ok[i] = good
                    dig[i] = decoded.startswith(p["digits"])
                gen[i] = decoded
            done += len(sub)
            if progress and done % 1024 < len(sub):
                progress(done, len(items))
    res = {"correct": sum(ok), "count": len(items), "accuracy": sum(ok) / max(1, len(items)),
           "digits_correct": sum(dig), "finite": finite, "fallbacks": fallbacks,
           "failures": [(items[i]["a"], items[i]["b"], gen[i]) for i in range(len(items)) if not ok[i]][:200]}
    if keep:
        res["_ok"], res["_dig"], res["_gen"] = ok, dig, gen
    return res


def digits_only_generate(runner, items, max_new=8, batch=32):
    """Free greedy generation (no teacher forcing) for baseline diagnostics: returns the
    generated strings so a caller can test whether the sum appears with any terminator."""
    outs = []
    for o in range(0, len(items), batch):
        sub = items[o:o + batch]
        groups = defaultdict(list)
        for j, p in enumerate(sub):
            groups[len(p["tokens"])].append(j)
        res = [None] * len(sub)
        for L, js in groups.items():
            cur = [list(sub[j]["tokens"]) for j in js]
            new = [[] for _ in js]
            for _ in range(max_new):
                h = runner.hidden(cur)
                nxt = runner.logits(h[:, -1]).argmax(-1).tolist()
                for r, t in enumerate(nxt):
                    cur[r].append(t)
                    new[r].append(t)
            for r, j in enumerate(js):
                res[j] = dec(new[r])
        outs.extend(res)
    return outs


# ---------------------------------------------------------------- text comparison
def text_compare(run_a, run_b, windows, chunk=32, progress=None):
    """Astra's text definition: window = [BOS] + 64 tokens; inputs are the first 64, targets the
    last 64. Returns top-1 agreement, mean KL(a||b), cross-entropies of both."""
    agree = total = 0
    kl = ce_a = ce_b = 0.0
    finite = True
    per_window = []
    for wi, w in enumerate(windows):
        w = torch.as_tensor(w, dtype=torch.long)[None]
        x, y = w[:, :-1], w[0, 1:]
        ha = run_a.hidden(x)[0]
        hb = run_b.hidden(x)[0]
        wa = 0
        for i in range(0, len(y), chunk):
            za = run_a.logits(ha[i:i + chunk]).double()
            zb = run_b.logits(hb[i:i + chunk]).double()
            finite &= bool(torch.isfinite(za).all() and torch.isfinite(zb).all())
            la, lb = torch.log_softmax(za, -1), torch.log_softmax(zb, -1)
            same = int((za.argmax(-1) == zb.argmax(-1)).sum())
            agree += same
            wa += same
            total += len(za)
            kl += float((la.exp() * (la - lb)).sum())
            yy = y[i:i + chunk, None]
            ce_a -= float(la.gather(1, yy).sum())
            ce_b -= float(lb.gather(1, yy).sum())
        per_window.append(wa)
        if progress and (wi + 1) % 64 == 0:
            progress(wi + 1, len(windows))
    return {"windows": len(windows), "tokens": total, "top1_agreement": agree / total, "mean_kl": kl / total,
            "ce_a": ce_a / total, "ce_b": ce_b / total, "ce_increase": (ce_b - ce_a) / total, "finite": finite,
            "per_window_agree": per_window}


def sample_windows(docs, n, seed, length=64):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = docs[int(rng.integers(len(docs)))]
        s = int(rng.integers(len(d) - length + 1))
        out.append([BOS] + [int(t) for t in d[s:s + length]])
    return out
