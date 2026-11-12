"""Why does A313 wreck some Python-code windows? Same 384 code windows as stage 2; per-token KL,
the worst windows, and the context right before each wrecked prediction."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3lib as L
import r3_stages as R

torch.set_grad_enabled(False)
sd, ed, st = R.models()
dom = R.text_domains()
w = L.sample_windows(dom["code"], 384, R.FRESH_SEED + 105)
host, edit = L.HFRunner(L.hf_model(sd)), L.HFRunner(L.hf_model(ed))
per = []
bad_tokens = []
for wi, win in enumerate(w):
    x = torch.tensor(win[:-1])[None]
    za = host.logits(host.hidden(x)[0]).double()
    zb = edit.logits(edit.hidden(x)[0]).double()
    la, lb = torch.log_softmax(za, -1), torch.log_softmax(zb, -1)
    kl = (la.exp() * (la - lb)).sum(-1)
    agree = (za.argmax(-1) == zb.argmax(-1)).float()
    per.append((wi, float(kl.mean()), float(agree.mean())))
    for i in torch.nonzero(kl > 1).flatten().tolist():
        bad_tokens.append({"window": wi, "pos": i, "kl": float(kl[i]),
                           "context": L.dec(win[max(1, i - 11):i + 1]),
                           "host_next": L.dec([int(za[i].argmax())]), "a313_next": L.dec([int(zb[i].argmax())]),
                           "a313_max_logit": float(zb[i].max()), "host_max_logit": float(za[i].max())})
    if (wi + 1) % 64 == 0:
        print("windows", wi + 1, flush=True)
kls = np.array([p[1] for p in per])
res = {"windows": len(w), "mean_kl": float(kls.mean()),
       "windows_mean_kl_over_1": int((kls > 1).sum()), "windows_mean_kl_over_0.1": int((kls > 0.1).sum()),
       "share_of_total_kl_from_worst_10_windows": float(np.sort(kls)[-10:].sum() / kls.sum()),
       "tokens_with_kl_over_1": len(bad_tokens),
       "worst_windows": sorted(per, key=lambda p: -p[1])[:10],
       "examples": sorted(bad_tokens, key=lambda b: -b["kl"])[:25]}
ctx = [b["context"] for b in bad_tokens]
res["bad_token_context_has_plus_or_equals"] = sum(("+" in c or "=" in c) for c in ctx)
res["bad_token_context_has_digit"] = sum(any(ch.isdigit() for ch in c) for c in ctx)
json.dump(res, open(f"{L.S}/r3/out/code_diag.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))
for b in res["examples"][:12]:
    print(round(b["kl"], 1), repr(b["context"]), "| host:", repr(b["host_next"]), "| a313:", repr(b["a313_next"]), "| a313 max logit", round(b["a313_max_logit"]))
