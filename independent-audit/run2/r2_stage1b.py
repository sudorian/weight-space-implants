"""Stage 1b: exhaustive small-operand regions for G3 and G3-robust (in-spec 3-digit problems
that uniform sampling rarely hits): every a,b < 100, and every a < 1000 with b < 10."""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2lib as L

torch.set_num_threads(2)
torch.set_grad_enabled(False)
t0 = time.time()
host = L.host_state()
out = {}
regions = {
    "a_and_b_below_100": [(a, b) for a in range(100) for b in range(100)],
    "b_below_10_all_a": [(a, b) for a in range(1000) for b in range(10)],
}
for name in ["G3", "G3-robust"]:
    sd, _ = L.apply_delta(host, f"{L.ZIP}/best/{name}/delta.pt")
    m = L.build(sd)
    del sd
    out[name] = {}
    for rname, pairs in regions.items():
        ok, fin = L.exact(m, [L.item(a, b, 3) for a, b in pairs])
        fails = [pairs[i] for i in np.where(~ok)[0]]
        # where do failures sit?
        fa = np.array([p[0] for p in fails]) if fails else np.array([])
        fb = np.array([p[1] for p in fails]) if fails else np.array([])
        out[name][rname] = {"correct": int(ok.sum()), "count": len(pairs), "failures_first_40": fails[:40],
                            "fail_a_range": [int(fa.min()), int(fa.max())] if fails else None,
                            "fail_b_range": [int(fb.min()), int(fb.max())] if fails else None}
        print(f"{name} {rname}: {int(ok.sum())}/{len(pairs)}  fails {fails[:25]}  [{time.time()-t0:.0f}s]", flush=True)
    del m
json.dump(out, open(f"{L.S}/r2/stage1b.json", "w"), indent=1)
print("stage 1b done", flush=True)
