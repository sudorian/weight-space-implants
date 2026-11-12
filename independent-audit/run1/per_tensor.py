import sys
import torch

host = torch.load(sys.argv[1], map_location="cpu", weights_only=True)["model"]
for rung in ["R1", "R2", "R3", "R4"]:
    m = torch.load(f"{sys.argv[2]}/{rung}/ckpt.pt", map_location="cpu", weights_only=True)["model"]
    print(rung)
    for n, v in host.items():
        if n == "lm_head.weight":
            continue
        c = int((m[n] != v).sum())
        if c:
            mx = float(m[n].abs().max())
            print(f"   {n:38s} changed {c:7d}/{v.numel():7d}  max|w| {mx:.3g}")
