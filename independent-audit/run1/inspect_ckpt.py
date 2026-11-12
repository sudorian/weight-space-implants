import json, sys, os, hashlib
import torch

root = sys.argv[1]
for rung in ["R1", "R2", "R3", "R4"]:
    p = os.path.join(root, "best", rung, "ckpt.pt")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    ck = torch.load(p, map_location="cpu", weights_only=True)
    print("==", rung, h[:16], type(ck).__name__)
    if isinstance(ck, dict):
        for k, v in ck.items():
            if isinstance(v, dict):
                print("  key", k, "dict", len(v), list(v.keys())[:6])
            elif isinstance(v, (list, str, int, float)):
                s = repr(v)
                print("  key", k, type(v).__name__, s[:160])
            else:
                print("  key", k, type(v).__name__)
        sd = ck.get("model", ck.get("state_dict", ck))
        tot = 0
        rows = []
        for k, v in sd.items():
            if torch.is_tensor(v):
                tot += v.numel()
                rows.append((k, tuple(v.shape), str(v.dtype), float(v.abs().max()), int((~torch.isfinite(v)).sum())))
        print("  tensors", len(rows), "total entries", tot)
        for r in rows:
            print("   ", r)

c = json.load(open(os.path.join(root, "best", "R4", "construction.json")))


def walk(o, prefix="", depth=0):
    if depth > 2:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            t = type(v).__name__
            size = len(v) if isinstance(v, (list, dict, str)) else v
            print("  " * depth + f"{k}: {t} {size if not isinstance(size,str) else size[:120]}")
            walk(v, prefix + "/" + k, depth + 1)


print("== R4 construction.json top level")
walk(c)
