"""Trace each saved checkpoint's build chain through manifests, verify source hashes,
and scan the scripts for forbidden or disclosure-worthy patterns."""
import hashlib
import json
import re
import sys
from pathlib import Path

Z = Path(sys.argv[1])


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest_for(delta_rel):
    p = Z / delta_rel
    for cand in (p.with_suffix(".manifest.json"), p.parent / "manifest.json"):
        if cand.exists():
            return cand
    return None


def chain(delta_rel, seen=None):
    seen = seen if seen is not None else {}
    m = manifest_for(delta_rel)
    if m is None or delta_rel in seen:
        return seen
    man = json.loads(m.read_text())
    seen[delta_rel] = man
    for inp in man.get("inputs", {}):
        if inp.endswith("delta.pt"):
            chain(inp, seen)
    return seen


banks = {b: f"best/{b}/delta.pt" for b in ["G1", "G2", "G3", "G3-robust", "G4-best-attempt"]}
all_sources = {}
summary = {}
for b, rel in banks.items():
    c = chain(rel)
    srcs = {}
    for d, man in c.items():
        for s, h in man.get("sources", {}).items():
            srcs[s] = h
    bad = []
    missing = []
    for s, h in srcs.items():
        p = Z / s
        if not p.exists():
            missing.append(s)
        elif sha(p) != h:
            bad.append(s)
    summary[b] = {"chain_length": len(c), "sources": len(srcs), "hash_mismatch": bad, "missing": missing}
    all_sources.update(srcs)
print(json.dumps(summary, indent=1))
print("union of build sources:", len(all_sources))

patterns = {
    "gradient/training": r"\.backward\(|torch\.autograd|torch\.optim|requires_grad_\(\s*True|requires_grad\s*=\s*True|\.grad\b|enable_grad|torch\.func|jacrev|jacfwd|vjp\(",
    "finite differences / search": r"finite.?diff|numerical.?grad|hill.?climb|random.?search|scipy\.optimize|minimize\(|nevergrad|cma\b|bisect\(",
    "model built outside meters": r"from model import|import model\b|GPT\(GPTConfig|GPTConfig\(|load_model\(|nanogpt",
    "official test generator or seeds": r"tools\.score|from tools import score|problems\(|_SEED|_FRESH_SEED|18419253|690271447|heldout",
    "closed-form solves": r"linalg\.lstsq|linalg\.solve|linalg\.pinv|torch\.pinverse|\blstsq\b|cholesky_solve|nnls",
    "uses of sums/answers": r"\ba\s*\+\s*b\b|answer|a_plus_b|sum_digits|target_tokens|correct_sum|expected_sum",
    "layer computations on recorded activations": r"layer_norm\(|F\.gelu|functional\.gelu|softmax\(|scaled_dot_product",
}
hits = {k: [] for k in patterns}
for s in sorted(all_sources):
    p = Z / s
    if not p.exists() or p.suffix != ".py":
        continue
    text = p.read_text(errors="replace")
    for k, pat in patterns.items():
        for m in re.finditer(pat, text):
            line_no = text.count("\n", 0, m.start()) + 1
            line = text.splitlines()[line_no - 1].strip()
            hits[k].append((s, line_no, line[:200]))
for k, v in hits.items():
    files = sorted({h[0] for h in v})
    print(f"\n[{k}] {len(v)} hits in {len(files)} files")
    for h in v[:14]:
        print("   ", h[0], h[1], "|", h[2][:170])
json.dump({"summary": summary, "sources": sorted(all_sources), "hits": hits}, open(sys.argv[2], "w"), indent=1)
