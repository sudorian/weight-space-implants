"""Fresh text for run 3, tokenised with Gemma's tokenizer. No source overlaps the run's
calibration (Frankenstein, Pride and Prejudice, tiny Shakespeare, WikiText-2 train) or its
held-out meter text (Moby-Dick, Sherlock Holmes, WikiText-2 test); WikiText-2 test is rebuilt
separately, with the run's own cleaning rules, only to compare with the meter's numbers."""
import glob, json, os, re, sys, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3lib as L
SRC = f"{L.S}/r2/text_src"
fiction = ["austen-emma", "austen-persuasion", "carroll-alice", "chesterton-thursday", "chesterton-brown",
           "chesterton-ball", "bryant-stories", "edgeworth-parents", "burgess-busterbrown", "whitman-leaves",
           "milton-paradise", "blake-poems", "bible-kjv"]
domains = {
    "fiction": [f"{SRC}/gutenberg/gutenberg/{n}.txt" for n in fiction],
    "web": sorted(glob.glob(f"{SRC}/webtext/webtext/*.txt")),
    "speeches": sorted(glob.glob(f"{SRC}/inaugural/inaugural/*.txt") + glob.glob(f"{SRC}/state_union/state_union/*.txt")),
}
pyfiles = sorted(glob.glob(os.environ.get("PY_STDLIB", "/usr/lib/python3.12") + "/*.py"))
big = sorted(pyfiles, key=lambda p: -os.path.getsize(p))[:40]
domains["code"] = big

def run_clean_wikitext(text):
    # the run's own frozen cleaning rules for WikiText (tools/text.py), re-typed
    text = text.replace('\r\n', '\n').replace('\r', '\n').lstrip('﻿')
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('=') or '<unk>' in line:
            continue
        line = line.replace(' @-@ ', '-').replace(' @,@ ', ',').replace(' @.@ ', '.')
        line = re.sub(r'\s+([,.;:!?%\)\]\}])', r'\1', line)
        line = re.sub(r'([\(\[\{])\s+', r'\1', line)
        line = re.sub(r"\s+('(?:s|t|re|ve|m|ll|d))\b", r'\1', line)
        line = re.sub(r'"\s*([^"\n]+?)\s*"', r'"\1"', line)
        line = re.sub(r'\s+', ' ', line)
        out.append(line)
    return '\n\n'.join(out) + '\n'

tok, meta = {}, {}
for dom, files in domains.items():
    docs = []
    for p in files:
        raw = open(p, encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
        ids = L.enc(raw)
        if len(ids) > 200:
            docs.append(np.asarray(ids, dtype=np.int32))
    tok[dom] = docs
    meta[dom] = {"files": [os.path.basename(p) for p in files], "docs": len(docs), "tokens": int(sum(len(d) for d in docs))}
wt = open(os.environ["WIKITEXT2_TEST"], encoding="utf-8").read()
clean = run_clean_wikitext(wt)
ids = L.enc(clean)
tok["wikitext2_test"] = [np.asarray(ids, dtype=np.int32)]
meta["wikitext2_test"] = {"files": ["test.txt"], "docs": 1, "tokens": len(ids),
    "clean_sha256_matches_run": hashlib.sha256(clean.encode()).hexdigest() == "9884038de91e532bcb8e7ebadc8161d849bfc5ae13429f2ccc6f31e0672c95e1",
    "token_count_matches_run": len(ids) == 6319}
np.savez(f"{L.S}/r3/fresh_tokens.npz", **{f"{d}__{i}": a for d, docs in tok.items() for i, a in enumerate(docs)})
json.dump(meta, open(f"{L.S}/r3/fresh_tokens.json", "w"), indent=1)
for d, m in meta.items():
    print(d, m["docs"], "docs", m["tokens"], "tokens", {k: v for k, v in m.items() if k.endswith("matches_run")})
