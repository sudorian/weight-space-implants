"""Build fresh text domains that no run used (calib: WikiText-2 train, tiny Shakespeare,
Pride and Prejudice, Frankenstein; heldout: WikiText-2 test, Moby-Dick, Sherlock Holmes),
plus WikiText-2 test itself for comparison with Astra's held-out numbers."""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2lib as L

SRC = f"{L.S}/r2/text_src"
fiction = ["austen-emma", "austen-persuasion", "carroll-alice", "chesterton-thursday", "chesterton-brown",
           "chesterton-ball", "bryant-stories", "edgeworth-parents", "burgess-busterbrown", "whitman-leaves",
           "milton-paradise", "blake-poems", "bible-kjv"]
domains = {
    "fiction": [f"{SRC}/gutenberg/gutenberg/{n}.txt" for n in fiction],
    "web": sorted(p for p in glob.glob(f"{SRC}/webtext/webtext/*.txt")),
    "speeches": sorted(glob.glob(f"{SRC}/inaugural/inaugural/*.txt") + glob.glob(f"{SRC}/state_union/state_union/*.txt")),
}
pyfiles = sorted(glob.glob("/usr/lib/python3.12/*.py"))
big = sorted(pyfiles, key=lambda p: -os.path.getsize(p))[:40]
assert len({os.path.basename(p) for p in big}) == len(big)
domains["code"] = big


def wikitext_clean(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("=") or "<unk>" in line:
            continue
        line = line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        line = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", line)
        line = re.sub(r"([\(\[\{])\s+", r"\1", line)
        line = re.sub(r"\s+('(?:s|t|re|ve|m|ll|d))\b", r"\1", line)
        line = re.sub(r"\s+", " ", line)
        out.append(line)
    return "\n\n".join(out) + "\n"


tok = {}
meta = {}
for dom, files in domains.items():
    docs = []
    for p in files:
        raw = open(p, encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
        ids = L.enc(raw)
        if len(ids) > 200:
            docs.append(np.asarray(ids, dtype=np.int32))
    tok[dom] = docs
    meta[dom] = {"files": [os.path.basename(p) for p in files], "docs": len(docs), "tokens": int(sum(len(d) for d in docs))}
wt = open(os.environ.get("WIKITEXT2_TEST", "wikitext-2/test.txt"), encoding="utf-8").read()
tok["wikitext2_test"] = [np.asarray(L.enc(wikitext_clean(wt)), dtype=np.int32)]
meta["wikitext2_test"] = {"files": ["test.txt (pytorch/examples acc295d)"], "docs": 1, "tokens": int(len(tok["wikitext2_test"][0]))}
np.savez(f"{L.S}/r2/fresh_tokens.npz", **{f"{d}__{i}": a for d, docs in tok.items() for i, a in enumerate(docs)})
json.dump(meta, open(f"{L.S}/r2/fresh_tokens.json", "w"), indent=1)
for d, m in meta.items():
    print(d, m["docs"], "docs", m["tokens"], "tokens", m["files"][:6])
