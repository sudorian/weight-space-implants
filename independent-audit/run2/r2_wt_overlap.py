"""For forward inputs that share 8-word sequences with WikiText-2 test, check whether
those sequences also occur in WikiText-2 train (calibration). Sequences found only in
test would mean held-out text reached the forward meter."""
import json
import re
import sys

sys.argv = sys.argv
exec(open(sys.argv[5]).read().split("sources = {")[0])  # reuse decode/shingles helpers
Z = sys.argv[1]
test = shingles(open(sys.argv[2], encoding="utf-8").read())
train = shingles(open(sys.argv[3], encoding="utf-8").read())
only_test = test - train
print("8-word sequences: test", len(test), "train", len(train), "test-only", len(only_test))
hits = []
for line in open(f"{Z}/ledger.jsonl", "rb"):
    r = json.loads(line)
    if r.get("tool") == "forward" and r.get("event") == "start":
        q = r["request"]
        ins = list(q.get("texts", []) or [])
        ins += [decode(ids) for ids in (q.get("tokens", []) or []) if isinstance(ids, list) and ids and isinstance(ids[0], int)]
        for t in ins:
            s = shingles(t)
            if s & only_test:
                hits.append((r["run_id"], sorted(s & only_test)[:3], t[:160]))
print("inputs sharing a TEST-ONLY 8-word sequence:", len(hits))
for h in hits[:12]:
    print(" ", h)
