import collections
import hashlib
import json
import sys

path = sys.argv[1]
raw = open(path, "rb").read().split(b"\n")
if raw and raw[-1] == b"":
    raw = raw[:-1]
rows = [json.loads(r) for r in raw]
print("lines", len(rows))
print("first row keys", list(rows[0].keys()))
print("first row", json.dumps(rows[0])[:400])

prev_key = None
for k in rows[1].keys():
    if "prev" in k.lower():
        prev_key = k
print("prev key:", prev_key)
bad = 0
if prev_key:
    for i in range(1, len(rows)):
        h = hashlib.sha256(raw[i - 1]).hexdigest()
        if rows[i].get(prev_key) != h:
            bad += 1
            if bad < 5:
                print("chain mismatch at", i, rows[i].get(prev_key), h)
print("chain mismatches", bad)

c = collections.Counter()
for r in rows:
    c[(r.get("tool"), r.get("event") or r.get("status") or r.get("phase") or r.get("type"))] += 1
for k, v in sorted(c.items(), key=lambda x: str(x[0])):
    print(k, v)

print("time span", rows[0].get("time") or rows[0].get("timestamp"), rows[-1].get("time") or rows[-1].get("timestamp"))
print("last row", json.dumps(rows[-1])[:600])
