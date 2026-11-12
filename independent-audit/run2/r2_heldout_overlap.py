"""Did any explicit text sent through the forward meter come from the held-out corpora?
Checks 8-word shingles of every explicit `texts`/`tokens` input against WikiText-2 test and
Moby-Dick (two of the three held-out books; Sherlock Holmes is not available here), and
against tiny Shakespeare (a calibration file) as a positive control."""
import json
import re
import sys

Z = sys.argv[1]
vocab = json.load(open(f"{Z}/downloads/vocab.json"))
inv = {v: k for k, v in vocab.items()}


def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BD = byte_decoder()


def decode(ids):
    s = "".join(inv[int(i)] for i in ids)
    return bytes(BD[c] for c in s).decode("utf-8", "replace")


def words(t):
    return re.findall(r"[a-z]+", t.lower())


def shingles(t, n=8):
    w = words(t)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


sources = {
    "wikitext2_test (held-out)": open(sys.argv[2], encoding="utf-8").read(),
    "moby_dick (held-out)": open(sys.argv[3], encoding="utf-8", errors="replace").read(),
    "tiny_shakespeare (calibration, control)": open(sys.argv[4], encoding="utf-8").read(),
}
sh = {k: shingles(v) for k, v in sources.items()}
texts = []
for line in open(f"{Z}/ledger.jsonl", "rb"):
    r = json.loads(line)
    if r.get("tool") == "forward" and r.get("event") == "start":
        q = r["request"]
        for t in q.get("texts", []) or []:
            texts.append((r["run_id"], t))
        for ids in q.get("tokens", []) or []:
            if isinstance(ids, list) and ids and isinstance(ids[0], int):
                texts.append((r["run_id"], decode(ids)))
print("explicit inputs:", len(texts), "total words:", sum(len(words(t)) for _, t in texts))
for name, s in sh.items():
    hits = [(rid, sorted(shingles(t) & s)[:2]) for rid, t in texts if shingles(t) & s]
    print(f"{name}: {len(hits)} inputs share an 8-word sequence", hits[:5])
