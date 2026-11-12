"""Did any text Astra sent through the forward meter come from the held-out files?
Compares 8-word sequences in every inline forward-meter input with WikiText-2 test (rebuilt exactly as
the run cleaned it), WikiText-2 train (calibration) and NLTK's Moby-Dick (a different edition of the
held-out novel). Sherlock Holmes was not available offline and is not checked."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3lib as L

WT = os.environ.get("WIKITEXT2_DIR", os.path.dirname(os.environ.get("WIKITEXT2_TEST", "wikitext-2/test.txt")))
MOBY = os.environ.get("NLTK_MOBY", f"{L.S}/r2/text_src/gutenberg/gutenberg/melville-moby_dick.txt")


def run_clean(text):
    out = []
    for line in text.replace('\r\n', '\n').splitlines():
        line = line.strip()
        if not line or line.startswith('=') or '<unk>' in line:
            continue
        line = line.replace(' @-@ ', '-').replace(' @,@ ', ',').replace(' @.@ ', '.')
        line = re.sub(r'\s+([,.;:!?%\)\]\}])', r'\1', line)
        line = re.sub(r'([\(\[\{])\s+', r'\1', line)
        line = re.sub(r"\s+('(?:s|t|re|ve|m|ll|d))\b", r'\1', line)
        line = re.sub(r'"\s*([^"\n]+?)\s*"', r'"\1"', line)
        out.append(re.sub(r'\s+', ' ', line))
    return '\n\n'.join(out) + '\n'


def shingles(text, n=8):
    w = re.findall(r"[A-Za-z']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


test = shingles(run_clean(open(f"{WT}/test.txt", encoding="utf-8").read()))
train = shingles(run_clean(open(f"{WT}/train.txt", encoding="utf-8").read()))
moby = shingles(open(MOBY, encoding="utf-8", errors="replace").read())
rows = [json.loads(l) for l in open(L.Z / "ledger.jsonl")]
seqs = []
for r in rows:
    if r["event"] == "start" and r["tool"] == "forward":
        q = r["request"]
        q = json.loads(q) if isinstance(q, str) else q
        for s in q.get("tokens") or []:
            seqs.append(s)
text_seqs = [L.dec(s[1:]) for s in seqs]
wordy = [t for t in text_seqs if len(re.findall(r"[A-Za-z']+", t)) >= 8]
hit_test = [t for t in wordy if shingles(t) & test]
test_only = [t for t in hit_test if (shingles(t) & test) - train]
hit_moby = [t for t in wordy if shingles(t) & moby]
res = {"inline_sequences": len(seqs), "with_8_words": len(wordy),
       "share_8_words_with_wikitext_test": len(hit_test),
       "of_which_some_shingle_not_in_wikitext_train": len(test_only),
       "share_8_words_with_moby_dick_nltk": len(hit_moby),
       "examples_test_only": [t[:200] for t in test_only[:5]],
       "examples_moby": [t[:200] for t in hit_moby[:5]]}
print(json.dumps(res, indent=1))
json.dump(res, open(f"{L.S}/r3/out/heldout_overlap.json", "w"), indent=1)
