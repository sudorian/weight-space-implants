"""Offline GPT-2 byte BPE, with a stdlib implementation of its Unicode pretokenizer."""
from functools import lru_cache
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]

def byte_encoder():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + len(cs) - 188)
    # The base printable-byte set has exactly 188 elements.
    return dict(zip(bs, map(chr, cs)))

def category(c):
    cat = unicodedata.category(c)[0]
    return cat if cat in ('L', 'N') else ('W' if c.isspace() else 'P')

def pieces(text):
    """Equivalent alternatives: contractions, optional-space L/N/P runs, whitespace."""
    i, n = 0, len(text)
    while i < n:
        contraction = next((s for s in ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
                            if text.startswith(s, i)), None)
        if contraction:
            yield contraction
            i += len(contraction)
            continue
        start = i
        j = i + (text[i] == ' ' and i + 1 < n and not text[i + 1].isspace())
        cat = category(text[j])
        if cat != 'W':
            j += 1
            while j < n and category(text[j]) == cat:
                j += 1
        else:
            j = i
            while j < n and text[j].isspace():
                j += 1
            # Greedy whitespace with (?!\S) leaves its last character before nonspace.
            if j < n and j - i > 1:
                j -= 1
        yield text[start:j]
        i = j

class Tokenizer:
    def __init__(self):
        self.encoder = json.loads((ROOT / 'downloads/vocab.json').read_text())
        self.decoder = {v:k for k,v in self.encoder.items()}
        rows = (ROOT / 'downloads/merges.txt').read_text().splitlines()
        self.ranks = {tuple(s.split()):i for i,s in enumerate(rows[1:]) if s.strip()}
        self.byte_encoder = byte_encoder()
        self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}

    @lru_cache(maxsize=200000)
    def bpe(self, token):
        word = tuple(token)
        while len(word) > 1:
            pair = min(zip(word, word[1:]), key=lambda p:self.ranks.get(p, float('inf')))
            if pair not in self.ranks:
                break
            out, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and word[i:i+2] == pair:
                    out.append(word[i] + word[i+1]); i += 2
                else:
                    out.append(word[i]); i += 1
            word = tuple(out)
        return word

    def encode(self, text):
        return [self.encoder[b] for piece in pieces(text)
                for b in self.bpe(''.join(self.byte_encoder[v] for v in piece.encode('utf-8')))]

    def decode(self, ids):
        s = ''.join(self.decoder[int(i)] for i in ids)
        return bytes(self.byte_decoder[c] for c in s).decode('utf-8', errors='replace')
