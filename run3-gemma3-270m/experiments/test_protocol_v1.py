"""Pre-freeze synthetic tests of scoring semantics and strict host reconstruction.

The tiny deterministic mock is a scorer fixture, not a Gemma implementation.
This script never runs Gemma inference or opens heldout data.
"""
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import ELIGIBLE, load_model, sample_windows
from tools.score import carry_case, continuation_state, exact_greedy, make_problem, random_operand
from tools.text import clean
from tools.tokenizer import Tokenizer
import numpy as np

torch.set_num_threads(6)
torch.set_grad_enabled(False)
tok = Tokenizer()

class MockHead(nn.Module):
    def forward(self, h):
        result = torch.full((*h.shape[:-1], tok.vocab_size), -10.)
        result.scatter_(-1, h[..., :1].long(), 10.)
        return result

class MockModel(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.model = nn.Module()
        self.model.norm = nn.Identity()
        self.lm_head = MockHead()
        self.mapping = mapping

    def forward(self, input_ids, **kwargs):
        h = torch.zeros(*input_ids.shape, 640)
        for i, row in enumerate(input_ids.tolist()):
            for j in range(len(row)):
                h[i, j, 0] = self.mapping.get(tuple(row[:j+1]), 1)
        h = self.model.norm(h)
        return type('Output', (), {'logits': self.lm_head(h[:, -1:])})()


items, mapping = [], {}
spec = [('H1', 111, 222, '333\n', True, True),
        ('H1', 112, 222, '335', False, False),
        ('H1', 113, 222, '335 ', False, True),
        ('H1', 114, 222, '3360', False, True),
        ('H5', 115, 222, '337 is the sum', True, True)]
for rung, a, b, response, ok, digits_ok in spec:
    p = make_problem(a, b, rung, tok)
    items.append(p)
    sequence = p['tokens'][:]
    for new in tok.encode(response):
        mapping[tuple(sequence)] = new
        sequence.append(new)
# Deliberate alternate byte-token spelling of a correct leading digit.
p = make_problem(116, 222, 'H1', tok)
items.append(p)
byte = tok.raw.token_to_id('<0x33>')
assert tok.decode([byte]) == '3'
sequence = p['tokens'][:]
for new in [byte]+tok.encode('38\n'):
    mapping[tuple(sequence)] = new
    sequence.append(new)
spec.append(('H1', 116, 222, '338\n', True, True))
result = exact_greedy(MockModel(mapping), items, batch_size=2, return_mask=True)
assert result['_mask'] == [s[4] for s in spec], result
assert result['digits_correct_ignoring_terminator'] == sum(s[5] for s in spec)
assert result['alternative_tokenization_fallbacks'] == 1
assert result['count'] == 6 and result['correct'] == 3
assert tok.chat([{'role': 'user', 'content': 'What is 123 + 456?'}])[0] == 2
assert clean(' = Heading =\n a @-@ b , c .\nbad <unk>\n', 'wikitext') == 'a-b, c.\n'
assert clean('header\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nfooter', 'novel') == 'body\n'
rng = np.random.default_rng(451)
for k in (3, 6):
    for _ in range(100):
        a, b = carry_case(rng, k)
        assert len(str(a)) == k == len(str(b))
        assert a % 10 + b % 10 >= 10
        assert len(str(random_operand(rng, k))) == k
windows = sample_windows([np.arange(10, 100)], 4, 64, 31)
assert all(len(w) == 64 and w[0] == 2 for w in windows)
model, stats = load_model()
assert stats['eligible_scalars'] == ELIGIBLE and stats['changed_scalars'] == 0
assert model.config._attn_implementation == 'sdpa'
assert not any(p.requires_grad for p in model.parameters())
assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
print(json.dumps({'scorer_fixture': 'passed', 'tokenizer_and_cleaning': 'passed',
                  'operand_and_window_invariants': 'passed', 'strict_host_reconstruction': 'passed'}))
