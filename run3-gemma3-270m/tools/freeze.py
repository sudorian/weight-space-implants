"""One-time sealing of host, sources, calibration and protocol. No inference."""
import datetime
import json
from pathlib import Path
import sys
from transformers.models.gemma3 import modeling_gemma3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import LIMITS, append_event, atomic_json, sha


def main():
    dst = ROOT / 'tools/frozen.json'
    assert not dst.exists(), 'Already frozen'
    paths = ['GOAL.md', 'host/manifest.json', 'host/text_manifest.json', 'host/calib_tokens.npz',
             'downloads/config.json', 'downloads/tokenizer.json', 'downloads/chat_template.jinja']
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'tools').glob('*.py'))]
    manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host_sha256': sha(ROOT / 'host/ckpt.pt'),
        'modeling_gemma3_path': str(Path(modeling_gemma3.__file__).relative_to(ROOT)),
        'modeling_gemma3_sha256': sha(modeling_gemma3.__file__),
        'sources': {p: sha(ROOT / p) for p in paths},
        'calib': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'calib').glob('*.txt'))},
        'budgets': LIMITS, 'forward_generation_cost': 'One unit per new token; one per activation pass',
        'multi_rung_cost': 'One score unit per rung; shared text measurement',
        'scoring': 'Canonical correct-prefix comparison plus autoregressive alternative-tokenization fallback, maximum 12 new tokens',
        'digits_only': 'Correct sum digit prefix irrespective of the following token',
        'text': '64 next-token targets after BOS in each independently reset 64-token text window',
        'noise': 'Independent multiplicative Gaussian on changed scalars; 2000 arithmetic problems per sigma',
        'verification': 'Fresh internal seed; one full score per bank; no edits afterwards'}
    atomic_json(dst, manifest)
    append_event({'event': 'freeze', 'tool': 'freeze', 'request': {},
                  'checkpoint_sha256': manifest['host_sha256'],
                  'result': {'manifest_sha256': sha(dst), 'source_count': len(paths)}})
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
