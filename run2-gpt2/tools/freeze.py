"""One-time Phase-0 seal. No inference, edits, or access to heldout text."""
import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,append_event,sha

def main():
    dst=ROOT/'tools/frozen.json'; assert not dst.exists(), 'Already frozen'
    paths=['model.py','GOAL.md','host/manifest.json','host/text_manifest.json',
           'host/calib_tokens.npz','downloads/vocab.json','downloads/merges.txt']
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'tools').glob('*.py'))]
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'host_sha256':sha(ROOT/'host/ckpt.pt'),
              'sources':{p:sha(ROOT/p) for p in paths},
              'calib':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'calib').glob('*.txt'))},
              'budgets':{'forward':300,'quick':100,'full':40},
              'forward_generation_cost':'one forward budget unit per generation step',
              'multi_rung_score_cost':'one score budget unit per rung; shared text evaluation',
              'spaced_scoring':'all correct-prefix argmax matches, exactly equivalent to greedy sequence equality',
              'text_windows':'64 inputs with 64 next-token targets; contexts reset per window',
              'noise':'independent multiplicative Gaussian noise on each changed scalar only'}
    atomic_json(dst,manifest)
    append_event({'event':'freeze','tool':'freeze','request':{},'checkpoint_sha256':manifest['host_sha256'],
                  'result':{'manifest_sha256':sha(dst),'source_count':len(paths)}})
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
