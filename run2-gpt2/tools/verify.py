"""Run once at finalization. Re-test each bank with score.py's fresh internal seed."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha,source_check,ledger_rows

def main():
    source_check(); ledger_rows()
    dest=ROOT/'verification'; dest.mkdir(exist_ok=True)
    marker=dest/'started.json'; assert not marker.exists(), 'Final verification is one-shot'
    banks=sorted((ROOT/'best').glob('G[1-5]/delta.pt'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in banks}
    atomic_json(marker,{'banked_sha256':hashes})
    results={}
    for p in banks:
        rung=p.parent.name; output=dest/f'{rung}.json'
        subprocess.run([sys.executable,str(ROOT/'tools/score.py'),'--delta',str(p),
                        '--rungs',rung,'--fresh','--output',str(output)],check=True,cwd=ROOT)
        results[rung]=json.loads(output.read_text())
    assert all(sha(ROOT/p)==digest for p,digest in hashes.items()), 'Bank changed during verification'
    source_check(); rows,head=ledger_rows()
    atomic_json(dest/'results.json',{'rungs':results,'all_banked_pass':all(r['passed'][k] for k,r in results.items()),
        'banked_sha256':hashes,'ledger_head_sha256':head,'no_banked_rungs':not bool(banks)})
    print(json.dumps({'verified':list(results),'all_banked_pass':all(r['passed'][k] for k,r in results.items())}))
if __name__=='__main__': main()
