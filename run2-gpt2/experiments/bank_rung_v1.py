"""Bank an already fully scored sparse checkpoint without changing its weights."""
import argparse,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--rung',required=True);p.add_argument('--candidate',required=True);p.add_argument('--score',required=True);a=p.parse_args()
 src=ROOT/a.candidate;score=json.loads((ROOT/a.score).read_text());manifest=json.loads(src.with_suffix('.manifest.json').read_text())
 assert score['mode']=='full' and score['passed'][a.rung]
 assert score['checkpoint_sha256']==manifest['delta_sha256']
 dest=ROOT/'best'/a.rung;assert not dest.exists();temp=dest.with_name('.'+dest.name+'.tmp');assert not temp.exists();temp.mkdir(parents=True)
 shutil.copyfile(src,temp/'delta.pt');shutil.copyfile(ROOT/a.score,temp/'score.json')
 manifest['banked_from']=a.candidate;manifest['full_score_source']=a.score
 (temp/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 text=score['text'];arith=score['arithmetic'][a.rung]
 (temp/'DESCRIPTION.md').write_text(f"# {a.rung} passing checkpoint\n\nSource: {a.candidate}.\n\nFull exact arithmetic: {arith['correct']}/{arith['count']}. Text agreement {text['top1_agreement']:.10%}; KL {text['mean_kl']:.10f}; CE increase {text['cross_entropy_increase']:.10f} nats.\n\nDirectly programmed format gates, separate operand-token reads, hand-derived decimal carry, and a fixed-unembedding parabolic digit readout. No arithmetic-answer fitting or gradients. See the manifest and REPORT.md for derivation, numerical/noise limitations and current verification status.\n")
 os.replace(temp,dest);print(dest)
if __name__=='__main__':main()
