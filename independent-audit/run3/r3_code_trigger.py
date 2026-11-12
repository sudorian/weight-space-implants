"""For the 10 worst code windows: where does the damage start, and what precedes it?"""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3lib as L
import r3_stages as R
torch.set_grad_enabled(False)
sd, ed, st = R.models()
dom = R.text_domains()
w = L.sample_windows(dom["code"], 384, R.FRESH_SEED + 105)
worst = [p[0] for p in json.load(open(f"{L.S}/r3/out/code_diag.json"))["worst_windows"]]
host, edit = L.Own(sd), L.Own(ed)
out = []
for wi in worst:
    win = w[wi]
    x = torch.tensor(win[:-1])[None]
    za, zb = host.logits(host.hidden(x)[0]).double(), edit.logits(edit.hidden(x)[0]).double()
    la, lb = torch.log_softmax(za, -1), torch.log_softmax(zb, -1)
    kl = (la.exp() * (la - lb)).sum(-1)
    bad = torch.nonzero(kl > 1).flatten().tolist()
    first = bad[0]
    out.append({"window": wi, "first_bad_pos": first, "bad_positions": len(bad),
                "bad_after_first": sum(1 for b in bad if b > first) / max(1, 63 - first),
                "text_up_to_first_bad": L.dec(win[max(1, first - 14):first + 1]),
                "a313_prediction_there": L.dec([int(zb[first].argmax())]), "host_prediction": L.dec([int(za[first].argmax())])})
    print(json.dumps(out[-1]), flush=True)
json.dump(out, open(f"{L.S}/r3/out/code_trigger.json", "w"), indent=1)
