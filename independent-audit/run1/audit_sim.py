import json
import re
import sys

lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
calls = []
for l in lines:
    o = json.loads(l)
    p = o.get("payload", {})
    if o.get("type") == "response_item" and p.get("type") in ("function_call", "custom_tool_call"):
        a = p.get("arguments") or p.get("input") or ""
        calls.append((o.get("timestamp", "")[11:19], a))

# split each exec into heredoc script bodies, decode escapes crudely
scripts = []
for ts, a in calls:
    t = a.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
    for m in re.finditer(r"cat > ([\w/\.\-]+) <<\s*'?(\w+)'?\n(.*?)\n\2", t, re.S):
        scripts.append((ts, m.group(1), m.group(3)))
    for m in re.finditer(r"python[^\n]* - <<\s*'?(\w+)'?\n(.*?)\n\1", t, re.S):
        scripts.append((ts, "<inline>", m.group(2)))
print("scripts/heredocs found", len(scripts))
loads = re.compile(r"torch\.load\(")
ckpt = re.compile(r"experiments/\w+\.pt|host/ckpt\.pt|args\.base|a\.base|\.checkpoint")
sim = re.compile(r"\.softmax\(|F\.gelu|gelu\(|\bmodel\(|\.lm_head|def forward")
hits = []
for ts, name, body in scripts:
    if not (loads.search(body) and sim.search(body)):
        continue
    usesact = "activations" in body
    usesweights = bool(re.search(r"\['model'\]|\[\"model\"\]|c_attn\.weight|c_fc\.weight", body))
    nsoft = len(re.findall(r"\.softmax\(", body))
    ngelu = len(re.findall(r"gelu\(", body))
    hits.append((ts, name, len(body), usesact, usesweights, nsoft, ngelu))
print("with torch.load plus softmax/gelu:", len(hits))
for h in hits[:60]:
    print(h)
print()
# print the contexts for gelu( calls and for softmax combined with candidate weights
n = 0
for ts, name, body in scripts:
    if "gelu(" in body and loads.search(body):
        for m in re.finditer(r"gelu\(", body):
            s = max(0, m.start() - 220)
            print("GELU", ts, name, repr(body[s:m.end() + 120])[:430])
            n += 1
            if n > 12:
                break
    if n > 12:
        break

# thresholds provenance
print()
for ts, a in calls:
    i = a.find(".225")
    if i >= 0 and ("thresholds" in a or "window" in a):
        print("THRESH", ts, repr(a[max(0, i - 300) : i + 200])[:620])
        break
for l in lines:
    o = json.loads(l)
    p = o.get("payload", {})
    if o.get("type") == "response_item" and p.get("type") == "message":
        txt = json.dumps(p.get("content"))
        if "threshold" in txt and (".42" in txt or "0.42" in txt):
            print("MSG", o.get("timestamp", "")[11:19], txt[:600])
            break
