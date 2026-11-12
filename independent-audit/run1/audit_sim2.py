import collections
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

scripts = []
for ts, a in calls:
    t = a.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
    for m in re.finditer(r"cat > ([\w/\.\-]+) <<\s*'?(\w+)'?\n(.*?)\n\2", t, re.S):
        scripts.append((ts, m.group(1), m.group(3)))
    for m in re.finditer(r"python[^\n]* - <<\s*'?(\w+)'?\n(.*?)\n\1", t, re.S):
        scripts.append((ts, "<inline>", m.group(2)))

# what is softmax applied to, in every script that loads something
kinds = collections.Counter()
examples = {}
for ts, name, body in scripts:
    if "torch.load(" not in body:
        continue
    for m in re.finditer(r"([^\n;]{0,120})\.softmax\(", body):
        ctx = m.group(1)
        key = re.sub(r"\d+", "N", ctx.strip())[-44:]
        kinds[key] += 1
        examples.setdefault(key, (ts, name, body[max(0, m.start() - 160): m.end() + 70]))
print("distinct softmax contexts", len(kinds))
for k, v in kinds.most_common(40):
    print(v, repr(k))
print()
# flag scripts that multiply a *weight* tensor with recorded activations AND softmax the result
flag = 0
for ts, name, body in scripts:
    if "torch.load(" not in body or ".softmax(" not in body:
        continue
    weightuse = re.search(r"(c_attn|c_proj|c_fc|lm_head|wte)\.weight|\bstate_dict\b|\['model'\]|\[\"model\"\]|\bs\[", body)
    if weightuse:
        flag += 1
        print("=== WEIGHT+SOFTMAX", ts, name, len(body))
        print(body[:1900])
        print()
print("scripts combining weights and softmax:", flag)

print()
for ts, a in calls:
    if "thresholds" in a:
        for m in re.finditer(r"thresholds\s*=\s*\[[^\]]{1,80}\]", a):
            print("THRESH-DEF", ts, m.group(0))
