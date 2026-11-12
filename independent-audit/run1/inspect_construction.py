import json
import sys

c = json.load(open(sys.argv[1]))
keywords = ("calib", "probe", "artifact", "threshold", "fit", "solve", "regress", "lstsq", "pinv", "example", "arith", "text", "detector", "bound", "source", "prompt", "carry_", "training")


def short(v):
    s = json.dumps(v)
    return s if len(s) <= 170 else s[:170] + f"... [{len(s)} chars]"


seen = 0


def walk(o, path=""):
    global seen
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{path}/{k}"
            if any(w in k.lower() for w in keywords):
                print(p, "=", short(v))
                seen += 1
            if isinstance(v, (dict, list)):
                walk(v, p)
    elif isinstance(o, list) and o and isinstance(o[0], dict):
        walk(o[0], path + "[0]")


for k in c:
    print("TOP", k, type(c[k]).__name__)
walk(c)
print("printed", seen)
