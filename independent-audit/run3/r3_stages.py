"""Run 3 re-test stages. Usage: r3_stages.py <stage> ; results go to r3/out/<stage>.json"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3gen as G
import r3lib as L

torch.set_grad_enabled(False)
OUT = f"{L.S}/r3/out"
os.makedirs(OUT, exist_ok=True)
if os.environ.get("QUICK"):
    OUT = f"{L.S}/r3/out_quick"
    os.makedirs(OUT, exist_ok=True)
    _score, _text = L.score, L.text_compare
    L.score = lambda runner, items, **k: _score(runner, items[:24], **k)
    L.text_compare = lambda a, b, w, **k: _text(a, b, w[:2], **k)
    _gen = L.digits_only_generate
    L.digits_only_generate = lambda r, items, **k: _gen(r, items[:6], **k)
FRESH_SEED = 20261111
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def prog(done, n):
    log(f"  {done}/{n}")


def save(name, obj):
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    log("saved", name)


def strip(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def models():
    sd = L.host_state(check_hash=True)
    assert L.sha(L.BEST) == L.BEST_SHA
    ed, st = L.apply_delta(sd, L.load_delta())
    return sd, ed, st


def group_order_failures(items, ok, n=20):
    order = sorted(range(len(ok)), key=lambda i: (len(items[i]["tokens"]) + len(items[i]["digit_tokens"]), i))
    return [(items[i]["a"], items[i]["b"]) for i in order if not ok[i]][:n]


ASTRA_FAILS = [(253, 696), (639, 290), (101, 698), (610, 379), (700, 276), (465, 214), (163, 826), (380, 349),
               (149, 490), (679, 290), (145, 594), (209, 290), (700, 266), (111, 688), (201, 669), (102, 157),
               (102, 117), (140, 409), (149, 270), (201, 628)]


def stage0():
    """Reproduce every arithmetic number in Astra's score for A313 on the run's own seed, HF path."""
    sd, ed, st = models()
    res = {"edit_stats": {k: v for k, v in st.items() if k != "per_tensor"},
           "per_tensor": st["per_tensor"]}
    hf = L.HFRunner(L.hf_model(ed))
    items = [L.make(a, b) for a, b in G.standard_h1()]
    r = L.score(hf, items, keep=True, progress=prog)
    fails = group_order_failures(items, r["_ok"])
    res["h1_standard"] = strip(r) | {"first20_in_meter_order": fails, "first20_match_astra": fails == ASTRA_FAILS,
                                     "astra_correct": 24961}
    np.save(f"{OUT}/s0_ok_hf.npy", np.array(r["_ok"]))
    log("H1 standard", r["correct"], r["digits_correct"], fails == ASTRA_FAILS)
    # two in a row, re-built from the run's recipe (seed + 171987, first 4000 problems)
    src = G.standard_h1(G.STANDARD_SEED + 171987)[:4000]
    first = [L.make(a, b) for a, b in src[:2000]]
    second = [L.make(b2[0], b2[1], before=f["tokens"] + f["digit_tokens"] + [L.NL])
              for f, b2 in zip(first, src[2000:])]
    one = L.score(hf, first, keep=True)
    two = L.score(hf, second, keep=True)
    joint = sum(x and y for x, y in zip(one["_ok"], two["_ok"]))
    res["two_in_a_row"] = {"first": one["correct"], "second": two["correct"], "both": joint,
                           "astra": [1997, 1, 1], "second_examples": two["failures"][:10]}
    log("two in a row", one["correct"], two["correct"], joint)
    del hf
    # noise, re-built from the run's recipe (first 2000 standard problems, seed + int(sigma*1e9))
    res["noise"] = {}
    for sigma, astra in ((1e-5, 1997), (1e-4, 1997), (1e-3, 1602)):
        noisy, _ = L.apply_delta(sd, L.load_delta(), noise=sigma, seed=G.STANDARD_SEED + int(sigma * 1e9), check=False)
        rr = L.score(L.HFRunner(L.hf_model(noisy)), items[:2000])
        res["noise"][str(sigma)] = {"correct": rr["correct"], "digits": rr["digits_correct"], "astra": astra}
        log("noise", sigma, rr["correct"], "astra", astra)
        del noisy
    save("stage0", res)


def stage0b():
    """Same 25,000 standard problems through the from-scratch forward pass; compare per problem."""
    sd, ed, st = models()
    own = L.Own(ed)
    items = [L.make(a, b) for a, b in G.standard_h1()]
    r = L.score(own, items, keep=True, progress=prog)
    hf_ok = np.load(f"{OUT}/s0_ok_hf.npy").tolist()
    diff = [(items[i]["a"], items[i]["b"], hf_ok[i], r["_ok"][i]) for i in range(len(items)) if hf_ok[i] != r["_ok"][i]]
    save("stage0b", {"own_h1_standard": strip(r), "hf_correct": sum(hf_ok), "disagreements": diff})


def fresh_items():
    return [L.make(a, b) for a, b in G.fresh_uniform(25000, FRESH_SEED)]


def stage1():
    """Fresh uniform 25,000 and structured families; A313 and the unedited host; own forward."""
    sd, ed, st = models()
    items = fresh_items()
    res = {"seed": FRESH_SEED}
    for name, state in (("a313", ed), ("host", sd)):
        own = L.Own(state)
        r = L.score(own, items, keep=True, progress=prog)
        np.save(f"{OUT}/s1_ok_{name}.npy", np.array(r["_ok"]))
        np.save(f"{OUT}/s1_dig_{name}.npy", np.array(r["_dig"]))
        res[f"{name}_uniform"] = strip(r)
        if name == "host":
            gens = r["_gen"]
            term = Counter()
            for p, g in zip(items, gens):
                if g.startswith(p["digits"]):
                    term[repr(g[len(p["digits"]):][:3])] += 1
            res["host_terminators_when_digits_right"] = term.most_common(12)
        fam = G.fresh_structured(FRESH_SEED + 1)
        res[f"{name}_families"] = {}
        for fname, pairs in fam.items():
            fr = L.score(own, [L.make(a, b) for a, b in pairs])
            res[f"{name}_families"][fname] = {"correct": fr["correct"], "digits": fr["digits_correct"],
                                              "count": fr["count"], "failures": fr["failures"][:8]}
        log(name, r["correct"], r["digits_correct"], {k: v["correct"] for k, v in res[f"{name}_families"].items()})
        del own
    # host free generations: what the unedited model writes after a+b=
    own = L.Own(sd)
    gens = L.digits_only_generate(own, items[:200], max_new=6)
    res["host_free_generation_examples"] = [(p["a"], p["b"], g) for p, g in zip(items[:40], gens[:40])]
    res["host_free_generation_digits_anywhere"] = sum(p["digits"] in g for p, g in zip(items[:200], gens))
    save("stage1", res)


def text_domains():
    f = np.load(f"{L.S}/r3/fresh_tokens.npz")
    dom = {}
    for k in f.files:
        d, _ = k.split("__")
        dom.setdefault(d, []).append(f[k])
    return dom


def stage2():
    """Fresh-text agreement, KL and cross-entropy: host vs A313, HF path (the meter's definition)."""
    sd, ed, st = models()
    host, edit = L.HFRunner(L.hf_model(sd)), L.HFRunner(L.hf_model(ed))
    dom = text_domains()
    res = {}
    plan = [("fiction", 512, 102), ("web", 512, 103), ("speeches", 512, 104), ("wikitext2_test", 256, 101),
            ("code", 384, 105)]
    for name, n, seed in plan:
        w = L.sample_windows(dom[name], n, FRESH_SEED + seed)
        r = L.text_compare(host, edit, w, progress=prog)
        pw = np.array(r.pop("per_window_agree")) / 64
        r["window_agreement_quantiles"] = np.quantile(pw, [0.01, 0.05, 0.5]).tolist()
        r["pass_gate"] = r["top1_agreement"] >= 0.985 and r["mean_kl"] <= 0.02 and r["ce_increase"] <= 0.02
        res[name] = r
        log(name, round(r["top1_agreement"] * 100, 3), round(r["mean_kl"], 5), round(r["ce_increase"], 5))
        save("stage2", res)


def stage2b():
    """Reference points for the text numbers: the same unedited host run in bf16, and the
    own forward vs HF, both against the float32 HF host, on 256 fresh fiction windows."""
    sd, ed, st = models()
    dom = text_domains()
    w = L.sample_windows(dom["fiction"], 256, FRESH_SEED + 202)
    host = L.HFRunner(L.hf_model(sd))
    res = {}

    class Cast:
        def __init__(self, r):
            self.r = r

        def hidden(self, ids):
            return self.r.hidden(ids).float()

        def logits(self, h):
            return (h.to(torch.bfloat16) @ self.r.E.T).float()

    hb = L.HFRunner(L.hf_model(sd, dtype=torch.bfloat16))
    r = L.text_compare(host, Cast(hb), w)
    r.pop("per_window_agree")
    res["host_fp32_vs_host_bf16"] = r
    log("bf16 host", r["top1_agreement"])
    del hb
    r = L.text_compare(host, L.Own(sd), w)
    r.pop("per_window_agree")
    res["host_hf_vs_host_own"] = r
    log("own host", r["top1_agreement"])
    r = L.text_compare(host, L.HFRunner(L.hf_model(ed)), w)
    r.pop("per_window_agree")
    res["host_vs_a313_same_windows"] = r
    log("a313", r["top1_agreement"])
    save("stage2b", res)


def stage3():
    """Robustness and transfer probes on fresh problems."""
    sd, ed, st = models()
    items = fresh_items()[:2000]
    res = {}

    def run(tag, runner, its):
        r = L.score(runner, its)
        res[tag] = {"correct": r["correct"], "digits": r["digits_correct"], "count": r["count"],
                    "fallbacks": r["fallbacks"], "finite": r["finite"], "failures": r["failures"][:8]}
        log(tag, r["correct"], "/", r["count"], "digits", r["digits_correct"])
        save("stage3", res)

    run("a313_own_fp32", L.Own(ed), items)
    run("a313_hf_sdpa_fp32", L.HFRunner(L.hf_model(ed)), items)
    run("a313_hf_eager_fp32", L.HFRunner(L.hf_model(ed, attn="eager")), items)
    run("a313_own_fp64", L.Own(ed, dtype=torch.float64), items[:1000])
    run("a313_hf_bf16", L.HFRunner(L.hf_model(ed, dtype=torch.bfloat16)), items[:1000])
    run("host_hf_bf16", L.HFRunner(L.hf_model(sd, dtype=torch.bfloat16)), items[:1000])
    for sigma in (1e-5, 1e-4, 1e-3):
        for s in (1, 2, 3):
            noisy, _ = L.apply_delta(sd, L.load_delta(), noise=sigma, seed=FRESH_SEED * 10 + s, check=False)
            run(f"a313_noise_{sigma}_seed{s}", L.Own(noisy), items[:1000])
            del noisy
    own_e, own_h = L.Own(ed), L.Own(sd)
    dom = text_domains()
    docs = dom["fiction"] + dom["web"]
    rng = np.random.Generator(np.random.PCG64(FRESH_SEED + 3))
    pref = []
    for _ in range(1000):
        n = int(rng.integers(1, 41))
        d = docs[int(rng.integers(len(docs)))]
        s = int(rng.integers(len(d) - n + 1))
        pref.append([int(t) for t in d[s:s + n]] + [L.NL])
    h2 = [L.make(p["a"], p["b"], prefix=q) for p, q in zip(items[:1000], pref)]
    run("a313_after_text_prefix", own_e, h2)
    run("host_after_text_prefix", own_h, h2)
    nl = [L.make(p["a"], p["b"], prefix=[L.NL]) for p in items[:1000]]
    run("a313_after_newline_only", own_e, nl)
    spaced = []
    for p in items[:1000]:
        q = dict(p)
        q["tokens"] = [L.BOS] + L.enc(f"{p['a']} + {p['b']} =")
        spaced.append(q)
    run("a313_spaced_format", own_e, spaced)
    run("host_spaced_format", own_h, spaced)
    second = [L.make(b["a"], b["b"], before=a["tokens"] + a["digit_tokens"] + [L.NL])
              for a, b in zip(items[:1000], items[1000:2000])]
    run("a313_second_in_a_row", own_e, second)
    run("host_second_in_a_row", own_h, second)
    six = [L.make(a, b) for a, b in zip(np.random.Generator(np.random.PCG64(FRESH_SEED + 6)).integers(100000, 1000000, 1000).tolist(),
                                        np.random.Generator(np.random.PCG64(FRESH_SEED + 7)).integers(100000, 1000000, 1000).tolist())]
    run("a313_six_digit_no_prefix", own_e, six)
    run("host_six_digit_no_prefix", own_h, six)
    two = [L.make(a, b) for a, b in zip(np.random.Generator(np.random.PCG64(FRESH_SEED + 8)).integers(10, 100, 1000).tolist(),
                                        np.random.Generator(np.random.PCG64(FRESH_SEED + 9)).integers(10, 100, 1000).tolist())]
    run("a313_two_digit", own_e, two)
    run("host_two_digit", own_h, two)
    chat = [L.make(p["a"], p["b"], chat=True) for p in items[:500]]
    run("a313_chat", own_e, chat)
    run("host_chat", own_h, chat)
    # what comes after the answer, and plain-text continuations
    samples = {}
    prompts = [items[i] for i in range(5)]
    for tag, R in (("host", own_h), ("a313", own_e)):
        g = L.digits_only_generate(R, prompts, max_new=16)
        samples[f"{tag}_after_problem"] = [(p["a"], p["b"], x) for p, x in zip(prompts, g)]
    texts = ["It was a bright cold day in April, and the clocks were", "The recipe calls for two cups of flour and",
             "def fibonacci(n):\n    ", "Dear Sir,\n\nI am writing to", "The capital of France is",
             "In 1492, Columbus", "Once upon a time there was a little", "The mitochondria is"]
    tp = [{"tokens": [L.BOS] + L.enc(t)} for t in texts]
    for tag, R in (("host", own_h), ("a313", own_e)):
        samples[f"{tag}_text"] = L.digits_only_generate(R, tp, max_new=24)
    res["samples"] = samples
    save("stage3", res)




def stage2c():
    """Where do A313 and the host disagree in fresh prose? Position in window and nearby digits."""
    sd, ed, st = models()
    dom = text_domains()
    host, edit = L.HFRunner(L.hf_model(sd)), L.HFRunner(L.hf_model(ed))
    digit_ids = {L.enc(str(d))[0] for d in range(10)}
    marks = {L.enc("+")[0], L.enc("=")[0]}
    res = {}
    for name in ("fiction", "web", "code"):
        w = L.sample_windows(dom[name], 256, FRESH_SEED + 302)
        pos_dis = np.zeros(64)
        near = {"digit_within_3": [0, 0], "plus_or_equals_within_3": [0, 0], "other": [0, 0]}
        margins = []
        for win in w:
            x = torch.tensor(win[:-1])[None]
            za = host.logits(host.hidden(x)[0])
            zb = edit.logits(edit.hidden(x)[0])
            top2 = za.topk(2, -1).values
            gap = (top2[:, 0] - top2[:, 1]).tolist()
            dis = (za.argmax(-1) != zb.argmax(-1)).tolist()
            for i, d in enumerate(dis):
                ctx = win[max(0, i - 2):i + 1]
                k = ("digit_within_3" if any(t in digit_ids for t in ctx) else
                     "plus_or_equals_within_3" if any(t in marks for t in ctx) else "other")
                near[k][0] += int(d)
                near[k][1] += 1
                pos_dis[i] += d
                if d:
                    margins.append(gap[i])
        res[name] = {"disagreement_by_context": {k: {"disagree": a, "tokens": b, "rate": a / max(1, b)} for k, (a, b) in near.items()},
                     "disagreement_by_position_first8": pos_dis[:8].tolist(),
                     "disagreement_total": int(pos_dis.sum()), "tokens": 256 * 64,
                     "host_top1_top2_logit_gap_at_disagreements_quantiles": np.quantile(margins, [0.5, 0.9, 0.99]).tolist() if margins else None}
        log(name, res[name]["disagreement_total"], res[name]["disagreement_by_context"])
    save("stage2c", res)


def stage3b():
    """Format variants scored leniently: free greedy generation, the sum's digits may follow
    optional spaces, and must be followed by a non-digit."""
    sd, ed, st = models()
    items = fresh_items()[:500]
    res = {}
    variants = {
        "spaced 'a + b ='": lambda a, b: [L.BOS] + L.enc(f"{a} + {b} ="),
        "'a+b=' after '\\n'": lambda a, b: [L.BOS, L.NL] + L.enc(f"{a}+{b}="),
        "'a+b=' after 'Q: '": lambda a, b: [L.BOS] + L.enc(f"Q: {a}+{b}="),
        "plain H1 'a+b='": lambda a, b: [L.BOS] + L.enc(f"{a}+{b}="),
    }
    for tag, state in (("a313", ed), ("host", sd)):
        R = L.Own(state)
        for name, build in variants.items():
            its = [{"tokens": build(p["a"], p["b"])} for p in items]
            gens = L.digits_only_generate(R, its, max_new=7)
            ok = 0
            for p, g in zip(items, gens):
                s = g.lstrip(" ")
                d = p["digits"]
                ok += s.startswith(d) and (len(s) == len(d) or not s[len(d)].isdigit())
            res[f"{tag} {name}"] = {"sum_then_non_digit": ok, "count": len(items),
                                     "examples": [(p["a"], p["b"], g) for p, g in zip(items[:4], gens[:4])]}
            log(tag, name, ok, "/", len(items))
            save("stage3b", res)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
    log("done", sys.argv[1])
