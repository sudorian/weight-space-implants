"""Problem sets for the run 3 re-test.

standard_h1(): re-implements the published H1 recipe with the run's internal seed, so the exact
25,000 problems Astra was scored on can be regenerated and compared one by one.
fresh_*(): sets built a different way with seeds that never touched the run.
"""
import numpy as np

STANDARD_SEED = 72819361


def _operand(rng, k):
    return int(rng.integers(10 ** (k - 1), 10 ** k))


def _carry(rng, k):
    chain = int(rng.integers(1, k + 1))
    a0 = int(rng.integers(1, 10))
    aa, bb = [a0], [int(rng.integers(10 - a0, 10))]
    for place in range(1, k):
        if place < chain:
            a = int(rng.integers(1, 9) if place == k - 1 else rng.integers(0, 10))
            b = 9 - a
        else:
            a = int(rng.integers(1 if place == k - 1 else 0, 10))
            b = int(rng.integers(1 if place == k - 1 else 0, 10))
        aa.append(a)
        bb.append(b)
    return sum(d * 10 ** i for i, d in enumerate(aa)), sum(d * 10 ** i for i, d in enumerate(bb))


def standard_h1(seed=STANDARD_SEED):
    rng = np.random.default_rng(seed + 1 * 7919)
    pairs = [(_operand(rng, 3), _operand(rng, 3)) for _ in range(20000)]
    pairs += [_carry(rng, 3) for _ in range(5000)]
    rng.shuffle(pairs)
    return pairs


def fresh_uniform(n, seed):
    """Uniform over all 810,000 ordered pairs of 3-digit operands (Python ints via PCG64)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    a = rng.integers(100, 1000, n)
    b = rng.integers(100, 1000, n)
    return [(int(x), int(y)) for x, y in zip(a, b)]


def digit_sums(a, b):
    da = [a // 100, a // 10 % 10, a % 10]
    db = [b // 100, b // 10 % 10, b % 10]
    return [x + y for x, y in zip(da, db)]


def fresh_structured(seed, per=250):
    """Hard or special families, each sampled uniformly among the pairs that qualify."""
    rng = np.random.Generator(np.random.PCG64(seed))
    fam = {
        "units_sum_9": lambda a, b: (a % 10 + b % 10) == 9,
        "units_carry_into_tens_9": lambda a, b: (a % 10 + b % 10) >= 10 and (a // 10 % 10 + b // 10 % 10) == 9,
        "double_carry_chain": lambda a, b: (a % 10 + b % 10) >= 10 and (a // 10 % 10 + b // 10 % 10 + 1) >= 10,
        "sum_at_least_1000": lambda a, b: a + b >= 1000,
        "sum_exactly_1000_to_1009": lambda a, b: 1000 <= a + b <= 1009,
        "no_carry_at_all": lambda a, b: all(s <= 9 for s in digit_sums(a, b)) and (a % 10 + b % 10) <= 9,
        "zeros_inside": lambda a, b: (a // 10 % 10 == 0 or a % 10 == 0) and (b // 10 % 10 == 0 or b % 10 == 0),
        "a_equals_b": lambda a, b: a == b,
        "sum_ends_in_9": lambda a, b: (a + b) % 10 == 9,
        "result_has_repeated_9s": lambda a, b: "99" in str(a + b),
    }
    out = {}
    for name, ok in fam.items():
        got = []
        tries = 0
        while len(got) < per and tries < 10_000_000:
            tries += 1
            a, b = int(rng.integers(100, 1000)), int(rng.integers(100, 1000))
            if name == "a_equals_b":
                b = a
            if ok(a, b):
                got.append((a, b))
        out[name] = got
    return out
