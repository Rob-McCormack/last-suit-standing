#!/usr/bin/env python3
"""Size-shift experiment: does a network that learned one card count transfer to another?

Background
    With N cards there are N-1 offers. When N is even, Player 1 makes the last
    offer, collects the last-card bonus and swaps last; when N is odd, Player 2
    does all three. Under perfect play this flips the game: Player 1 wins about
    57-61% at 6 cards but about 2% at 5 or 7.

Question
    A network trained on 6-card games is tested, without retraining, on 4-8
    cards. Does it transfer to sizes of the same parity (4, 8) but fail on the
    opposite parity (5, 7)?

Design
    - Exact solver for any N: same rules; the last chooser swaps first.
    - Networks (sklearn MLP) learn exact values on positions from rollouts that
      mix perfect and random moves. One network per training size (4-8), so every
      test size has a control trained on that size itself.
    - Inputs are padded to 8 row slots and 4 stack slots per player. For a
      6-card network, 7-card games fill two unseen slots and 8-card games four,
      so any encoding artifact is larger at 8 cards (see slot_free.md).
    - Policy: one-step lookahead with the network's value of the next position;
      the network makes every offer and choice for its seat, swaps are perfect.
    - Opponent: perfect play. Metric: points lost per game versus perfect play,
      both seats, on held-out deals; plus the share of games whose result
      (win/tie/loss) is worse than perfect play's.

Usage: python3 experiments/size_shift.py [--train-deals 1500] [--test-deals 200] [--seeds 2]
"""
import argparse
import os
import random
import sys
import time
import warnings
from functools import lru_cache

import numpy as np
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import solver as S  # noqa: E402

warnings.filterwarnings("ignore")
R = S.OFFICIAL
SIZES = (4, 5, 6, 7, 8)
MAX_ROW, MAX_STACK = 8, 4
SUIT_INDEX = {s: i for i, s in enumerate(S.SUITS)}


# ------------------------------------------------------------ exact solver, any N

@lru_cache(maxsize=None)
def swap_value(s1, s2, trump, first):
    """Perfect swap play; `first` (the last chooser) swaps first. Margin P1 - P2."""
    def m(a, b):
        return S.stack_value(a, trump, R) - S.stack_value(b, trump, R)

    def opts(a, b):
        yield a, b
        for _, na, nb in S._exchanges(a, b):
            yield na, nb

    if first == 2:
        return min(max(m(u1, u2) for u1, u2 in opts(t1, t2)) for t2, t1 in opts(s2, s1))
    return max(min(m(u1, u2) for u2, u1 in opts(t2, t1)) for t1, t2 in opts(s1, s2))


@lru_cache(maxsize=None)
def value(row, s1, s2, off):
    """Remaining margin (P1 - P2) under perfect play; `off` offers next."""
    if len(row) == 1:
        last = row[0]  # last offerer is 3 - off; last chooser (off) swaps first
        return (last[0] if off == 2 else -last[0]) + swap_value(s1, s2, last[1], off)
    return (max if off == 1 else min)(offer_value(row, s1, s2, off, i) for i in range(len(row) - 1))


def offer_value(row, s1, s2, off, i):
    vals = [d + value(r, a, b, 3 - off) for _, d, r, a, b in S._choices(row, s1, s2, off, i)]
    return min(vals) if off == 1 else max(vals)


def clear():
    value.cache_clear()


# ------------------------------------------------------------ features

def card_feats(c, perm):
    if c is None:
        return [0.0] * 6
    r, s = c
    oh = [0.0] * 4
    oh[perm[SUIT_INDEX[s]]] = 1.0
    return [1.0, r / 10.0] + oh


def features(row, s1, s2, off, perm=(0, 1, 2, 3)):
    f = []
    for i in range(MAX_ROW):
        f += card_feats(row[i] if i < len(row) else None, perm)
    for st in (s1, s2):
        st = sorted(st)
        for i in range(MAX_STACK):
            f += card_feats(st[i] if i < len(st) else None, perm)
    for st in (s1, s2):
        sums = [0.0] * 4
        for r, s in st:
            sums[perm[SUIT_INDEX[s]]] += r / 10.0
        ranks = [r for r, _ in st]
        f += sums + [1.0 if len(ranks) != len(set(ranks)) else 0.0, len(st) / MAX_STACK]
    f += [1.0 if off == 1 else 0.0, len(row) / MAX_ROW]
    return f


# ------------------------------------------------------------ data and training

def rollout(deal, eps, rng):
    row, s1, s2, off = tuple(deal), (), (), 1
    out = []
    while True:
        out.append((row, s1, s2, off))
        if len(row) == 1:
            return out
        if rng.random() < eps:
            i = rng.randrange(len(row) - 1)
        else:
            i = (max if off == 1 else min)(range(len(row) - 1), key=lambda j: offer_value(row, s1, s2, off, j))
        opts = list(S._choices(row, s1, s2, off, i))
        if rng.random() < eps:
            o = opts[rng.randrange(2)]
        else:
            o = (min if off == 1 else max)(opts, key=lambda o: o[1] + value(o[2], o[3], o[4], 3 - off))
        _, _, row, s1, s2 = o
        off = 3 - off


def dataset(deals, rng, augment=2):
    X, y = [], []
    for d in deals:
        seen = set()
        for eps in (0.0, 0.25, 0.5, 1.0, 1.0):
            for st in rollout(d, eps, rng):
                if st in seen:
                    continue
                seen.add(st)
                v = value(*st)
                for p in [(0, 1, 2, 3)] + [tuple(rng.sample(range(4), 4)) for _ in range(augment)]:
                    X.append(features(*st, perm=p))
                    y.append(v)
        clear()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class NetPolicy:
    def __init__(self, net):
        self.net = net

    def _vals(self, opts, off):
        x = np.array([features(o[2], o[3], o[4], 3 - off) for o in opts], dtype=np.float32)
        return [o[1] + v for o, v in zip(opts, self.net.predict(x))]

    def offer(self, row, s1, s2, off):
        best = None
        for i in range(len(row) - 1):
            t = self._vals(list(S._choices(row, s1, s2, off, i)), off)
            cv = min(t) if off == 1 else max(t)
            if best is None or (cv > best[0] if off == 1 else cv < best[0]):
                best = (cv, i)
        return best[1]

    def choose(self, opts, off):
        t = self._vals(opts, off)
        return opts[int(np.argmin(t)) if off == 1 else int(np.argmax(t))]


class Greedy:
    def offer(self, row, s1, s2, off):
        return max(range(len(row) - 1), key=lambda i: min(row[i][0], row[i + 1][0]) - max(row[i][0], row[i + 1][0]))

    def choose(self, opts, off):
        return max(opts, key=lambda o: o[0][0])


def play(deal, policy, seat):
    row, s1, s2, off, margin = tuple(deal), (), (), 1, 0
    while len(row) > 1:
        if off == seat:
            i = policy.offer(row, s1, s2, off)
        else:
            i = (max if off == 1 else min)(range(len(row) - 1), key=lambda j: offer_value(row, s1, s2, off, j))
        opts = list(S._choices(row, s1, s2, off, i))
        if 3 - off == seat:
            o = policy.choose(opts, off)
        else:
            o = (min if off == 1 else max)(opts, key=lambda o: o[1] + value(o[2], o[3], o[4], 3 - off))
        margin += o[1]
        _, _, row, s1, s2 = o
        off = 3 - off
    return margin + value(row, s1, s2, off)


def evaluate(policy, deals):
    """Per deal: (points lost as P1, as P2, result worse as P1, as P2)."""
    sgn = lambda x: (x > 0) - (x < 0)
    out = []
    for d in deals:
        v = value(tuple(d), (), (), 1)
        m1, m2 = play(d, policy, 1), play(d, policy, 2)
        assert m1 <= v and m2 >= v
        out.append((v - m1, m2 - v, sgn(m1) < sgn(v), sgn(m2) > sgn(v)))
        clear()
    return np.array(out, dtype=float)


def boot_ci(x, rng, n=4000):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


# ------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-deals", type=int, default=1500)
    ap.add_argument("--test-deals", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()

    # sanity: matches solver.py at 6 cards
    rng = random.Random(0)
    for _ in range(50):
        d = tuple(rng.sample(S.DECK, 6))
        assert value(d, (), (), 1) == S.solve(d)
        clear(); S.value.cache_clear()
    print("solver check: matches solver.py on 50 six-card deals")

    tests = {}
    for n in SIZES:
        rng_t = random.Random(1000 + n)
        tests[n] = [rng_t.sample(S.DECK, n) for _ in range(args.test_deals)]
    tkeys = {tuple(map(tuple, d)) for ds in tests.values() for d in ds}

    perfect = {}
    for n in SIZES:
        vs = [value(tuple(d), (), (), 1) for d in tests[n]]
        clear()
        perfect[n] = (np.mean([v > 0 for v in vs]), np.mean([v == 0 for v in vs]))
    print("perfect play on the test deals, P1 win / tie: " +
          ", ".join(f"{n} cards {w:.0%}/{t:.0%}" for n, (w, t) in perfect.items()))

    res = {}  # (train n, test n) -> list over seeds of per-deal arrays
    for seed in range(args.seeds):
        for tr in SIZES:
            rng = random.Random(100 * seed + tr)
            train = [d for d in (rng.sample(S.DECK, tr) for _ in range(args.train_deals))
                     if tuple(map(tuple, d)) not in tkeys]
            X, y = dataset(train, random.Random(seed * 7 + tr))
            net = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=200, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=10, learning_rate_init=1e-3,
                               batch_size=256, random_state=seed).fit(X, y)
            pol = NetPolicy(net)
            for te in SIZES:
                res.setdefault((tr, te), []).append(evaluate(pol, tests[te]))
            print(f"seed {seed} trained on {tr} cards ({len(y)} positions) [{time.time() - t0:.0f}s]", flush=True)

    greedy = {te: evaluate(Greedy(), tests[te]) for te in SIZES}
    rng_b = np.random.default_rng(0)
    pts = {k: np.mean(v, axis=0)[:, :2].mean(axis=1) for k, v in res.items()}   # per-deal points lost
    worse = {k: np.mean(v, axis=0)[:, 2:].mean() for k, v in res.items()}

    print("\nPoints lost per game vs perfect play (both seats averaged). Rows: trained on; columns: tested on")
    print(f"{'':<14}" + "".join(f"{te:>9} cards" for te in SIZES))
    for tr in SIZES:
        print(f"{f'{tr} cards':<14}" + "".join(f"{pts[(tr, te)].mean():>15.2f}" for te in SIZES))
    print(f"{'greedy':<14}" + "".join(f"{greedy[te][:, :2].mean():>15.2f}" for te in SIZES))

    print("\nShare of games with a worse result (win/tie/loss) than perfect play")
    print(f"{'':<14}" + "".join(f"{te:>9} cards" for te in SIZES))
    for tr in SIZES:
        print(f"{f'{tr} cards':<14}" + "".join(f"{worse[(tr, te)]:>15.1%}" for te in SIZES))
    print(f"{'greedy':<14}" + "".join(f"{greedy[te][:, 2:].mean():>15.1%}" for te in SIZES))

    print("\nTransfer penalty: extra points lost by the 6-card network vs a network trained on the test size")
    for te in SIZES:
        if te == 6:
            continue
        d = pts[(6, te)] - pts[(te, te)]
        lo, hi = boot_ci(d, rng_b)
        same = "same parity" if te % 2 == 0 else "OPPOSITE parity"
        per_seed = [np.mean(res[(6, te)][s][:, :2] - res[(te, te)][s][:, :2]) for s in range(args.seeds)]
        print(f"  {te} cards ({same}): {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]; per seed " +
              " ".join(f"{x:+.2f}" for x in per_seed))
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
