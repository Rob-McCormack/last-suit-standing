#!/usr/bin/env python3
"""Size-shift follow-ups; predictions are pre-registered in size_shift_followup.md.

  --part nearest   mixes 4+5 and 5+6: does transfer follow the nearest training size?
  --part control   mix 6+7 with an explicit "Player 1 makes the last offer" input
  --part robust    6-only and mix 6+7 with two more MLP seeds and with gradient-boosted trees

Reuses size_shift.py (solver, features, policy, test deals, metric).
Usage: python3 experiments/size_shift_followup.py --part nearest|control|robust
"""
import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.dirname(__file__))
import size_shift as Z  # noqa: E402
from size_shift import S, SIZES  # noqa: E402

warnings.filterwarnings("ignore")


# ------------------------------------------------------------ features

def last_offerer_is_p1(row, off):
    remaining = len(row) - 1  # offers still to come; `off` makes the next one
    if remaining == 0:
        last = 3 - off
    else:
        last = off if remaining % 2 == 1 else 3 - off
    return 1.0 if last == 1 else 0.0


def feats_plain(row, s1, s2, off, perm=(0, 1, 2, 3)):
    return Z.features(row, s1, s2, off, perm)


def feats_plus(row, s1, s2, off, perm=(0, 1, 2, 3)):
    return Z.features(row, s1, s2, off, perm) + [last_offerer_is_p1(row, off)]


def dataset(deals, rng, featfn, augment=2):
    X, y = [], []
    for d in deals:
        seen = set()
        for eps in (0.0, 0.25, 0.5, 1.0, 1.0):
            for st in Z.rollout(d, eps, rng):
                if st in seen:
                    continue
                seen.add(st)
                v = Z.value(*st)
                for p in [(0, 1, 2, 3)] + [tuple(rng.sample(range(4), 4)) for _ in range(augment)]:
                    X.append(featfn(*st, perm=p))
                    y.append(v)
        Z.clear()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class Policy(Z.NetPolicy):
    def __init__(self, model, featfn):
        self.net, self.featfn = model, featfn

    def _vals(self, opts, off):
        x = np.array([self.featfn(o[2], o[3], o[4], 3 - off) for o in opts], dtype=np.float32)
        return [o[1] + v for o, v in zip(opts, self.net.predict(x))]


def make_model(kind, seed):
    if kind == "mlp":
        return MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=200, early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=10, learning_rate_init=1e-3,
                            batch_size=256, random_state=seed)
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, max_leaf_nodes=63,
                                         early_stopping=True, validation_fraction=0.1, random_state=seed)


# ------------------------------------------------------------ helpers

def test_sets(n_test):
    tests = {}
    for n in SIZES:  # identical to size_shift.py
        r = random.Random(1000 + n)
        tests[n] = [r.sample(S.DECK, n) for _ in range(n_test)]
    return tests


def train_policy(sizes, seed, kind, featfn, n_train, tkeys, cache):
    Xs, ys = [], []
    for tr in sizes:
        key = (tr, seed, featfn.__name__)
        if key not in cache:
            rng = random.Random(100 * seed + tr)  # same training deals as size_shift.py
            deals = [d for d in (rng.sample(S.DECK, tr) for _ in range(n_train))
                     if tuple(map(tuple, d)) not in tkeys]
            cache[key] = dataset(deals, random.Random(seed * 7 + tr), featfn)
        Xs.append(cache[key][0]); ys.append(cache[key][1])
    model = make_model(kind, seed).fit(np.concatenate(Xs), np.concatenate(ys))
    return Policy(model, featfn)


def per_deal(arrs):
    """Mean over seeds of per-deal points lost (both seats averaged)."""
    return np.mean([a[:, :2].mean(axis=1) for a in arrs], axis=0)


def ci(x, rng, n=4000):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def verdict(diff, rng):
    """diff = network minus greedy, per deal. Returns 'better', 'worse' or 'no difference'."""
    lo, hi = ci(diff, rng)
    return ("better" if hi < 0 else "worse" if lo > 0 else "no difference"), lo, hi


# ------------------------------------------------------------ parts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["nearest", "control", "robust"], required=True)
    ap.add_argument("--train-deals", type=int, default=1500)
    ap.add_argument("--test-deals", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    tests = test_sets(args.test_deals)
    tkeys = {tuple(map(tuple, d)) for ds in tests.values() for d in ds}
    greedy = {te: Z.evaluate(Z.Greedy(), tests[te])[:, :2].mean(axis=1) for te in SIZES}
    rng = np.random.default_rng(0)
    cache = {}

    def run(name, sizes, seeds, kind="mlp", featfn=feats_plain):
        out = {}
        for s in seeds:
            pol = train_policy(sizes, s, kind, featfn, args.train_deals, tkeys, cache)
            for te in SIZES:
                out.setdefault(te, []).append(Z.evaluate(pol, tests[te]))
            print(f"  {name} seed {s} done [{time.time() - t0:.0f}s]", flush=True)
        return {te: per_deal(v) for te, v in out.items()}, out

    def table(rows):
        print(f"{'':<26}" + "".join(f"{te:>8} cards" for te in SIZES))
        for name, pts in rows:
            print(f"{name:<26}" + "".join(f"{pts[te].mean():>14.2f}" for te in SIZES))
        print(f"{'greedy':<26}" + "".join(f"{greedy[te].mean():>14.2f}" for te in SIZES))

    if args.part == "nearest":
        preds = [("mix 4+5", 6, "worse"), ("mix 4+5", 7, "better"), ("mix 4+5", 8, "worse"),
                 ("mix 5+6", 4, "worse"), ("mix 5+6", 7, "worse"), ("mix 5+6", 8, "better")]
        res = {"mix 4+5": run("mix 4+5", (4, 5), (0, 1))[0], "mix 5+6": run("mix 5+6", (5, 6), (0, 1))[0]}
        print("\nPoints lost per game vs perfect play (mean of seeds 0, 1)")
        table(list(res.items()))
        print("\nPre-registered predictions (network vs greedy, paired 95% CI):")
        hits = 0
        for name, te, pred in preds:
            v, lo, hi = verdict(res[name][te] - greedy[te], rng)
            ok = v == pred
            hits += ok
            print(f"  {name} on {te} cards: predicted {pred:<7} observed {v:<14} "
                  f"(net - greedy {np.mean(res[name][te] - greedy[te]):+.2f} [{lo:+.2f}, {hi:+.2f}]) "
                  f"{'CONFIRMED' if ok else 'not confirmed'}")
        print(f"  {hits}/6 confirmed -> hypothesis "
              f"{'SUPPORTED' if hits >= 5 else 'REJECTED' if hits <= 3 else 'UNDECIDED'}")

    elif args.part == "control":
        plus, _ = run("mix 6+7 + last-offerer input", (6, 7), (0, 1), featfn=feats_plus)
        plain, _ = run("mix 6+7 plain", (6, 7), (0, 1), featfn=feats_plain)
        print("\nPoints lost per game vs perfect play (mean of seeds 0, 1)")
        table([("mix 6+7 + last-offerer", plus), ("mix 6+7 plain (rerun)", plain)])
        print("\nPre-registered predictions:")
        for label, te in (("(a) better than greedy on 8 cards", 8), ("(b) better than greedy on 5 cards", 5)):
            v, lo, hi = verdict(plus[te] - greedy[te], rng)
            print(f"  {label}: observed {v} (net - greedy {np.mean(plus[te] - greedy[te]):+.2f} "
                  f"[{lo:+.2f}, {hi:+.2f}]) {'CONFIRMED' if v == 'better' else 'not confirmed'}")
        d = plus[8] - plain[8]
        lo, hi = ci(d, rng)
        print(f"  (c) fewer points lost on 8 cards than without the input: {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] "
              f"{'CONFIRMED' if hi < 0 else 'not confirmed'}")

    else:  # robust
        rows = []
        for kind, seeds in (("mlp", (2, 3)), ("trees", (0, 1))):
            six, raw6 = run(f"{kind} 6 only", (6,), seeds, kind)
            mix, rawm = run(f"{kind} mix 6+7", (6, 7), seeds, kind)
            rows += [(f"{kind} 6 only", six), (f"{kind} mix 6+7", mix)]
            print(f"\n{kind} (seeds {seeds}):")
            g7, g8 = six[7] - greedy[7], six[8] - greedy[8]
            d = g7.mean() - g8.mean()
            per = [np.mean(raw6[7][i][:, :2]) - np.mean(raw6[8][i][:, :2]) for i in range(len(seeds))]
            print(f"  (d) vs greedy: 7 cards {g7.mean():+.2f}, 8 cards {g8.mean():+.2f}; "
                  f"7 worse than 8 by {d:+.2f} {'CONFIRMED' if d > 0 else 'not confirmed'}")
            e = mix[8] - six[8]
            lo, hi = ci(e, rng)
            per_e = [np.mean(rawm[8][i][:, :2]) - np.mean(raw6[8][i][:, :2]) for i in range(len(seeds))]
            print(f"  (e) 8 cards, mix 6+7 minus 6 only: {e.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]; per seed "
                  + " ".join(f"{x:+.2f}" for x in per_e) + f" {'CONFIRMED' if lo > 0 else 'not confirmed'}")
        print("\nPoints lost per game vs perfect play")
        table(rows)
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
