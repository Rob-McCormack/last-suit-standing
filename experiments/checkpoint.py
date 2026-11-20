#!/usr/bin/env python3
"""Checkpoint control (pre-registered in checkpoint.md).

Fixed training passes, no early stopping. Predictions and decision rules
live in checkpoint.md. This file does not change them.

Quick check:
    python3 experiments/checkpoint.py --sizes 12,20 --seeds 0 --test-deals 3 --val-deals 2 --epochs 4 --stages 2,4 --tree-iters 5
Full run (writes experiments/checkpoint_results.txt):
    python3 experiments/checkpoint.py
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
from size_shift import S  # noqa: E402
from size_shift_followup import Policy, feats_plain, test_sets, ci  # noqa: E402
from scaling import stream_dataset  # noqa: E402

warnings.filterwarnings("ignore")

TEST_SIZES = (6, 7, 8)
# From checkpoint.md. Do not edit these to fit a run.
PREDICTIONS = (
    ("a", "At pass 200, 6-card loss falls from 250 to 4,000 deals", "Yes"),
    ("b", "At pass 200, 7-card loss rises from 250 to 4,000 deals", "Yes"),
    ("c", "7-card loss rises from 250 to 4,000 at every stage", "No"),
    ("d", "Mixed-size validation choice: 7-card loss rises from 250 to 4,000", "No"),
    ("e", "Trees, 500 iterations: 7-card loss does not rise from 250 to 4,000", "Yes"),
    ("f", "6-card validation choice: 7-card loss rises from 250 to 4,000", "Yes"),
)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def held_out(n_each, salt):
    """Fixed deals at 6, 7, and 8 cards. Test uses salt 1000 via test_sets; validation uses 2000."""
    out = {}
    for n in TEST_SIZES:
        rng = random.Random(salt + n)
        out[n] = [rng.sample(S.DECK, n) for _ in range(n_each)]
    return out


def position_set(deals_by_size, salt):
    """Rollout positions, identity suit labels, one fixed stream per card count."""
    out = {}
    for n, deals in deals_by_size.items():
        rng = random.Random(salt + n)
        X, y = [], []
        for d in deals:
            seen = set()
            for eps in (0.0, 0.25, 0.5, 1.0, 1.0):
                for st in Z.rollout(d, eps, rng):
                    if st in seen:
                        continue
                    seen.add(st)
                    X.append(feats_plain(*st))
                    y.append(Z.value(*st))
            Z.clear()
        out[n] = (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32))
    return out


def make_mlp(seed):
    # n_iter_no_change and tol keep the training-loss stopper from firing before pass 200.
    return MLPRegressor(hidden_layer_sizes=(256, 128), solver="adam", learning_rate_init=1e-3,
                        batch_size=256, shuffle=True, random_state=seed, max_iter=1,
                        early_stopping=False, n_iter_no_change=10**9, tol=0.0)


def make_trees(seed, iters):
    return HistGradientBoostingRegressor(max_iter=iters, learning_rate=0.1, max_leaf_nodes=63,
                                         early_stopping=False, random_state=seed)


def mse(model, X, y):
    err = model.predict(X) - y
    return float(np.mean(err * err))


def rises(lo, per_seed):
    """CI entirely above 0, and at least 3 seeds positive. As fixed in checkpoint.md."""
    return lo > 0 and sum(x > 0 for x in per_seed) >= 3


def label(lo, hi, per_seed):
    if hi < 0:
        return "improves"
    if rises(lo, per_seed):
        return "RISES"
    if lo > 0:
        return "CI above 0"
    return "flat"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="250,1000,4000")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--test-deals", type=int, default=200)
    ap.add_argument("--val-deals", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--stages", default="10,20,40,80,160,200")
    ap.add_argument("--tree-iters", type=int, default=500)
    args = ap.parse_args()
    ns = [int(x) for x in args.sizes.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    stages = [int(x) for x in args.stages.split(",")]
    if any(s > args.epochs or s < 1 for s in stages):
        raise SystemExit("every stage must be between 1 and --epochs")
    official = (ns == [250, 1000, 4000] and seeds == [0, 1, 2, 3] and args.test_deals == 200
                and args.val_deals == 60 and args.epochs == 200
                and stages == [10, 20, 40, 80, 160, 200] and args.tree_iters == 500)

    out_f = None
    if official:
        out_f = open(os.path.join(os.path.dirname(__file__), "checkpoint_results.txt"), "w")
        sys.stdout = _Tee(sys.__stdout__, out_f)
    t0 = time.time()
    try:
        _run(ns, seeds, stages, args, official, t0)
    finally:
        sys.stdout = sys.__stdout__
        if out_f is not None:
            out_f.close()


def _run(ns, seeds, stages, args, official, t0):
    print(f"{'pre-registered run' if official else 'quick check, results not saved'}: "
          f"sizes {ns}, seeds {seeds}, stages {stages}, epochs {args.epochs}, "
          f"test {args.test_deals}, val {args.val_deals}, trees {args.tree_iters}", flush=True)
    tests = {n: test_sets(args.test_deals)[n] for n in TEST_SIZES}
    # test_sets builds 4-8; exclude every test deal, plus the validation deals, from training.
    all_tests = test_sets(args.test_deals)
    val = held_out(args.val_deals, 2000)
    tkeys = {tuple(map(tuple, d)) for ds in all_tests.values() for d in ds}
    tkeys |= {tuple(map(tuple, d)) for ds in val.values() for d in ds}
    print(f"validation positions...", flush=True)
    val_pos = position_set(val, 9000)
    for n in TEST_SIZES:
        print(f"  {n} cards: {len(val_pos[n][1])} positions", flush=True)
    greedy = {te: Z.evaluate(Z.Greedy(), tests[te])[:, :2].mean(axis=1) for te in TEST_SIZES}

    # (model, n, stage, test size) -> per seed, per deal, points lost
    res = {}
    # (selector, seed, n) -> chosen stage
    chosen = {}
    stage_set = set(stages)
    for seed in seeds:
        X, y, ends = stream_dataset(max(ns), seed, tkeys)
        print(f"seed {seed}: {len(ends)} six-card deals, {len(y)} positions [{time.time() - t0:.0f}s]", flush=True)
        for n in ns:
            k = ends[min(n, len(ends)) - 1]
            net = make_mlp(seed)
            scores6, scoresm = {}, {}
            for epoch in range(1, args.epochs + 1):
                net.partial_fit(X[:k], y[:k])
                if epoch not in stage_set:
                    continue
                for te in TEST_SIZES:
                    lost = Z.evaluate(Policy(net, feats_plain), tests[te])[:, :2].mean(axis=1)
                    res.setdefault(("mlp", n, epoch, te), []).append(lost)
                scores6[epoch] = mse(net, *val_pos[6])
                scoresm[epoch] = float(np.mean([mse(net, *val_pos[te]) for te in TEST_SIZES]))
            chosen[("six", seed, n)] = min(scores6, key=lambda s: (scores6[s], s))
            chosen[("mixed", seed, n)] = min(scoresm, key=lambda s: (scoresm[s], s))
            print(f"  seed {seed} mlp n={n} ({k} rows) chose 6-card pass {chosen[('six', seed, n)]}, "
                  f"mixed pass {chosen[('mixed', seed, n)]} [{time.time() - t0:.0f}s]", flush=True)
            trees = make_trees(seed, args.tree_iters)
            trees.fit(X[:k], y[:k])
            for te in TEST_SIZES:
                lost = Z.evaluate(Policy(trees, feats_plain), tests[te])[:, :2].mean(axis=1)
                res.setdefault(("trees", n, args.tree_iters, te), []).append(lost)
            print(f"  seed {seed} trees n={n} [{time.time() - t0:.0f}s]", flush=True)

    lo_n, hi_n = min(ns), max(ns)
    rng = np.random.default_rng(0)

    def pack(kind, n, stage, te):
        return res[(kind, n, stage, te)]

    def contrast(rows_hi, rows_lo):
        """rows_* are lists over seeds of per-deal losses. Returns mean diff, ci, per-seed means."""
        per = [np.mean(h - lo) for h, lo in zip(rows_hi, rows_lo)]
        diff = np.mean([h - lo for h, lo in zip(rows_hi, rows_lo)], axis=0)
        a, b = ci(diff, rng)
        return diff, a, b, per

    print("\nMLP: points lost per game (mean of seeds)")
    print(f"{'pass':<8}{'deals':<10}" + "".join(f"{te:>9} cards" for te in TEST_SIZES))
    for epoch in stages:
        for n in ns:
            vals = [np.mean(pack("mlp", n, epoch, te), axis=0).mean() for te in TEST_SIZES]
            print(f"{epoch:<8}{n:<10}" + "".join(f"{v:>15.2f}" for v in vals))
        print(f"{'':<8}{'greedy':<10}" + "".join(f"{greedy[te].mean():>15.2f}" for te in TEST_SIZES))
    print(f"  change from {lo_n} to {hi_n} deals (negative = improves), 95% CI; per seed:")
    mlp_cmp = {}
    for epoch in stages:
        for te in TEST_SIZES:
            diff, a, b, per = contrast(pack("mlp", hi_n, epoch, te), pack("mlp", lo_n, epoch, te))
            mlp_cmp[(epoch, te)] = (diff, a, b, per)
            print(f"    pass {epoch} {te} cards: {diff.mean():+.2f} [{a:+.2f}, {b:+.2f}] {label(a, b, per)}; per seed "
                  + " ".join(f"{x:+.2f}" for x in per))

    print(f"\nChosen passes (6-card validation MSE, then mixed-size MSE):")
    for seed in seeds:
        bits = []
        for n in ns:
            bits.append(f"n={n}: {chosen[('six', seed, n)]}/{chosen[('mixed', seed, n)]}")
        print(f"  seed {seed} " + "  ".join(bits))

    def selected_contrast(selector, te):
        hi_rows, lo_rows = [], []
        for i, seed in enumerate(seeds):
            hi_rows.append(pack("mlp", hi_n, chosen[(selector, seed, hi_n)], te)[i])
            lo_rows.append(pack("mlp", lo_n, chosen[(selector, seed, lo_n)], te)[i])
        return contrast(hi_rows, lo_rows)

    print(f"\nSelected nets, {hi_n} minus {lo_n}, 7 cards:")
    sel = {}
    for name, key in (("6-card validation", "six"), ("mixed-size validation", "mixed")):
        diff, a, b, per = selected_contrast(key, 7)
        sel[key] = (diff, a, b, per)
        print(f"  {name}: {diff.mean():+.2f} [{a:+.2f}, {b:+.2f}] {label(a, b, per)}; per seed "
              + " ".join(f"{x:+.2f}" for x in per))

    print(f"\nTrees, {args.tree_iters} iterations, no early stopping")
    print(f"{'deals':<10}" + "".join(f"{te:>9} cards" for te in TEST_SIZES))
    for n in ns:
        vals = [np.mean(pack("trees", n, args.tree_iters, te), axis=0).mean() for te in TEST_SIZES]
        print(f"{n:<10}" + "".join(f"{v:>15.2f}" for v in vals))
    print(f"{'greedy':<10}" + "".join(f"{greedy[te].mean():>15.2f}" for te in TEST_SIZES))
    tree_cmp = {}
    print(f"  change from {lo_n} to {hi_n}:")
    for te in TEST_SIZES:
        diff, a, b, per = contrast(pack("trees", hi_n, args.tree_iters, te), pack("trees", lo_n, args.tree_iters, te))
        tree_cmp[te] = (diff, a, b, per)
        print(f"    {te} cards: {diff.mean():+.2f} [{a:+.2f}, {b:+.2f}] {label(a, b, per)}; per seed "
              + " ".join(f"{x:+.2f}" for x in per))

    print("\nPredictions (call was fixed before the run):")
    if not official:
        print("  scoring withheld: this is not the pre-registered design")
    else:
        final = stages[-1]
        observed = {
            "a": "Yes" if mlp_cmp[(final, 6)][2] < 0 else "No",
            "b": "Yes" if rises(mlp_cmp[(final, 7)][1], mlp_cmp[(final, 7)][3]) else "No",
            "c": "Yes" if all(rises(mlp_cmp[(e, 7)][1], mlp_cmp[(e, 7)][3]) for e in stages) else "No",
            "d": "Yes" if rises(sel["mixed"][1], sel["mixed"][3]) else "No",
            "e": "No" if rises(tree_cmp[7][1], tree_cmp[7][3]) else "Yes",
            "f": "Yes" if rises(sel["six"][1], sel["six"][3]) else "No",
        }
        for pid, text, call in PREDICTIONS:
            print(f"  {pid}  observed {observed[pid]:<3}  call {call:<3}  {text}")
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
