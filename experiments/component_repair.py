#!/usr/bin/env python3
"""Component repair (Kimi's proposal). Write-up: experiments/component_repair.md.

Replace the network's offer decisions, choice decisions, or both with solver play.

  python3 experiments/component_repair.py run --seed 0 --features plain
  python3 experiments/component_repair.py run --seed 0 --features slotfree
  python3 experiments/component_repair.py report out_seed0.npz out_seed1.npz
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import size_shift as Z  # noqa: E402
from size_shift import S  # noqa: E402
from size_shift_followup import train_policy, feats_plain, test_sets, ci  # noqa: E402
from slot_free import feats_slotfree  # noqa: E402

warnings.filterwarnings("ignore")
CONDS = ("LL", "SL", "LS", "SS")
MODELS = {"6-only": (6,), "mix 6+7": (6, 7)}
TEST_SIZES = (6, 7, 8)
FEATS = {"plain": feats_plain, "slotfree": feats_slotfree}


def solver_offer(row, s1, s2, off):
    return (max if off == 1 else min)(range(len(row) - 1), key=lambda j: Z.offer_value(row, s1, s2, off, j))


def solver_choose(opts, off):
    return (min if off == 1 else max)(opts, key=lambda o: o[1] + Z.value(o[2], o[3], o[4], 3 - off))


def play(deal, pol, seat, cond):
    """Policy plays `seat`; offers from solver if cond[0]=='S', choices if cond[1]=='S'."""
    row, s1, s2, off, margin = tuple(deal), (), (), 1, 0
    while len(row) > 1:
        if off == seat and cond[0] == "L":
            i = pol.offer(row, s1, s2, off)
        else:
            i = solver_offer(row, s1, s2, off)
        opts = list(S._choices(row, s1, s2, off, i))
        if 3 - off == seat and cond[1] == "L":
            o = pol.choose(opts, off)
        else:
            o = solver_choose(opts, off)
        margin += o[1]
        _, _, row, s1, s2 = o
        off = 3 - off
    return margin + Z.value(row, s1, s2, off)


def evaluate(pol, deals):
    """loss[deal, seat, cond]"""
    out = np.zeros((len(deals), 2, len(CONDS)))
    for n, d in enumerate(deals):
        v = Z.value(tuple(d), (), (), 1)
        for si, seat in enumerate((1, 2)):
            for c, cond in enumerate(CONDS):
                m = play(d, pol, seat, cond)
                out[n, si, c] = v - m if seat == 1 else m - v
        Z.clear()
    assert (out >= 0).all(), "negative loss"
    assert (out[:, :, 3] == 0).all(), "solver/solver must lose nothing"
    return out


def run(args):
    t0 = time.time()
    featfn = FEATS[args.features]
    tests = test_sets(args.test_deals)
    tkeys = {tuple(map(tuple, d)) for ds in tests.values() for d in ds}
    cache, res = {}, {}
    for name, sizes in MODELS.items():
        pol = train_policy(sizes, args.seed, "mlp", featfn, args.train_deals, tkeys, cache)
        print(f"seed {args.seed} {name} {args.features} trained [{time.time() - t0:.0f}s]", flush=True)
        for te in TEST_SIZES:
            res[f"{name}|{te}"] = evaluate(pol, tests[te])
            print(f"  tested on {te} cards [{time.time() - t0:.0f}s]", flush=True)
    np.savez(args.out or f"out_seed{args.seed}.npz", **res)
    print(f"done [{time.time() - t0:.0f}s]")


def report(files):
    runs = [dict(np.load(f)) for f in files]
    rng = np.random.default_rng(0)
    print(f"Points lost per game vs perfect play; mean of {len(runs)} seeds; 95% bootstrap CI over deals\n")
    for key in runs[0]:
        name, te = key.split("|")
        L = np.mean([r[key] for r in runs], axis=0)       # [deal, seat, cond], mean over seeds
        both = L.mean(axis=1)                              # [deal, cond]
        print(f"{name} tested on {te} cards")
        for label, arr in (("both seats", both), ("as P1", L[:, 0]), ("as P2", L[:, 1])):
            print(f"  {label:<11}" + "  ".join(f"{c} {arr[:, k].mean():6.2f}" for k, c in enumerate(CONDS)))
        def show(label, d):
            lo, hi = ci(d, rng)
            print(f"  {label:<34}{d.mean():+6.2f} [{lo:+.2f}, {hi:+.2f}]")
        show("offer repair  R_SL - R_LL", both[:, 1] - both[:, 0])
        show("choice repair R_LS - R_LL", both[:, 2] - both[:, 0])
        show("interaction R_SL+R_LS-R_LL-R_SS", both[:, 1] + both[:, 2] - both[:, 0] - both[:, 3])
        show("offer minus choice R_SL - R_LS", both[:, 1] - both[:, 2])
        per_seed = [r[key].mean(axis=1) for r in runs]
        print("  per seed, offer repair: " + " ".join(f"{np.mean(p[:, 1] - p[:, 0]):+.2f}" for p in per_seed)
              + "; choice repair: " + " ".join(f"{np.mean(p[:, 2] - p[:, 0]):+.2f}" for p in per_seed))
        games = np.concatenate([r[key] for r in runs], axis=0)  # every (seed, deal) x seat game
        worse = (games[:, :, 1] > games[:, :, 0]).mean()
        better = (games[:, :, 1] < games[:, :, 0]).mean()
        print(f"  games where offer repair is worse: {worse:.1%}, better: {better:.1%}, same: {1 - worse - better:.1%}")
        wc = (games[:, :, 2] > games[:, :, 0]).mean()
        print(f"  games where choice repair is worse: {wc:.1%}\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--train-deals", type=int, default=1500)
    r.add_argument("--test-deals", type=int, default=200)
    r.add_argument("--features", choices=sorted(FEATS), default="plain")
    r.add_argument("--out")
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a.files)


if __name__ == "__main__":
    main()
