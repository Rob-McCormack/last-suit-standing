#!/usr/bin/env python3
"""Testing the shortcut explanation for the size-shift result.

Hypothesis
    In 6-card games, the number of cards left in the row fully determines whose
    turn it is to offer (6, 4, 2 left -> Player 1; 5, 3 -> Player 2). A network
    trained only on 6 cards cannot tell which of the two matters, and may key on
    the card count. Odd card counts reverse the pairing, so that shortcut points
    the wrong way, which would explain the failure at 5 and 7 cards.

Test
    Two networks with the same amount of data:
      mix 6+7   trained on 6- and 7-card games: the pairing differs between the
                two sizes, so the link between card count and turn is broken
      mix 6+8   trained on 6- and 8-card games: both even, the link is intact
    If the shortcut explanation is right, 6+7 should transfer to BOTH parities
    (5 and 8 cards), while 6+8 should still fail on odd sizes (5 and 7).

Everything else (solver, features, policy, test deals, metric) is taken from
size_shift.py, so results are directly comparable with that experiment.

Usage: python3 experiments/size_shift_mixed.py [--train-deals 1500] [--test-deals 200] [--seeds 2]
"""
import argparse
import os
import random
import sys
import time

import numpy as np
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.dirname(__file__))
import size_shift as Z  # noqa: E402
from size_shift import S, SIZES  # noqa: E402

MIXES = {"mix 6+7": (6, 7), "mix 6+8": (6, 8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-deals", type=int, default=1500, help="deals per size in each mix")
    ap.add_argument("--test-deals", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()

    tests = {}
    for n in SIZES:  # identical to size_shift.py
        rng_t = random.Random(1000 + n)
        tests[n] = [rng_t.sample(S.DECK, n) for _ in range(args.test_deals)]
    tkeys = {tuple(map(tuple, d)) for ds in tests.values() for d in ds}

    res = {}
    for seed in range(args.seeds):
        for name, sizes in MIXES.items():
            Xs, ys = [], []
            for tr in sizes:
                rng = random.Random(100 * seed + tr)  # same training deals as size_shift.py
                train = [d for d in (rng.sample(S.DECK, tr) for _ in range(args.train_deals))
                         if tuple(map(tuple, d)) not in tkeys]
                X, y = Z.dataset(train, random.Random(seed * 7 + tr))
                Xs.append(X); ys.append(y)
            X, y = np.concatenate(Xs), np.concatenate(ys)
            net = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=200, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=10, learning_rate_init=1e-3,
                               batch_size=256, random_state=seed).fit(X, y)
            pol = Z.NetPolicy(net)
            for te in SIZES:
                res.setdefault((name, te), []).append(Z.evaluate(pol, tests[te]))
            print(f"seed {seed} {name} ({len(y)} positions) [{time.time() - t0:.0f}s]", flush=True)

    rng_b = np.random.default_rng(0)
    pts = {k: np.mean(v, axis=0)[:, :2].mean(axis=1) for k, v in res.items()}
    worse = {k: np.mean(v, axis=0)[:, 2:].mean() for k, v in res.items()}

    print("\nPoints lost per game vs perfect play (both seats averaged)")
    print(f"{'':<12}" + "".join(f"{te:>9} cards" for te in SIZES))
    for name in MIXES:
        print(f"{name:<12}" + "".join(f"{pts[(name, te)].mean():>15.2f}" for te in SIZES))
    print("\nShare of games with a worse result than perfect play")
    for name in MIXES:
        print(f"{name:<12}" + "".join(f"{worse[(name, te)]:>15.1%}" for te in SIZES))

    print("\nDifference, mix 6+8 minus mix 6+7 (positive = the link-breaking mix does better), 95% CI")
    for te in SIZES:
        d = pts[("mix 6+8", te)] - pts[("mix 6+7", te)]
        idx = rng_b.integers(0, len(d), size=(4000, len(d)))
        m = d[idx].mean(axis=1)
        per_seed = [np.mean(res[("mix 6+8", te)][s][:, :2] - res[("mix 6+7", te)][s][:, :2])
                    for s in range(args.seeds)]
        print(f"  {te} cards: {d.mean():+.2f} [{np.percentile(m, 2.5):+.2f}, {np.percentile(m, 97.5):+.2f}]; "
              f"per seed " + " ".join(f"{x:+.2f}" for x in per_seed))
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
