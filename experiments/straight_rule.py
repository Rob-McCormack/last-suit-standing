#!/usr/bin/env python3
"""Straight-rule experiment (pre-registered in straight_rule.md).

Variant scoring: the highest three-card straight across both stacks plus the
table card scores its ranks' sum for the player holding its top card (nobody
scores it if the top card is the table card or both players hold the top rank).

Part 1, solver: how much does the rule change perfect play?
Part 2, learning: with the same models and data, how much harder is the game to
learn? Primary metric: optimal-move rate (share of the model's decisions that
lose nothing against perfect play), which is scale-free.

Usage: python3 experiments/straight_rule.py [--solver-deals 3000] [--train-deals 2000]
                                            [--test-deals 400] [--seeds 0,1,2]
"""
import argparse
import os
import random
import sys
import time
import warnings
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import solver as S  # noqa: E402
from rule_shift import features  # noqa: E402  (6-card features)
from size_shift_followup import make_model  # noqa: E402

warnings.filterwarnings("ignore")
R = S.OFFICIAL


# ------------------------------------------------------------ rule

def straight_bonus(s1, s2, table):
    """Bonus (P1 - P2) for the highest straight; 0 if none or nobody scores it."""
    holders = {}
    for who, cards in ((1, s1), (2, s2), (0, (table,))):
        for r, _ in cards:
            holders.setdefault(r, set()).add(who)
    for top in range(10, 2, -1):
        if top in holders and top - 1 in holders and top - 2 in holders:
            h = holders[top] - {0}
            if len(h) != 1:          # top card on the table only, or both players hold it
                return 0
            pts = top + (top - 1) + (top - 2)
            return pts if h == {1} else -pts
    return 0


def final_margin(s1, s2, table, straight):
    m = S.stack_value(s1, table[1], R) - S.stack_value(s2, table[1], R)
    return m + (straight_bonus(s1, s2, table) if straight else 0)


@lru_cache(maxsize=None)
def swap(s1, s2, table, straight):
    """Official swap rule: Player 2 swaps first, Player 1 may make any exchange."""
    def opts(a, b):
        yield a, b
        for _, na, nb in S._exchanges(a, b):
            yield na, nb
    return min(max(final_margin(u1, u2, table, straight) for u1, u2 in opts(t1, t2))
               for t2, t1 in opts(s2, s1))


@lru_cache(maxsize=None)
def value(row, s1, s2, off, straight):
    if len(row) == 1:
        last = row[0]
        return (last[0] if off == 2 else -last[0]) + swap(s1, s2, last, straight)
    return (max if off == 1 else min)(offer_value(row, s1, s2, off, i, straight) for i in range(len(row) - 1))


def offer_value(row, s1, s2, off, i, straight):
    vals = [d + value(r, a, b, 3 - off, straight) for _, d, r, a, b in S._choices(row, s1, s2, off, i)]
    return min(vals) if off == 1 else max(vals)


def clear():
    value.cache_clear()


# ------------------------------------------------------------ learning

def rollout(deal, eps, rng, straight):
    row, s1, s2, off = tuple(deal), (), (), 1
    out = []
    while True:
        out.append((row, s1, s2, off))
        if len(row) == 1:
            return out
        if rng.random() < eps:
            i = rng.randrange(len(row) - 1)
        else:
            i = (max if off == 1 else min)(range(len(row) - 1),
                                           key=lambda j: offer_value(row, s1, s2, off, j, straight))
        opts = list(S._choices(row, s1, s2, off, i))
        if rng.random() < eps:
            o = opts[rng.randrange(2)]
        else:
            o = (min if off == 1 else max)(opts, key=lambda o: o[1] + value(o[2], o[3], o[4], 3 - off, straight))
        _, _, row, s1, s2 = o
        off = 3 - off


def dataset(deals, rng, straight, augment=2):
    X, y = [], []
    for d in deals:
        seen = set()
        for eps in (0.0, 0.25, 0.5, 1.0, 1.0):
            for st in rollout(d, eps, rng, straight):
                if st in seen:
                    continue
                seen.add(st)
                v = value(*st, straight)
                for p in [(0, 1, 2, 3)] + [tuple(rng.sample(range(4), 4)) for _ in range(augment)]:
                    X.append(features(*st, perm=p))
                    y.append(v)
        clear()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class ModelPolicy:
    def __init__(self, model):
        self.m = model

    def _vals(self, opts, off):
        x = np.array([features(o[2], o[3], o[4], 3 - off) for o in opts], dtype=np.float32)
        return [o[1] + v for o, v in zip(opts, self.m.predict(x))]

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


class Greedy:  # same definition as basic_stats.py / size_shift.py
    def offer(self, row, s1, s2, off):
        return max(range(len(row) - 1), key=lambda i: min(row[i][0], row[i + 1][0]) - max(row[i][0], row[i + 1][0]))

    def choose(self, opts, off):
        return max(opts, key=lambda o: o[0][0])


def play(deal, policy, seat, straight):
    """Policy plays `seat`, perfect play the other; swaps perfect.
    Returns (final margin, optimal decisions, decisions) for the policy."""
    row, s1, s2, off, margin = tuple(deal), (), (), 1, 0
    good = total = 0
    while len(row) > 1:
        ovals = [offer_value(row, s1, s2, off, j, straight) for j in range(len(row) - 1)]
        if off == seat:
            i = policy.offer(row, s1, s2, off)
            total += 1
            good += ovals[i] == (max(ovals) if off == 1 else min(ovals))
        else:
            i = ovals.index(max(ovals) if off == 1 else min(ovals))
        opts = list(S._choices(row, s1, s2, off, i))
        cvals = [o[1] + value(o[2], o[3], o[4], 3 - off, straight) for o in opts]
        if 3 - off == seat:
            o = policy.choose(opts, off)
            total += 1
            good += cvals[opts.index(o)] == (min(cvals) if off == 1 else max(cvals))
        else:
            o = opts[cvals.index(min(cvals) if off == 1 else max(cvals))]
        margin += o[1]
        _, _, row, s1, s2 = o
        off = 3 - off
    return margin + value(row, s1, s2, off, straight), good, total


def evaluate(policy, deals, straight):
    """Per deal: points lost (both seats averaged), optimal decisions, decisions."""
    out = []
    for d in deals:
        v = value(tuple(d), (), (), 1, straight)
        m1, g1, t1 = play(d, policy, 1, straight)
        m2, g2, t2 = play(d, policy, 2, straight)
        assert m1 <= v and m2 >= v
        out.append(((v - m1 + m2 - v) / 2, g1 + g2, t1 + t2))
        clear()
    return np.array(out, dtype=float)


# ------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-deals", type=int, default=3000)
    ap.add_argument("--train-deals", type=int, default=2000)
    ap.add_argument("--test-deals", type=int, default=400)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()

    # sanity: without the straight this is the official game
    rng = random.Random(0)
    for _ in range(100):
        d = tuple(rng.sample(S.DECK, 6))
        assert value(d, (), (), 1, False) == S.solve(d)
        clear(); S.value.cache_clear()
    print("sanity: without the straight, values match solver.py on 100 deals")

    # ---- part 1: solver
    rng = random.Random(1)
    deals = [rng.sample(S.DECK, 6) for _ in range(args.solver_deals)]
    sg = lambda x: (x > 0) - (x < 0)
    base, strt, possible = [], [], 0
    for d in deals:
        base.append(value(tuple(d), (), (), 1, False))
        strt.append(value(tuple(d), (), (), 1, True))
        clear()
        ranks = {r for r, _ in d}
        possible += any(t in ranks and t - 1 in ranks and t - 2 in ranks for t in range(3, 11))
    n = len(deals)
    wb = sum(v > 0 for v in base) / n
    ws = sum(v > 0 for v in strt) / n
    ch = sum(a != b for a, b in zip(base, strt)) / n
    rc = sum(sg(a) != sg(b) for a, b in zip(base, strt)) / n
    print(f"\nPART 1 - solver ({n} deals)")
    print(f"  a straight is possible (three consecutive ranks among the 6 cards) in {possible / n:.1%} of deals")
    print(f"  P1 wins: official {wb:.1%}, with straight {ws:.1%}  (change {100 * (ws - wb):+.1f} points)")
    print(f"  deals whose perfect-play value changes: {ch:.1%}; whose result changes: {rc:.1%}")
    print(f"  mean |value|: official {np.mean(np.abs(base)):.2f}, with straight {np.mean(np.abs(strt)):.2f}")
    print(f"  (S1) P1 win rate within 5 points: {'CONFIRMED' if abs(ws - wb) < 0.05 else 'not confirmed'}")
    print(f"  (S2) result changes in under 15% of deals: {'CONFIRMED' if rc < 0.15 else 'not confirmed'}")

    # ---- part 2: learning
    rt = random.Random(999)
    test = [rt.sample(S.DECK, 6) for _ in range(args.test_deals)]
    tkeys = {tuple(map(tuple, d)) for d in test}
    greedy = {st: evaluate(Greedy(), test, st) for st in (False, True)}
    res = {}
    for seed in seeds:
        rng = random.Random(seed)
        train = [d for d in (rng.sample(S.DECK, 6) for _ in range(args.train_deals))
                 if tuple(map(tuple, d)) not in tkeys]
        for st in (False, True):
            X, y = dataset(train, random.Random(seed * 10 + 1), st)
            for kind in ("mlp", "trees"):
                pol = ModelPolicy(make_model(kind, seed).fit(X, y))
                res.setdefault((kind, st), []).append(evaluate(pol, test, st))
                print(f"  seed {seed} {kind} straight={st} [{time.time() - t0:.0f}s]", flush=True)

    print(f"\nPART 2 - learning ({len(test)} test deals, seeds {seeds})")
    name = {False: "official", True: "straight"}
    for kind in ("mlp", "trees"):
        print(f"\n  {kind}:")
        rates = {}
        for st in (False, True):
            runs = res[(kind, st)]
            rate = [r[:, 1].sum() / r[:, 2].sum() for r in runs]
            pts = np.mean([r[:, 0] for r in runs], axis=0)
            g = greedy[st]
            rates[st] = rate
            print(f"    {name[st]:<9} optimal-move rate {np.mean(rate):.1%} (per seed "
                  + " ".join(f"{x:.1%}" for x in rate) + f"); points lost {pts.mean():.2f}; "
                  f"greedy {g[:, 0].mean():.2f}; model / greedy {pts.mean() / g[:, 0].mean():.2f}; "
                  f"greedy optimal-move rate {g[:, 1].sum() / g[:, 2].sum():.1%}")
        drop = [a - b for a, b in zip(rates[False], rates[True])]
        print(f"    (L1) optimal-move rate drop (official - straight): {100 * np.mean(drop):+.1f} points; per seed "
              + " ".join(f"{100 * x:+.1f}" for x in drop)
              + f" -> {'CONFIRMED' if np.mean(drop) >= 0.03 else 'not confirmed'}")
        r0 = np.mean([r[:, 0] for r in res[(kind, False)]]) / greedy[False][:, 0].mean()
        r1 = np.mean([r[:, 0] for r in res[(kind, True)]]) / greedy[True][:, 0].mean()
        print(f"    (L2) model/greedy loss ratio: official {r0:.2f}, straight {r1:.2f} -> "
              f"{'CONFIRMED' if r1 > r0 else 'not confirmed'}")
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
