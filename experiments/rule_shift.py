#!/usr/bin/env python3
"""Rule-shift experiment: does a learned value network survive a one-sentence rule change?

Question
    A small network learns to evaluate positions under one swap rule. When the
    swap rule changes (official "any" -> "undo"), how many extra points does it
    lose, compared with a network trained on the new rule itself?

Design
    - Two networks of identical size: NET_any trained on official-rule values,
      NET_undo trained on undo-rule values. Same deals, same features, same seed.
    - Training positions come from rollouts mixing perfect and random moves, so
      the network sees off-path positions too. Labels are exact solver values.
    - Policy: one-step lookahead with the network's value of the next position.
      The network makes every offer and choice for its seat. The swap phase is
      played perfectly by both sides, so any loss comes from offer/choice moves
      made with a wrong picture of how the game will end.
    - Opponent: the exact solver, under the rules actually in force.
    - Metric: regret = points lost versus perfect play by the network's seat,
      on held-out deals. Brittleness = regret(NET_any on undo) - regret(NET_undo on undo).
    - Baselines: random and greedy policies; perfect play has regret 0 by definition.

Usage: python3 experiments/rule_shift.py [--train-deals N] [--test-deals N] [--seeds K]
"""
import argparse
import os
import random
import sys
import time

import numpy as np
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import solver as S  # noqa: E402

RULES = {"any": S.Rules(swap="any"), "undo": S.Rules(swap="undo")}
SUIT_INDEX = {s: i for i, s in enumerate(S.SUITS)}


# ------------------------------------------------------------ features

def card_feats(c, perm):
    if c is None:
        return [0.0] * 6
    r, s = c
    oh = [0.0] * 4
    oh[perm[SUIT_INDEX[s]]] = 1.0
    return [1.0, r / 10.0] + oh


def features(row, s1, s2, offerer, perm=(0, 1, 2, 3)):
    f = []
    for i in range(6):
        f += card_feats(row[i] if i < len(row) else None, perm)
    for stack in (s1, s2):
        st = sorted(stack)
        for i in range(3):
            f += card_feats(st[i] if i < len(st) else None, perm)
    # simple aggregates: rank sum per suit per stack, pair flags
    for stack in (s1, s2):
        sums = [0.0] * 4
        for r, s in stack:
            sums[perm[SUIT_INDEX[s]]] += r / 10.0
        f += sums
        ranks = [r for r, _ in stack]
        f.append(1.0 if len(ranks) != len(set(ranks)) else 0.0)
    f += [1.0 if offerer == 1 else 0.0, len(row) / 6.0]
    return f


# ------------------------------------------------------------ data

def rollout_states(deal, rules, eps, rng):
    """States visited by a policy that plays perfectly with prob 1-eps, else randomly."""
    row, s1, s2, off = tuple(deal), (), (), 1
    states = []
    while True:
        states.append((row, s1, s2, off))
        if len(row) == 1:
            return states
        if rng.random() < eps:
            i = rng.randrange(len(row) - 1)
        else:
            pick = max if off == 1 else min
            i = pick(range(len(row) - 1), key=lambda j: S.offer_value(row, s1, s2, off, j, rules))
        opts = list(S._choices(row, s1, s2, off, i))
        if rng.random() < eps:
            o = opts[rng.randrange(2)]
        else:
            cp = min if off == 1 else max
            o = cp(opts, key=lambda o: o[1] + S.value(o[2], o[3], o[4], 3 - off, rules))
        _, _, row, s1, s2 = o
        off = 3 - off


def make_dataset(deals, rules, rng, augment=2):
    X, y = [], []
    for deal in deals:
        seen = set()
        for eps in (0.0, 0.25, 0.5, 1.0, 1.0):
            for st in rollout_states(deal, rules, eps, rng):
                if st in seen:
                    continue
                seen.add(st)
                v = S.value(*st, rules)
                perms = [(0, 1, 2, 3)] + [tuple(rng.sample(range(4), 4)) for _ in range(augment)]
                for p in perms:
                    X.append(features(*st, perm=p))
                    y.append(v)
        S.value.cache_clear()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ------------------------------------------------------------ policies

class NetPolicy:
    def __init__(self, net):
        self.net = net

    def values(self, states):
        return self.net.predict(np.array([features(*st) for st in states], dtype=np.float32))

    def offer(self, row, s1, s2, off):
        best = None
        for i in range(len(row) - 1):
            opts = list(S._choices(row, s1, s2, off, i))
            vs = self.values([(o[2], o[3], o[4], 3 - off) for o in opts])
            tot = [o[1] + v for o, v in zip(opts, vs)]
            cv = min(tot) if off == 1 else max(tot)  # assume chooser replies well
            if best is None or (cv > best[0] if off == 1 else cv < best[0]):
                best = (cv, i)
        return best[1]

    def choose(self, opts, off):
        vs = self.values([(o[2], o[3], o[4], 3 - off) for o in opts])
        tot = [o[1] + v for o, v in zip(opts, vs)]
        k = int(np.argmin(tot)) if off == 1 else int(np.argmax(tot))
        return opts[k]


class GreedyPolicy:
    """Offer the pair that maximizes immediate chips assuming the chooser takes the higher card;
    as chooser, take the higher card."""

    def offer(self, row, s1, s2, off):
        return max(range(len(row) - 1), key=lambda i: min(row[i][0], row[i + 1][0]))

    def choose(self, opts, off):
        return max(opts, key=lambda o: o[0][0])


class RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def offer(self, row, s1, s2, off):
        return self.rng.randrange(len(row) - 1)

    def choose(self, opts, off):
        return opts[self.rng.randrange(2)]


def play(deal, policy, seat, rules):
    """Policy controls `seat` (1 or 2); perfect solver controls the other.
    Swap phase is perfect for both. Returns final margin P1 - P2."""
    row, s1, s2, off, margin = tuple(deal), (), (), 1, 0
    while len(row) > 1:
        if off == seat:
            i = policy.offer(row, s1, s2, off)
        else:
            pick = max if off == 1 else min
            i = pick(range(len(row) - 1), key=lambda j: S.offer_value(row, s1, s2, off, j, rules))
        opts = list(S._choices(row, s1, s2, off, i))
        chooser = 3 - off
        if chooser == seat:
            o = policy.choose(opts, off)
        else:
            cp = min if off == 1 else max
            o = cp(opts, key=lambda o: o[1] + S.value(o[2], o[3], o[4], 3 - off, rules))
        margin += o[1]
        _, _, row, s1, s2 = o
        off = 3 - off
    return margin + S.value(row, s1, s2, off, rules)  # last-card bonus + perfect swaps


def evaluate(policy, deals, rules):
    """Per deal: regret for each seat (points lost vs perfect play) and outcome downgrades."""
    out = []
    for d in deals:
        v = S.solve(d, rules)
        m1 = play(d, policy, 1, rules)
        m2 = play(d, policy, 2, rules)
        sgn = lambda x: (x > 0) - (x < 0)
        out.append((v - m1, m2 - v,                       # regret as P1, as P2 (>= 0)
                    int(sgn(m1) < sgn(v)), int(sgn(m2) > sgn(v))))  # result got worse
        S.value.cache_clear()
    return np.array(out, dtype=float)


def boot_ci(x, rng, n=4000):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


# ------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-deals", type=int, default=3000)
    ap.add_argument("--test-deals", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=str, default="256,128")
    args = ap.parse_args()
    hidden = tuple(int(h) for h in args.hidden.split(","))
    t0 = time.time()

    rng_test = random.Random(999)
    test = [rng_test.sample(S.DECK, 6) for _ in range(args.test_deals)]
    # flip deals: best opening offer differs between the two rules
    flip = []
    for d in test:
        oa = S.opening_values(d, RULES["any"]); S.value.cache_clear()
        ou = S.opening_values(d, RULES["undo"]); S.value.cache_clear()
        flip.append(int(np.argmax(oa)) != int(np.argmax(ou)))
    flip = np.array(flip)
    print(f"test deals: {len(test)}, flip deals (best opening differs): {flip.sum()}")

    results = {}  # (train, test) -> list over seeds of per-deal arrays
    for seed in range(args.seeds):
        rng = random.Random(seed)
        train = [rng.sample(S.DECK, 6) for _ in range(args.train_deals)]
        tset = {tuple(map(tuple, d)) for d in test}
        train = [d for d in train if tuple(map(tuple, d)) not in tset]
        for tr in ("any", "undo"):
            X, y = make_dataset(train, RULES[tr], random.Random(seed * 10 + 1))
            net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=200, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=10,
                               learning_rate_init=1e-3, batch_size=256, random_state=seed)
            net.fit(X, y)
            # fit quality on the training rule, measured on held-out test-deal states
            Xt, yt = make_dataset(test[:100], RULES[tr], random.Random(5), augment=0)
            mae = float(np.mean(np.abs(net.predict(Xt) - yt)))
            print(f"seed {seed} net_{tr}: {len(y)} training states, held-out MAE {mae:.2f} "
                  f"[{time.time() - t0:.0f}s]")
            pol = NetPolicy(net)
            for te in ("any", "undo"):
                results.setdefault((tr, te), []).append(evaluate(pol, test, RULES[te]))

    base = {}
    for name, pol in (("greedy", GreedyPolicy()), ("random", RandomPolicy(random.Random(7)))):
        for te in ("any", "undo"):
            base[(name, te)] = evaluate(pol, test, RULES[te])

    rng_b = np.random.default_rng(0)

    def summarize(arr, mask=None):
        a = arr if mask is None else arr[mask]
        reg = a[:, :2].mean(axis=1)          # mean regret per game, both seats
        worse = a[:, 2:].mean(axis=1)        # share of games whose result got worse
        return reg.mean(), boot_ci(reg, rng_b), worse.mean()

    print("\nMean points lost per game vs perfect play (both seats averaged), 95% CI over deals")
    print(f"{'policy':<22}{'tested on':<11}{'all deals':>26}{'flip deals':>14}{'results worsened':>19}")
    for (tr, te), runs in results.items():
        arr = np.mean(runs, axis=0)  # average over seeds per deal
        m, ci, w = summarize(arr)
        mf, _, _ = summarize(arr, flip)
        print(f"{'net trained on ' + tr:<22}{te:<11}{m:>8.2f} [{ci[0]:.2f}, {ci[1]:.2f}]{mf:>14.2f}{w:>18.1%}")
    for (name, te), arr in base.items():
        m, ci, w = summarize(arr)
        mf, _, _ = summarize(arr, flip)
        print(f"{name:<22}{te:<11}{m:>8.2f} [{ci[0]:.2f}, {ci[1]:.2f}]{mf:>14.2f}{w:>18.1%}")

    # brittleness: paired difference on the same deals, per seed then pooled
    for te, other in (("undo", "any"), ("any", "undo")):
        own = np.mean(results[(te, te)], axis=0)[:, :2].mean(axis=1)
        shifted = np.mean(results[(other, te)], axis=0)[:, :2].mean(axis=1)
        d = shifted - own
        ci = boot_ci(d, rng_b)
        dfl = d[flip]
        print(f"\nTested on '{te}': net trained on '{other}' loses {d.mean():+.2f} pts/game more than "
              f"net trained on '{te}' (95% CI {ci[0]:+.2f} to {ci[1]:+.2f}); "
              f"on flip deals {dfl.mean():+.2f}, on other deals {d[~flip].mean():+.2f}")
        per_seed = [np.mean(np.array(results[(other, te)][s])[:, :2] - np.array(results[(te, te)][s])[:, :2])
                    for s in range(args.seeds)]
        print("  per seed:", " ".join(f"{x:+.2f}" for x in per_seed))
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
