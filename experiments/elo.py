#!/usr/bin/env python3
"""Elo experiment (pre-registered in elo.md).

Round robin of six players on the official 6-card game. Every pair plays every
deal from both seats. Elo is fitted by maximum likelihood from win/tie/loss
(a tie scores half). True strength is the exact mean points lost against the
perfect player. All players play the swap phase perfectly.

Usage: python3 experiments/elo.py [--deals 300] [--seed 7] [--boot 300]
"""
import argparse
import itertools
import math
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
from size_shift_followup import train_policy, feats_plain, test_sets  # noqa: E402

warnings.filterwarnings("ignore")
R = S.OFFICIAL


# ------------------------------------------------------------ players

class Perfect:
    name = "Perfect"

    def offer(self, row, s1, s2, off):
        return (max if off == 1 else min)(range(len(row) - 1), key=lambda j: S.offer_value(row, s1, s2, off, j, R))

    def choose(self, opts, off):
        return (min if off == 1 else max)(opts, key=lambda o: o[1] + S.value(o[2], o[3], o[4], 3 - off, R))


def greedy_offer(row):
    return max(range(len(row) - 1), key=lambda i: min(row[i][0], row[i + 1][0]) - max(row[i][0], row[i + 1][0]))


def greedy_choose(opts):
    return max(opts, key=lambda o: o[0][0])


class Greedy:
    name = "Greedy"

    def offer(self, row, s1, s2, off):
        return greedy_offer(row)

    def choose(self, opts, off):
        return greedy_choose(opts)


@lru_cache(maxsize=None)
def vs_greedy(row, s1, s2, off, xseat):
    """Remaining margin (P1 - P2) when seat `xseat` plays its best response to a greedy
    opponent, who really is greedy. The swap phase is perfect for both."""
    if len(row) == 1:
        return S.value(row, s1, s2, off, R)
    sign = 1 if xseat == 1 else -1
    if off == xseat:  # X offers, greedy chooses
        best = None
        for i in range(len(row) - 1):
            _, d, r, a, b = greedy_choose(list(S._choices(row, s1, s2, off, i)))
            v = d + vs_greedy(r, a, b, 3 - off, xseat)
            if best is None or sign * v > sign * best:
                best = v
        return best
    i = greedy_offer(row)  # greedy offers, X chooses
    return max((d + vs_greedy(r, a, b, 3 - off, xseat) for _, d, r, a, b in S._choices(row, s1, s2, off, i)),
               key=lambda v: sign * v)


class Exploiter:
    """Best response to greedy, used against everyone."""
    name = "Exploiter"

    def __init__(self):
        self.seat = None  # set per game

    def offer(self, row, s1, s2, off):
        sign = 1 if off == 1 else -1
        def val(i):
            _, d, r, a, b = greedy_choose(list(S._choices(row, s1, s2, off, i)))
            return sign * (d + vs_greedy(r, a, b, 3 - off, off))
        return max(range(len(row) - 1), key=val)

    def choose(self, opts, off):
        me = 3 - off
        sign = 1 if me == 1 else -1
        return max(opts, key=lambda o: sign * (o[1] + vs_greedy(o[2], o[3], o[4], 3 - off, me)))


class RandomPlayer:
    name = "Random"

    def __init__(self):
        self.rng = random.Random(0)

    def reset(self, seed):
        self.rng = random.Random(seed)

    def offer(self, row, s1, s2, off):
        return self.rng.randrange(len(row) - 1)

    def choose(self, opts, off):
        return opts[self.rng.randrange(2)]


class Named:
    def __init__(self, pol, name):
        self.pol, self.name = pol, name

    def offer(self, *a):
        return self.pol.offer(*a)

    def choose(self, *a):
        return self.pol.choose(*a)


# ------------------------------------------------------------ games and Elo

def play(deal, p1, p2):
    row, s1, s2, off, margin = tuple(deal), (), (), 1, 0
    while len(row) > 1:
        offerer, chooser = (p1, p2) if off == 1 else (p2, p1)
        i = offerer.offer(row, s1, s2, off)
        o = chooser.choose(list(S._choices(row, s1, s2, off, i)), off)
        margin += o[1]
        _, _, row, s1, s2 = o
        off = 3 - off
    return margin + S.value(row, s1, s2, off, R)


def fit_elo(games, n_players, iters=3000, lr=20.0):
    """games: list of (a, b, score_for_a). Maximum-likelihood Elo, mean 1500."""
    a = np.array([g[0] for g in games]); b = np.array([g[1] for g in games])
    s = np.array([g[2] for g in games], dtype=float)
    r = np.zeros(n_players)
    for _ in range(iters):
        p = 1.0 / (1.0 + 10 ** ((r[b] - r[a]) / 400.0))
        g = np.zeros(n_players)
        np.add.at(g, a, s - p)
        np.add.at(g, b, p - s)
        r += lr * g / max(1, len(games) / n_players)
        r -= r.mean()
    return r + 1500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()
    t0 = time.time()

    tests = test_sets(200)  # only to exclude these deals from training, as before
    tkeys = {tuple(map(tuple, d)) for ds in tests.values() for d in ds}
    rng = random.Random(args.seed)
    deals = [rng.sample(S.DECK, 6) for _ in range(args.deals)]
    tkeys |= {tuple(map(tuple, d)) for d in deals}
    cache = {}
    mlp = Named(train_policy((6,), 0, "mlp", feats_plain, 1500, tkeys, cache), "MLP")
    trees = Named(train_policy((6,), 0, "trees", feats_plain, 1500, tkeys, cache), "Trees")
    rand = RandomPlayer()
    players = [Perfect(), Exploiter(), mlp, trees, Greedy(), rand]
    names = [p.name for p in players]
    print(f"models trained [{time.time() - t0:.0f}s]", flush=True)

    # sanity: Exploiter really is the best response to greedy
    for d in deals[:30]:
        assert play(d, players[1], players[4]) == vs_greedy(tuple(d), (), (), 1, 1)
        assert play(d, players[4], players[1]) == vs_greedy(tuple(d), (), (), 1, 2)
    print("sanity: Exploiter achieves its computed best response against Greedy")

    results = []  # (deal index, a, b, margin with a as P1 and b as P2)
    for di, d in enumerate(deals):
        for a, b in itertools.permutations(range(len(players)), 2):
            rand.reset(hash((di, a, b)) & 0xFFFFFFFF)
            results.append((di, a, b, play(d, players[a], players[b])))
        S.value.cache_clear(); vs_greedy.cache_clear()
    print(f"{len(results)} games played [{time.time() - t0:.0f}s]", flush=True)

    def to_games(rows):
        out = []
        for _, a, b, m in rows:
            out.append((a, b, 1.0 if m > 0 else 0.5 if m == 0 else 0.0))
        return out

    elo = fit_elo(to_games(results), len(players))

    # exact strength: points lost vs Perfect (index 0), both seats
    v = {di: S.solve(d) for di, d in enumerate(deals)}
    lost = {k: [] for k in range(len(players))}
    for di, a, b, m in results:
        if b == 0 and a != 0:
            lost[a].append(v[di] - m)   # a as P1 vs Perfect
        if a == 0 and b != 0:
            lost[b].append(m - v[di])   # b as P2 vs Perfect
    lost[0] = [0.0]
    strength = {k: float(np.mean(x)) for k, x in lost.items()}

    # score rates
    score = np.zeros(len(players)); count = np.zeros(len(players))
    for a, b, s in to_games(results):
        score[a] += s; score[b] += 1 - s; count[a] += 1; count[b] += 1

    # bootstrap over deals
    brng = np.random.default_rng(0)
    by_deal = {}
    for row in results:
        by_deal.setdefault(row[0], []).append(row)
    boots = []
    for _ in range(args.boot):
        pick = brng.integers(0, len(deals), size=len(deals))
        rows = [r for i in pick for r in by_deal[i]]
        boots.append(fit_elo(to_games(rows), len(players), iters=1500))
    boots = np.array(boots)

    order = np.argsort(-elo)
    print("\nRank by Elo (95% bootstrap CI over deals) vs exact strength")
    print(f"{'':<4}{'player':<11}{'Elo':>7}{'CI':>17}{'score':>8}{'pts lost vs Perfect':>22}{'true rank':>11}")
    true_rank = {k: i + 1 for i, k in enumerate(sorted(range(len(players)), key=lambda k: strength[k]))}
    for i, k in enumerate(order):
        lo, hi = np.percentile(boots[:, k], [2.5, 97.5])
        print(f"{i + 1:<4}{names[k]:<11}{elo[k]:>7.0f}  [{lo:>6.0f}, {hi:>6.0f}]{score[k] / count[k]:>8.1%}"
              f"{strength[k]:>22.2f}{true_rank[k]:>11}")

    print("\nHead-to-head score (row player's score rate vs column player, both seats)")
    print(f"{'':<11}" + "".join(f"{n:>10}" for n in names))
    for a in range(len(players)):
        cells = []
        for b in range(len(players)):
            if a == b:
                cells.append(f"{'—':>10}")
                continue
            ss = []
            for _, x, y, m in results:
                if x == a and y == b:
                    ss.append(1.0 if m > 0 else 0.5 if m == 0 else 0.0)
                elif x == b and y == a:
                    ss.append(1.0 if m < 0 else 0.5 if m == 0 else 0.0)
            cells.append(f"{np.mean(ss):>10.1%}")
        print(f"{names[a]:<11}" + "".join(cells))

    print("\nPre-registered predictions:")
    top = int(order[0])
    print(f"  1. Perfect ranks first by Elo: {'CONFIRMED' if top == 0 else 'not confirmed (' + names[top] + ' is first)'}")
    ex = 1
    above = [k for k in range(len(players))
             if k != ex and strength[k] < strength[ex] and elo[ex] > elo[k]]
    print(f"  2. Exploiter rated above an objectively stronger player: "
          f"{'CONFIRMED (above ' + ', '.join(names[k] for k in above) + ')' if above else 'not confirmed'}")
    for k in above:
        d = boots[:, ex] - boots[:, k]
        print(f"     Exploiter minus {names[k]}: {elo[ex] - elo[k]:+.0f} Elo, 95% CI "
              f"[{np.percentile(d, 2.5):+.0f}, {np.percentile(d, 97.5):+.0f}], P(gap > 0) = {np.mean(d > 0):.3f}")
    d0 = boots[:, ex] - boots[:, 0]
    print(f"  Exploiter minus Perfect: {elo[ex] - elo[0]:+.0f} Elo, 95% CI "
          f"[{np.percentile(d0, 2.5):+.0f}, {np.percentile(d0, 97.5):+.0f}]")
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
