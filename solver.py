#!/usr/bin/env python3
"""Exact solver for Last Suit Standing (see RULES.md and solver.md).

Both players play perfectly. Each tries to maximize their own final margin.
Standard library only; Python 3.8+.

Examples:
    python3 solver.py                              # random deal
    python3 solver.py 3S 4D 10H 2C 7S AH           # a specific deal, left to right
    python3 solver.py --deals 3000                 # statistics over random deals
    python3 solver.py --deals 3000 --swap undo     # alternative swap reading
    python3 solver.py 3S 4D 10H 2C 7S AH --check   # cross-check with brute force
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from functools import lru_cache
from typing import NamedTuple

Card = tuple  # (rank 1-10, suit letter in "SHDC")
SUITS = "SHDC"
SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
DECK = [(r, s) for r in range(1, 11) for s in SUITS]


class Rules(NamedTuple):
    """Rule options. The defaults are the official rules in RULES.md."""

    swap: str = "any"      # "any": second swapper may make any exchange;
                           # "undo": after an exchange, only pass or undo it; after a pass, any exchange
    trump: str = "double"  # "double": trump cards score x2; "half": x1/2, rounded down (test variant)


OFFICIAL = Rules()


# ---------------------------------------------------------------- cards

def card_str(c: Card) -> str:
    r, s = c
    return ("A" if r == 1 else str(r)) + SYMBOLS[s]


def parse_card(text: str) -> Card:
    t = text.strip().upper()
    for letter, sym in SYMBOLS.items():
        t = t.replace(sym, letter)
    if len(t) < 2 or t[-1] not in SUITS:
        raise ValueError(f"bad card {text!r}: use forms like AS, 7H, 10D")
    rank = 1 if t[:-1] in ("A", "1") else int(t[:-1])
    if not 1 <= rank <= 10:
        raise ValueError(f"bad rank in {text!r}")
    return (rank, t[-1])


# ---------------------------------------------------------------- scoring

def stack_value(stack, trump_suit: str, rules: Rules = OFFICIAL) -> int:
    """Final value of a stack: trump multiplier, then x2 for each card in a pair.

    Each card is in at most one pair; with three of a rank the owner pairs the
    two highest-valued cards (the trump one, if any).
    """
    by_rank = defaultdict(list)
    for r, s in stack:
        v = r
        if s == trump_suit:
            v = v * 2 if rules.trump == "double" else v // 2
        by_rank[r].append(v)
    total = 0
    for vals in by_rank.values():
        vals.sort(reverse=True)
        paired = len(vals) // 2 * 2
        total += 2 * sum(vals[:paired]) + sum(vals[paired:])
    return total


def _exchanges(mine, theirs):
    """All single exchanges: yields ((give, get), new_mine, new_theirs)."""
    for i, give in enumerate(mine):
        for j, get in enumerate(theirs):
            m, t = list(mine), list(theirs)
            m[i], t[j] = get, give
            yield (give, get), tuple(m), tuple(t)


@lru_cache(maxsize=None)
def swap_phase(s1, s2, trump_suit: str, rules: Rules = OFFICIAL):
    """Perfect swap play. Player 2 (last chooser) acts first, then Player 1.

    Returns (stack margin P1 - P2, P2 action, P1 action); an action is None
    for a pass or (card given, card received) from that player's view.
    """
    def margin(a, b):
        return stack_value(a, trump_suit, rules) - stack_value(b, trump_suit, rules)

    def p1_reply(a, b, p2_action):
        options = [(margin(a, b), None)]
        if rules.swap == "any" or p2_action is None:
            options += [(margin(na, nb), act) for act, na, nb in _exchanges(a, b)]
        else:  # "undo": Player 1 may only reverse Player 2's exchange
            give2, get2 = p2_action
            undo_a = tuple(get2 if c == give2 else c for c in a)
            undo_b = tuple(give2 if c == get2 else c for c in b)
            options.append((margin(undo_a, undo_b), (give2, get2)))
        return max(options, key=lambda o: o[0])

    candidates = [(None, s1, s2)]
    candidates += [(act, na, nb) for act, nb, na in _exchanges(s2, s1)]
    best = None
    for p2_action, a, b in candidates:
        value, p1_action = p1_reply(a, b, p2_action)
        if best is None or value < best[0]:
            best = (value, p2_action, p1_action)
    return best


# ---------------------------------------------------------------- play phase

def _choices(row, s1, s2, offerer, i):
    """The chooser's two options for the offer (row[i], row[i+1]).

    Yields (taken card, chip change P1 - P2, new row, new s1, new s2).
    """
    for k in (0, 1):
        take, stay = row[i + k], row[i + 1 - k]
        new_row = row[: i + k] + row[i + k + 1:]
        if offerer == 1:
            yield take, stay[0] - take[0], new_row, s1, tuple(sorted(s2 + (take,)))
        else:
            yield take, take[0] - stay[0], new_row, tuple(sorted(s1 + (take,))), s2


@lru_cache(maxsize=None)
def value(row, s1=(), s2=(), offerer=1, rules: Rules = OFFICIAL) -> int:
    """Margin still to come (P1 - P2) from this position under perfect play."""
    if len(row) == 1:
        last = row[0]
        bonus = last[0] if offerer == 2 else -last[0]  # last offerer = the one who just moved
        return bonus + swap_phase(s1, s2, last[1], rules)[0]
    return (max if offerer == 1 else min)(
        offer_value(row, s1, s2, offerer, i, rules) for i in range(len(row) - 1)
    )


def offer_value(row, s1, s2, offerer, i, rules: Rules = OFFICIAL) -> int:
    """Value of offering (row[i], row[i+1]); the chooser replies perfectly."""
    vals = [d + value(r, a, b, 3 - offerer, rules) for _, d, r, a, b in _choices(row, s1, s2, offerer, i)]
    return min(vals) if offerer == 1 else max(vals)


def _check_even(row):
    if len(row) % 2:
        raise ValueError(
            f"solver.py handles an even number of cards only (got {len(row)}): it assumes Player 1 "
            "makes the last offer and Player 2 swaps first. For any card count, use "
            "experiments/size_shift.py, whose solver lets the last chooser swap first.")


def solve(row, rules: Rules = OFFICIAL) -> int:
    """Final margin (P1 - P2) of a deal under perfect play (even number of cards)."""
    _check_even(row)
    return value(tuple(row), (), (), 1, rules)


def opening_values(row, rules: Rules = OFFICIAL):
    """Final margin after each of Player 1's five possible opening offers."""
    _check_even(row)
    row = tuple(row)
    return [offer_value(row, (), (), 1, i, rules) for i in range(len(row) - 1)]


def principal_line(row, rules: Rules = OFFICIAL):
    """One perfect line of play as printable lines, plus the final scores."""
    _check_even(row)
    row, s1, s2, offerer = tuple(row), (), (), 1
    chips = {1: 0, 2: 0}
    out = []
    while len(row) > 1:
        pick = max if offerer == 1 else min
        i = pick(range(len(row) - 1), key=lambda j: offer_value(row, s1, s2, offerer, j, rules))
        opts = list(_choices(row, s1, s2, offerer, i))
        chooser_pick = min if offerer == 1 else max
        take, _, nrow, ns1, ns2 = chooser_pick(opts, key=lambda o: o[1] + value(o[2], o[3], o[4], 3 - offerer, rules))
        stay = row[i] if take == row[i + 1] else row[i + 1]
        chooser = 3 - offerer
        chips[chooser] += take[0]
        chips[offerer] += stay[0]
        out.append(f"P{offerer} offers {card_str(row[i])} {card_str(row[i + 1])}; "
                   f"P{chooser} takes {card_str(take)}, P{offerer} scores {card_str(stay)}   "
                   f"chips {chips[1]}-{chips[2]}")
        row, s1, s2, offerer = nrow, ns1, ns2, chooser
    last = row[0]
    chips[1] += last[0]  # Player 1 is always the last offerer with six cards
    out.append(f"Last card {card_str(last)}: trump is {SYMBOLS[last[1]]}; P1 collects {last[0]} more   "
               f"chips {chips[1]}-{chips[2]}")
    _, a2, a1 = swap_phase(s1, s2, last[1], rules)

    def show(act):
        return "pass" if act is None else f"gives {card_str(act[0])} for {card_str(act[1])}"

    out.append(f"Swap: P2 {show(a2)}; P1 {show(a1)}")
    # apply swaps to report final stacks
    if a2 is not None:
        give, get = a2
        s2 = tuple(get if c == give else c for c in s2)
        s1 = tuple(give if c == get else c for c in s1)
    if a1 is not None:
        give, get = a1
        s1 = tuple(get if c == give else c for c in s1)
        s2 = tuple(give if c == get else c for c in s2)
    t1 = chips[1] + stack_value(s1, last[1], rules)
    t2 = chips[2] + stack_value(s2, last[1], rules)
    out.append(f"Final: P1 {' '.join(map(card_str, s1))} -> {t1}   "
               f"P2 {' '.join(map(card_str, s2))} -> {t2}")
    return out, t1, t2


# ---------------------------------------------------------------- brute force check

def brute_force(row, rules: Rules = OFFICIAL) -> int:
    """Unmemoized full game-tree search, written separately, for validation."""
    def swap(s1, s2, trump):
        def m(a, b):
            return stack_value(a, trump, rules) - stack_value(b, trump, rules)

        def swaps(a, b):
            res = []
            for x in a:
                for y in b:
                    res.append((x, y,
                                [y if c == x else c for c in a],
                                [x if c == y else c for c in b]))
            return res

        p2_results = []
        p2_results.append(max([m(s1, s2)] + [m(na, nb) for _, _, na, nb in swaps(s1, s2)]))
        for x, y, nb, na in swaps(s2, s1):  # P2 gives x, gets y
            if rules.swap == "any":
                p2_results.append(max([m(na, nb)] + [m(a2, b2) for _, _, a2, b2 in swaps(na, nb)]))
            else:
                p2_results.append(max(m(na, nb), m(s1, s2)))
        return min(p2_results)

    def play(row, s1, s2, offerer, margin):
        if len(row) == 1:
            last = row[0]
            return margin + last[0] + swap(s1, s2, last[1])
        offers = []
        for i in range(len(row) - 1):
            replies = []
            for k in (0, 1):
                take, stay = row[i + k], row[i + 1 - k]
                rest = row[: i + k] + row[i + k + 1:]
                if offerer == 1:
                    replies.append(play(rest, s1, s2 + [take], 2, margin + stay[0] - take[0]))
                else:
                    replies.append(play(rest, s1 + [take], s2, 1, margin + take[0] - stay[0]))
            offers.append(min(replies) if offerer == 1 else max(replies))
        return max(offers) if offerer == 1 else min(offers)

    return play(list(row), [], [], 1, 0)


# ---------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(description="Exact solver for Last Suit Standing.")
    ap.add_argument("cards", nargs="*", help="six cards left to right, e.g. 3S 4D 10H 2C 7S AH")
    ap.add_argument("--deals", type=int, default=0, help="report statistics over N random deals")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    ap.add_argument("--swap", choices=["any", "undo"], default="any", help="swap reading (default: any)")
    ap.add_argument("--trump", choices=["double", "half"], default="double", help="trump scoring (default: double)")
    ap.add_argument("--check", action="store_true", help="also verify the deal by brute force")
    args = ap.parse_args(argv)
    rules = Rules(swap=args.swap, trump=args.trump)
    rng = random.Random(args.seed)

    if args.deals:
        wins = ties = total = 0
        for _ in range(args.deals):
            v = solve(rng.sample(DECK, 6), rules)
            value.cache_clear()
            wins += v > 0
            ties += v == 0
            total += v
        n = args.deals
        print(f"{n} random deals, rules: swap={rules.swap}, trump={rules.trump}")
        print(f"Player 1 wins {wins / n:.1%}, ties {ties / n:.1%}, loses {(n - wins - ties) / n:.1%}; "
              f"mean margin {total / n:+.2f}")
        return 0

    if args.cards:
        if len(args.cards) != 6:
            ap.error("give exactly six cards")
        try:
            row = [parse_card(c) for c in args.cards]
        except ValueError as e:
            ap.error(str(e))
        if len(set(row)) != 6:
            ap.error("the six cards must be different")
    else:
        row = rng.sample(DECK, 6)

    print("Deal:", " ".join(map(card_str, row)), f"  (rules: swap={rules.swap}, trump={rules.trump})")
    ov = opening_values(row, rules)
    print("Value of each opening offer for P1:",
          ", ".join(f"{card_str(row[i])}{card_str(row[i + 1])} {v:+d}" for i, v in enumerate(ov)))
    lines, t1, t2 = principal_line(row, rules)
    print("Perfect play:")
    for ln in lines:
        print("  " + ln)
    v = solve(row, rules)
    result = "Player 1 wins" if v > 0 else "Player 2 wins" if v < 0 else "Tie"
    print(f"Result: {result}, margin {v:+d}")
    assert t1 - t2 == v, "internal check failed: replayed line does not match game value"
    if args.check:
        b = brute_force(row, rules)
        print(f"Brute-force check: {'passed' if b == v else 'FAILED'} ({b:+d})")
        return 0 if b == v else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
