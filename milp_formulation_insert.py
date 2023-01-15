import random
import numpy as np

import pulp

from tqdm import tqdm
from scipy.special import factorial, comb
from collections import Counter
from itertools import combinations, permutations, product

from utils import coverage_search, coverage_search2
from utils import permute_letters, apply_perm
from utils import score_orders, score_orders2
from utils import aggregate_scores, normalize_score
from utils import word_to_dice, dice_to_word
from utils import l0_norm, l1_norm, l2_norm, l4_norm, max_norm
from utils import rotl

# ----------------------------------------------------------------------------


def milp_search_insert(
    word,
    order_len,
    solution_len=None,
    upBound=None,
    palindrome=False,
    positions=None,
    verbose=False,
):
    letters_list = list(word)
    last_letter = sorted(list(set(letters_list)))[-1]
    new_letter = chr(ord(last_letter) + 1)
    letters = "".join(sorted(list(set(letters_list))) + [new_letter])
    n = len(letters)
    if positions:
        all_positions = sorted(list(positions))
    else:
        all_positions = [i for i in range(len(word) + 1)]

    all_orders = list(permutations(letters, n))
    short_orders = list(permutations(letters, order_len))

    counts = []
    for i in tqdm(all_positions, disable=~verbose):
        temp = list(word)
        temp.insert(i, new_letter)
        counts.append(score_orders2(temp, n))

    xs = []
    for i in tqdm(all_positions, disable=~verbose):
        if upBound:
            xs.append(
                pulp.LpVariable("x%i" % i, lowBound=0, upBound=upBound, cat="Integer")
            )
        else:
            xs.append(pulp.LpVariable("x%i" % i, lowBound=0, cat="Integer"))

    m = len(word) // (n - 1)
    if solution_len is None:
        solution_len = m

    # target = ((m**n) // factorial(n, exact=True)) * factorial(n-order_len, exact=True) * comb(n, n-order_len, exact=True)
    # target = (m**n) // factorial(order_len, exact=True)
    target = (
        comb(n, n - order_len, exact=True)
        * factorial(n - order_len, exact=True)
        * np.mean(list(counts[0].values()))
        * solution_len
    )

    prob = pulp.LpProblem("myProblem", pulp.LpMaximize)
    prob += pulp.lpSum(xs) == solution_len

    for short_order in tqdm(short_orders, disable=~verbose):
        temp = []
        for order in tqdm(all_orders, disable=~verbose):
            temp_order = tuple([x for x in order if x in short_order])
            if short_order == temp_order:
                temp += [x * ct[order] for x, ct in zip(xs, counts)]
        prob += pulp.lpSum(temp) == target

    if palindrome:
        for x, y in zip(xs, xs[::-1]):
            prob += x == y

    status = prob.solve(pulp.PULP_CBC_CMD(msg=int(verbose)))

    if status == -1:
        ret = [""]
    else:
        solution = [pulp.value(x) for x in xs]
        ret = list(word)
        for i, mult in zip(all_positions[::-1], solution[::-1]):
            ret.insert(i, "".join(int(mult) * [new_letter]))

    return "".join(ret)


# ============================================================================

letters = "abcd"  # efg"
word2 = letters + letters[::-1]

search_results = milp_search(word2, 3, verbose=True)
if search_results:
    used_perms, segments_3 = search_results
    word_3 = "".join(segments_3)

search_results = milp_search(word_3, 4, solution_len=5, verbose=True)
if search_results:
    used_perms, segments_4 = search_results
    word4 = "".join(segments_4)

# word_4b = "".join([permute_letters(word_4, rotl((0, 1, 2, 3, 4), i)) for i in range(5)])
# segments_4b = ["".join(rotl(segments_4, i)) for i in range(6)]
# word_4b = "".join(segments_4b)

search_results = milp_search(word_4, order_len=5, solution_len=15, verbose=True)
if search_results:
    used_perms, segments_5 = search_results
    word_5 = "".join(segments_5)


# ============================================================================

seed3 = "cabbcabaaccbcabbca"
three_perms = list(permutations([0, 1, 2], 3))

hits = []
for i, perms in enumerate(product(three_perms, repeat=5)):
    print(i)
    word3 = "".join([permute_letters(seed3, perm) for perm in perms])
    word4 = milp_search_insert(word3, palindrome=True, upBound=2)
    if word4:
        print("hit")
        print("secondary test")
        word5 = milp_search_insert(word4)
        if word5:
            print("!!!!!HIT!!!!!")
            hits.append((word5))
        else:
            print("so close")
    else:
        print("miss")
