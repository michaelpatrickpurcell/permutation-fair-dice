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

# ============================================================================

word_1 = "abcde"
n = len(word_1)
k = 3
m = 6
word_2 = word_1 + word_1[::-1]
all_perms = list(permutations(range(n)))
all_orders = list(permutations(word_1, k))

counts = [score_orders2(permute_letters(word_2, p), k) for p in tqdm(all_perms)]
target = m * sum(counts[0].values()) // len(all_orders)

xs = [
    pulp.LpVariable("x%i" % i, 0, 1, cat="Integer") for i in tqdm(range(len(all_perms)))
]
prob = pulp.LpProblem("myProblem", pulp.LpMaximize)
prob += pulp.lpSum(xs)  # This is the objective!
for order in tqdm(all_orders):
    prob += pulp.lpSum([x * ct[order] for x, ct in zip(xs, counts)]) <= target

status = prob.solve()
print("Values of variables")
for p, x in zip(all_perms, xs):
    print("%s: %s" % (p, pulp.value(x)))
print()
print("Values of Constraints")
for i, key in enumerate(prob.constraints.keys()):
    print("C%s: %s" % (i, pulp.value(prob.constraints[key])))

used_perms = [p for p, x in zip(all_perms, xs) if pulp.value(x) == 1.0]
word_3 = "".join([permute_letters(word_2, p) for p in used_perms])
word_4 = word_3 + word_3[::-1]

# alt_perms_23 = [
#     (0, 1, 2, 3, 4),
#     (3, 2, 1, 0, 4),
#     (4, 1, 2, 0, 3),
#     (0, 2, 1, 3, 4),
#     (3, 1, 2, 0, 4),
#     (4, 2, 1, 0, 3),
# ]
# word_3 = ''.join([permute_letters(word_2, p) for p in alt_perms_23])
# alt_perms_34 = [(0, 1, 2, 3, 4), (4, 2, 1, 3, 0)]
# word_4 = ''.join([permute_letters(word_3, p) for p in alt_perms_34])

# ----------------------------------------------------------------------------

# n = 5
k = 5
m = 2
all_perms = list(permutations(range(n)))
all_orders = list(permutations(word_1, k))

# Permute the 4/n word directly
counts = [score_orders2(permute_letters(word_4, p), k) for p in all_perms]

# Permute the starting atom first and then lift it to a 4/n fair word
# counts = []
# for p in all_perms:
#     root = permute_letters(word_2, p)
#     foo = [permute_letters(root, up) for up in used_perms]
#     bar = ''.join(foo)
#     ram = bar + bar[::-1]
#     counts.append(score_orders2(ram, k))

# Permute segments and then relabel
# foo = [permute_letters(word_2, up) for up in used_perms]
# counts = []
# for p in tqdm(list(permutations(range(6)))):
#     temp = apply_perm(p, foo)
#     bar = ''.join(temp)
#     ram = bar + bar[::-1]
#     counts.extend([score_orders2(permute_letters(ram, p2), k) for p2 in all_perms])

target = m * sum(counts[0].values()) // len(all_orders)

xs = [pulp.LpVariable("x%i" % i, 0, 1, cat="Integer") for i, _ in enumerate(counts)]
prob = pulp.LpProblem("myProblem", pulp.LpMaximize)
prob += pulp.lpSum(xs)  # This is the objective!
for order in tqdm(all_orders):
    prob += pulp.lpSum([x * ct[order] for x, ct in zip(xs, counts)]) <= target

status = prob.solve()
print("Values of variables")
for p, x in zip(all_perms, xs):
    print("%s: %s" % (p, pulp.value(x)))
print()
print("Values of Constraints")
for i, key in enumerate(prob.constraints.keys()):
    print("C%s: %s" % (i, pulp.value(prob.constraints[key])))

# ============================================================================

word_1 = "abcde"
n = len(word_1)
k = 3
# m = 12
word_2 = word_1 + word_1[::-1]
all_perms = list(permutations(range(n)))
all_orders = list(permutations(word_1, k))

counts = [score_orders2(permute_letters(word_2, p), k) for p in tqdm(all_perms)]
ncounts = [normalize_score(ct) for ct in counts]

target = 0.0

xs = [
    pulp.LpVariable("x%i" % i, 0, 1, cat="Integer") for i in tqdm(range(len(all_perms)))
]

prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
prob += pulp.lpSum(xs)  # This is the objective!
for order in tqdm(all_orders):
    prob += pulp.lpSum([x * ct[order] for x, ct in zip(xs, ncounts)]) == target
prob += xs[0] == 1

status = prob.solve()
print("Values of variables")
for p, x in zip(all_perms, xs):
    print("%s: %s" % (p, pulp.value(x)))
print()
print("Values of Constraints")
for i, key in enumerate(prob.constraints.keys()):
    print("C%s: %s" % (i, pulp.value(prob.constraints[key])))

used_perms = [p for p, x in zip(all_perms, xs) if pulp.value(x) == 1.0]
word_3 = "".join([permute_letters(word_2, p) for p in used_perms])
word_4 = word_3 + word_3[::-1]

# ----------------------------------------------------------------------------

k = 4
all_perms = list(permutations(range(n)))
all_orders = list(permutations(word_1, k))

# Permute the 4/n word directly
counts = [score_orders2(permute_letters(word_4, p), k) for p in all_perms]
ncounts = [normalize_score(ct) for ct in counts]

target = 0.0

xs = [
    pulp.LpVariable("x%i" % i, 0, 1, cat="Integer") for i in tqdm(range(len(all_perms)))
]

prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
prob += pulp.lpSum(xs)  # This is the objective!
for order in tqdm(all_orders):
    prob += pulp.lpSum([x * ct[order] for x, ct in zip(xs, ncounts)]) == target
prob += xs[0] == 1

status = prob.solve()


print("Values of variables")
for p, x in zip(all_perms, xs):
    print("%s: %s" % (p, pulp.value(x)))
print()
print("Values of Constraints")
for i, key in enumerate(prob.constraints.keys()):
    print("C%s: %s" % (i, pulp.value(prob.constraints[key])))


used_perms = [p for p, x in zip(all_perms, xs) if pulp.value(x) == 1.0]
word_5 = "".join([permute_letters(word_4, p) for p in used_perms])
