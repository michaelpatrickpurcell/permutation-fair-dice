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
k = 4
m = 6
word_2 = word_1 + word_1[::-1]
all_perms = list(permutations(range(n)))
all_orders = list(permutations(word_1, k))

counts = [score_orders2(permute_letters(word_2, p), k) for p in tqdm(all_perms)]
target = m * sum(counts[0].values()) // len(all_orders)

xs = [
    pulp.LpVariable("x%i" % i, lowBound=0, cat="Integer")
    for i in tqdm(range(len(all_perms)))
]
prob = pulp.LpProblem("myProblem", pulp.LpMaximize)
prob += pulp.lpSum(xs) == m  # This is the objective!
for order in tqdm(all_orders):
    prob += pulp.lpSum([x * ct[order] for x, ct in zip(xs, counts)]) <= target

prob += xs[0] >= 1

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
# word_4 = word_3 + word_3[::-1]

# ----------------------------------------------------------------------------
m = 12  # 8

k = 4
all_orders = list(permutations(word_1, k))

counts = [score_orders2(permute_letters(word_3, p), k) for p in tqdm(all_perms)]
target = m * sum(counts[0].values()) // len(all_orders)

xs = [
    pulp.LpVariable("x%i" % i, lowBound=0, cat="Integer")
    for i in tqdm(range(len(all_perms)))
]
prob = pulp.LpProblem("myProblem", pulp.LpMaximize)
prob += pulp.lpSum(xs) == m  # This is the objective!
for order in tqdm(all_orders):
    prob += pulp.lpSum([x * ct[order] for x, ct in zip(xs, counts)]) <= target

prob += xs[0] >= 1

status = prob.solve()

print("Values of variables")
for p, x in zip(all_perms, xs):
    print("%s: %s" % (p, pulp.value(x)))
print()
print("Values of Constraints")
for i, key in enumerate(prob.constraints.keys()):
    print("C%s: %s" % (i, pulp.value(prob.constraints[key])))

used_perms = [p for p, x in zip(all_perms, xs) if pulp.value(x) == 1.0]
word_4 = "".join([permute_letters(word_3, p) for p in used_perms])


# ----------------------------------------------------------------------------
m = 12

k = 5
all_perms = list(permutations(range(n)))
all_orders = list(permutations(word_1, k))

# Permute the 4/n word directly
counts = [score_orders2(permute_letters(word_4, p), k) for p in all_perms]

target = m * sum(counts[0].values()) // len(all_orders)

xs = [
    pulp.LpVariable("x%i" % i, lowBound=0, cat="Integer") for i, _ in enumerate(counts)
]
prob = pulp.LpProblem("myProblem", pulp.LpMaximize)
prob += pulp.lpSum(xs) == m  # This is the objective!
for order in tqdm(all_orders):
    prob += pulp.lpSum([x * ct[order] for x, ct in zip(xs, counts)]) <= target

prob += xs[0] >= 1

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

# ============================================================================

letters = "abcde"
n = len(letters)
word_1 = letters
k = 4
# m = 12
word_2 = word_1 + word_1[::-1]
all_perms = list(permutations(range(n)))
all_orders = list(permutations(letters, k))

counts = [score_orders2(permute_letters(word_2, p), k) for p in tqdm(all_perms)]
ncounts = [normalize_score(ct) for ct in counts]

target = 0.0

xs = [
    pulp.LpVariable("x%i" % i, lowBound=0, cat="Integer")  # upBound=m, cat="Integer")
    for i in tqdm(range(len(all_perms)))
]

prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
prob += pulp.lpSum(xs)  # This is the objective!
for order in tqdm(all_orders):
    prob += pulp.lpSum([x * ct[order] for x, ct in zip(xs, ncounts)]) == target
prob += xs[0] >= 1

status = prob.solve()

print("Values of variables")
for p, x in zip(all_perms, xs):
    print("%s: %s" % (p, pulp.value(x)))
print()
print("Values of Constraints")
for i, key in enumerate(prob.constraints.keys()):
    print("C%s: %s" % (i, pulp.value(prob.constraints[key])))

used_perms = [p for p, x in zip(all_perms, xs) if pulp.value(x) == 1.0]
segments_3 = [permute_letters(word_2, p) for p in used_perms]
word_3 = "".join(segments_3)
word_4 = word_3 + word_3[::-1]

# ----------------------------------------------------------------------------
k = 4
# m = 12
all_perms = list(permutations(range(n)))
all_orders = list(permutations(word_1, k))

counts = [score_orders2(permute_letters(word_3, p), k) for p in tqdm(all_perms)]
ncounts = [normalize_score(ct) for ct in counts]

target = 0.0

xs = [
    pulp.LpVariable("x%i" % i, lowBound=0, cat="Integer")  # upBound=m, cat="Integer")
    for i in tqdm(range(len(all_perms)))
]

prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
prob += pulp.lpSum(xs)  # This is the objective!
for order in tqdm(all_orders):
    prob += pulp.lpSum([x * ct[order] for x, ct in zip(xs, ncounts)]) == target
prob += xs[0] >= 1

status = prob.solve()

print("Values of variables")
for p, x in zip(all_perms, xs):
    print("%s: %s" % (p, pulp.value(x)))
print()
print("Values of Constraints")
for i, key in enumerate(prob.constraints.keys()):
    print("C%s: %s" % (i, pulp.value(prob.constraints[key])))

used_perms = [p for p, x in zip(all_perms, xs) if pulp.value(x) == 1.0]
word_4 = "".join([permute_letters(word_3, p) for p in used_perms])


# ----------------------------------------------------------------------------
k = 5

all_orders = list(permutations(letters, k))

counts = [score_orders2(permute_letters(word_4, p), k) for p in tqdm(all_perms)]
ncounts = [normalize_score(ct) for ct in counts]

target = 0.0

xs = [
    pulp.LpVariable("x%i" % i, lowBound=0, cat="Integer")  # upBound=m, cat="Integer")
    for i in tqdm(range(len(all_perms)))
]

prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
prob += pulp.lpSum(xs)  # This is the objective!
for order in tqdm(all_orders):
    prob += pulp.lpSum([x * ct[order] for x, ct in zip(xs, ncounts)]) == target
prob += xs[0] >= 1

status = prob.solve()

print("Values of variables")
for p, x in zip(all_perms, xs):
    print("%s: %s" % (p, pulp.value(x)))
print()
print("Values of Constraints")
for i, key in enumerate(prob.constraints.keys()):
    print("C%s: %s" % (i, pulp.value(prob.constraints[key])))

used_perms = [p for p, x in zip(all_perms, xs) if pulp.value(x) == 1.0]
word_3 = "".join([permute_letters(word_3, p) for p in used_perms])


# ----------------------------------------------------------------------------
letters = "abcdef"
n = len(letters)
word_1 = letters

k = 5
all_perms = list(permutations(range(n)))
all_orders = list(permutations(letters, k))

# Permute the 4/n word directly
counts = [score_orders2(permute_letters(word_4, p), k) for p in all_perms]
ncounts = [normalize_score(ct) for ct in counts]

target = 0.0

xs = [
    pulp.LpVariable("x%i" % i, lowBound=0, cat="Integer")
    for i in tqdm(range(len(all_perms)))
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
