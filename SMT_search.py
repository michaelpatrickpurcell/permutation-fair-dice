import numpy as np
from tqdm import tqdm

from pysmt.shortcuts import Symbol, Int, get_model
from pysmt.shortcuts import And, Or
from pysmt.shortcuts import GE, LE, Equals
from pysmt.shortcuts import Plus, Times
from pysmt.typing import INT

from itertools import combinations, permutations, product
from scipy.special import factorial
from utils import score_orders2


def smt_to_word(smt_solution, indicator, dice_names, d):
    n = len(dice_names)
    bit_array = []
    for i in range(n):
        bit_array.append(
            [int(smt_solution.get_py_value(indicator[i][jj])) for jj in range(n * d)]
        )
    char_list = ["" for i in range(n * d)]
    for x, row in zip(dice_names, bit_array):
        for i in range(n * d):
            if row[i] == 1:
                char_list[i] = x
    return "".join(char_list)


# ============================================================================

dice_names = "abcd"
n = len(dice_names)
d = 6
k = 3

row_lut = {(x,): i for i, x in enumerate(sorted(dice_names))}

indicator = [
    [Symbol(x + "%i_ind" % i, INT) for i in range(n * d)] for x in sorted(dice_names)
]
indicator_domains = And(
    [And([And(GE(x, Int(0)), LE(x, Int(1))) for x in indicator[i]]) for i in range(n)]
)

accumulator = [
    [Symbol(x + "%i_acc" % i, INT) for i in range(n * d)] for x in sorted(dice_names)
]
constraint = [
    [Equals(accumulator[i][j], Plus(indicator[i][: j + 1])) for j in range(n * d)]
    for i in range(n)
]

accumulators = [accumulator]
constraints = [constraint]
row_luts = [row_lut]

for m in range(2, k + 1):
    keys = sorted(list(permutations(sorted(dice_names), m)))
    row_lut = {x: i for i, x in enumerate(keys)}
    accumulator = []
    constraint = []
    for i, x in enumerate(keys):
        mask = indicator[row_luts[0][x[-1:]]]
        j = row_luts[-1][x[:-1]]
        temp = [accumulators[-1][j][jj] * mask[jj] for jj in range(n * d)]
        accumulator.append(
            [Symbol("".join(x) + "%i_acc" % jj, INT) for jj in range(n * d)]
        )
        constraint.append(
            [Equals(accumulator[i][jj], Plus(temp[: jj + 1])) for jj in range(n * d)]
        )
    accumulators.append(accumulator)
    constraints.append(constraint)
    row_luts.append(row_lut)


indicator_columns = [[indicator[i][jj] for i in range(n)] for jj in range(n * d)]
indicator_constraints = And(
    [Equals(Plus(indicator_columns[jj]), Int(1)) for jj in range(n * d)]
)
symmetry_constraints = Equals(indicator[0][0], Int(1))

target_constraints = []
for i in range(k):
    target_vars = [x[-1] for x in accumulators[i]]
    target_val = d ** (i + 1) // factorial((i + 1), exact=True)
    target_constraints.append(And([Equals(x, Int(target_val)) for x in target_vars]))
target_constraints = And(target_constraints)

problem_constraints = And(sum(sum(constraints, []), []))
formula = And(
    indicator_domains,
    indicator_constraints,
    problem_constraints,
    target_constraints,
    symmetry_constraints,
)

model = get_model(formula)
if model:
    print(model)
else:
    print("No solution found")

for i in range(n):
    print([model[indicator[i][jj]] for jj in range(n * d)])

test = smt_to_word(model, indicator, dice_names, d)
print(test)
score_orders2(test, k)

# ============================================================================

# n = 19
# # dice_names = ["D%i" % i for i in range(n)]
# dice_names = "abcdefghijklmnopqrs"

n = 19
dice_names = "abcdefghijklmnopqrs"  # tuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%"

dice_pairs = list(permutations(dice_names, 2))
d = 3

k = 2

row_lut = {(x,): i for i, x in enumerate(sorted(dice_names))}

indicator = [
    [Symbol(x + "%i_ind" % i, INT) for i in range(n * d)] for x in sorted(dice_names)
]
indicator_domains = And(
    [And([And(GE(x, Int(0)), LE(x, Int(1))) for x in indicator[i]]) for i in range(n)]
)

accumulator = [
    [Symbol(x + "%i_acc" % i, INT) for i in range(n * d)] for x in sorted(dice_names)
]
accumulator_domains = And(
    [And([And(GE(x, Int(0)), LE(x, Int(d))) for x in indicator[i]]) for i in range(n)]
)

constraint = [
    [Equals(accumulator[i][j], Plus(indicator[i][: j + 1])) for j in range(n * d)]
    for i in range(n)
]

accumulators = [accumulator]
constraints = [constraint]
row_luts = [row_lut]

indicator_columns = [[indicator[i][jj] for i in range(n)] for jj in range(n * d)]
indicator_constraints = And(
    [Equals(Plus(indicator_columns[jj]), Int(1)) for jj in range(n * d)]
)
symmetry_constraints = Equals(indicator[0][0], Int(1))

score = d ** 2 // 2 + 1
mask_index = sorted([x for x in set(np.arange(1, n) ** 2 % n)])
mask = [1 if (i + 1) in mask_index else 0 for i in range(n - 1)]
temp = [score if mask[i] else d ** 2 - score for i in range(n - 1)]
S = [[temp[(j - i) % (n - 1)] for j in range(n - 1)] for i in range(n)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, [])) if s == score}

target_constraints = []
target_vars = [x[-1] for x in accumulators[0]]
target_val = d
target_constraints.append(And([Equals(x, Int(target_val)) for x in target_vars]))
for key, target_val in scores.items():
    i, j = row_lut[key[:1]], row_lut[key[-1:]]
    target_constraints.append(
        GE(
            Plus([Times(accumulator[i][jj], indicator[j][jj]) for jj in range(n * d)]),
            Int(target_val),
        )
    )
target_constraints = And(target_constraints)


problem_constraints = And(sum(sum(constraints, []), []))
formula = And(
    indicator_domains,
    indicator_constraints,
    accumulator_domains,
    problem_constraints,
    target_constraints,
    symmetry_constraints,
)

model = get_model(formula)
if model:
    print(model)
else:
    print("No solution found")

for i in range(n):
    print([model[indicator[i][jj]] for jj in range(n * d)])

test = smt_to_word(model, indicator, dice_names, d)
print(test)
counts = score_orders2(test, k)
for s in scores:
    print(s, scores[s], counts[s])
