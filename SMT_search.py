from pysmt.shortcuts import Symbol, Int, get_model
from pysmt.shortcuts import And, Or
from pysmt.shortcuts import GE, LE, Equals
from pysmt.shortcuts import Plus, Times
from pysmt.typing import INT

from itertools import combinations, permutations, product
from scipy.special import factorial

def smt_to_word(smt_solution, indicator, dice_names, d):
    n = len(dice_names)
    bit_array = []
    for i in range(n):
        bit_array.append([int(smt_solution.get_py_value(indicator[i][jj])) for jj in range(n*d)])
    char_list = ['' for i in range(n*d)]
    for x,row in zip(dice_names, bit_array):
        for i in range(n*d):
            if row[i] == 1:
                char_list[i] = x
    return ''.join(char_list)



dice_names = "abcde"
n = len(dice_names)
d = 12
k = 3

row_lut = {(x,):i for i,x in enumerate(sorted(dice_names))}

indicator = [[Symbol(x + "%i_ind" % i, INT) for i in range(n*d)] for x in sorted(dice_names)]
indicator_domains = And([And([And(GE(x, Int(0)), LE(x, Int(1))) for x in indicator[i]]) for i in range(n)])

accumulator = [[Symbol(x + "%i_acc" % i, INT) for i in range(n*d)] for x in sorted(dice_names)]
constraint = [[Equals(accumulator[i][j], Plus(indicator[i][:j+1])) for j in range(n*d)] for i in range(n)]

accumulators = [accumulator]
constraints = [constraint]
row_luts = [row_lut]

for m in range(2,k+1):
    keys = sorted(list(permutations(sorted(dice_names), m)))
    row_lut = {x:i for i,x in enumerate(keys)}
    accumulator = []
    constraint = []
    for i,x in enumerate(keys):
        mask = indicator[row_luts[0][x[-1:]]]
        j = row_luts[-1][x[:-1]]
        temp = [accumulators[-1][j][jj] * mask[jj] for jj in range(n*d)]
        accumulator.append([Symbol(''.join(x) + "%i_acc" % jj, INT) for jj in range(n*d)])
        constraint.append([Equals(accumulator[i][jj], Plus(temp[:jj+1])) for jj in range(n*d)])
    accumulators.append(accumulator)
    constraints.append(constraint)
    row_luts.append(row_lut)


indicator_columns = [[indicator[i][jj] for i in range(n)] for jj in range(n*d)]
indicator_constraints = And([Equals(Plus(indicator_columns[jj]), Int(1)) for jj in range(n*d)])
symmetry_constraints = Equals(indicator[0][0], Int(1))

target_constraints = []
for i in range(k):
    target_vars = [x[-1] for x in accumulators[i]]
    target_val = d**(i+1) // factorial((i+1), exact=True)
    target_constraints.append(And([Equals(x, Int(target_val)) for x in target_vars]))
target_constraints = And(target_constraints)

problem_constraints = And(sum(sum(constraints,[]),[]))
formula = And(indicator_domains, indicator_constraints, problem_constraints, target_constraints, symmetry_constraints)

%time model = get_model(formula)
if model:
  print(model)
else:
  print("No solution found")

for i in range(n):
    print([model[indicator[i][jj]] for jj in range(n*d)])

test = smt_to_word(model, indicator, dice_names, d)
print(test)
score_orders2(test, k)
