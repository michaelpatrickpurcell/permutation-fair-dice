import numpy as np

from itertools import permutations, product
from pysat.solvers import Minicard
from pysat.pb import PBEnc
from pysat.formula import IDPool

import string

from tqdm import tqdm

from utils import *

# ============================================================================

n = 5
m = 60

vpool = IDPool()

var_names = [['x_%i_%i' % (i,j) for j in range(m)] for i in range(n-1)]
var_dict = dict()
for index,var_name in enumerate(sum(var_names, []), 1):
    vpool.id(var_name)

var_dict = {v: vpool.id(v) for v in sum(var_names, [])}

clauses = []
bounds = [m // 2, m * (3*m + 2) // 12, m**2 * (m+1) // 6, m * (5*m**2 *(3*m+4) - 4) // 120]
for i,bound in enumerate(bounds):
    for row in var_names[i:]:
        lits = [var_dict[x] for x in row]
        weights = [j**i for j,x in enumerate(row,1)]
        cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
        clauses += cnf.clauses

if n == 5:
    print('Adding extra clauses')
    row = var_names[0]
    lits = [var_dict[x] for x in row]
    weights = [j for j,x in enumerate(row,1)]
    bound = 915
    cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
    clauses += cnf.clauses


    for j,k in combinations([1,2,3], 2):
        for i in range(m): # loop to build cardinality constraints for case 11
            x = vpool.id('x_%i_%i' % (j,i))
            y = vpool.id('x_%i_%i' % (k,i))

            var_name = 'x_%i%i_11_%i' % (j,k,i)
            vpool.id(var_name)
            var_dict[var_name] = vpool.id(var_name)
            z = vpool.id(var_name)
            clauses += [[-x, -y, z], [x, -z], [y, -z]] # (x & y) <-> z

            var_name = 'x_%i%i_00_%i' % (j,k,i)
            vpool.id(var_name)
            var_dict[var_name] = vpool.id(var_name)
            z = vpool.id(var_name)
            clauses += [[x, y, z], [-x, -z], [-y, -z]] # (-x & -y) <-> z

            var_name = 'x_%i%i_01_%i' % (j,k,i)
            vpool.id(var_name)
            var_dict[var_name] = vpool.id(var_name)
            z = vpool.id(var_name)
            clauses += [[x, -y, z], [-x, -z], [y, -z]] # (-x & y) <-> z

            var_name = 'x_%i%i_10_%i' % (j,k,i)
            vpool.id(var_name)
            var_dict[var_name] = vpool.id(var_name)
            z = vpool.id(var_name)
            clauses += [[-x, y, z], [x, -z], [-y, -z]] # (x & -y) <-> z

        lits = [var_dict['x_%i%i_11_%i' % (j,k,i)] for i in range(m)]
        weights = [1 for i in range(m)]
        bound = m // 6
        cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
        clauses += cnf.clauses

        lits = [var_dict['x_%i%i_00_%i' % (j,k,i)] for i in range(m)]
        weights = [1 for i in range(m)]
        bound = m // 6
        cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
        clauses += cnf.clauses

        lits = [var_dict['x_%i%i_01_%i' % (j,k,i)] for i in range(m)]
        weights = [1 for i in range(m)]
        bound = m // 3
        cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
        clauses += cnf.clauses

        lits = [var_dict['x_%i%i_10_%i' % (j,k,i)] for i in range(m)]
        weights = [1 for i in range(m)]
        bound = m // 3
        cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
        clauses += cnf.clauses

    for k in [1,2,3]:
        for i in range(m):
            x = vpool.id('x_0_%i' % i)
            y = vpool.id('x_%i_%i' % (k,i))

            var_name = 'x_0%i_11_%i' % (k,i)
            vpool.id(var_name)
            var_dict[var_name] = vpool.id(var_name)
            z = vpool.id(var_name)
            clauses += [[-x, -y, z], [x, -z], [y, -z]] # (x & y) <-> z

            var_name = 'x_0%i_00_%i' % (k,i)
            vpool.id(var_name)
            var_dict[var_name] = vpool.id(var_name)
            z = vpool.id(var_name)
            clauses += [[x, y, z], [-x, -z], [-y, -z]] # (x & y) <-> z

            var_name = 'x_0%i_01_%i' % (k,i)
            vpool.id(var_name)
            var_dict[var_name] = vpool.id(var_name)
            z = vpool.id(var_name)
            clauses += [[x, -y, z], [-x, -z], [y, -z]] # (x & y) <-> z

            var_name = 'x_0%i_10_%i' % (k,i)
            vpool.id(var_name)
            var_dict[var_name] = vpool.id(var_name)
            z = vpool.id(var_name)
            clauses += [[-x, y, z], [x, -z], [-y, -z]] # (x & y) <-> z

        lits = [var_dict['x_0%i_11_%i' % (k,i)] for i in range(m)]
        weights = [1 for i in range(m)]
        bound = m // 4
        cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
        clauses += cnf.clauses

        lits = [var_dict['x_0%i_00_%i' % (k,i)] for i in range(m)]
        weights = [1 for i in range(m)]
        bound = m // 4
        cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
        clauses += cnf.clauses

        lits = [var_dict['x_0%i_01_%i' % (k,i)] for i in range(m)]
        weights = [1 for i in range(m)]
        bound = m // 4
        cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
        clauses += cnf.clauses

        lits = [var_dict['x_0%i_10_%i' % (k,i)] for i in range(m)]
        weights = [1 for i in range(m)]
        bound = m // 4
        cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
        clauses += cnf.clauses


sat = Minicard()
for clause in clauses:
    sat.add_clause(clause)

start_time = time.time()

solutions = []
is_solvable = True
while is_solvable:
    if len(solutions) % 1 == 0:
        print('%i solutions discovered so far.' % len(solutions))
    is_solvable = sat.solve()
    if is_solvable:
        print('Found a solution.')
        sat_solution = sat.get_model()[:m*(n-1)]
        solutions.append(sat_solution)

        bits = (np.array(sat_solution) > 0).astype(int)
        array = bits_to_array(bits, m)
        letters = string.ascii_lowercase[:n]
        word = array_to_word(array, letters)
        # if is_gofirst_fair(word): gofirst_hits.append(word)
        if is_place_fair(word):
            print('Found a set of dice that is place fair.')
            break
        # if is_permutation_fair(word): permutation_hits.append(word)

        elimination_clause = [-x for x in sat_solution]
        sat.add_clause(elimination_clause)

    else:
        print('Found %i solutions.' % len(solutions))

end_time = time.time()

print(end_time - start_time)


gofirst_hits = []
place_hits = []
permutation_hits = []
for solution in tqdm(solutions):
    bits = (np.array(solution) > 0).astype(int)
    array = bits_to_array(bits, m)
    letters = string.ascii_lowercase[:n]
    word = array_to_word(array, letters)
    if is_gofirst_fair(word): gofirst_hits.append(word)
    if is_place_fair(word): place_hits.append(word)
    if is_permutation_fair(word): permutation_hits.append(word)




dice_60 = {'a': [2, 8, 12, 18, 24, 29, 32, 38, 44, 49, 53, 59, 63, 67, 73, 78, 83, 88, 92, 98,
103, 109, 113, 117, 122, 127, 133, 138, 143, 148, 153, 159, 164, 167, 173, 178, 183, 188, 194, 199,
202, 208, 214, 217, 224, 227, 233, 238, 243, 248, 253, 257, 263, 269, 272, 278, 284, 289, 292, 298],
'b': [3, 7, 13, 19, 23, 28, 33, 37, 43, 48, 52, 58, 64, 68, 74, 79, 84, 87, 93, 97,
104, 108, 112, 118, 123, 128, 134, 137, 142, 149, 154, 158, 163, 168, 172, 177, 184, 187, 193, 198,
203, 207, 213, 218, 223, 228, 234, 239, 242, 249, 252, 258, 264, 268, 273, 279, 283, 288, 293, 297],
'c': [4, 9, 11, 20, 22, 27, 31, 39, 42, 47, 51, 60, 65, 69, 75, 77, 82, 89, 91, 99,
102, 110, 114, 119, 124, 129, 132, 136, 144, 147, 152, 157, 162, 169, 171, 176, 182, 189, 192, 200,
204, 209, 215, 219, 222, 226, 235, 237, 244, 247, 254, 259, 262, 267, 271, 280, 282, 290, 294, 296],
'd': [5, 6, 14, 17, 21, 30, 34, 40, 41, 50, 54, 57, 62, 66, 72, 76, 85, 86, 94, 100,
101, 107, 111, 116, 125, 130, 135, 139, 145, 146, 155, 156, 165, 166, 174, 179, 185, 190, 191, 197,
201, 210, 212, 216, 225, 229, 232, 236, 241, 250, 251, 260, 261, 270, 274, 277, 281, 287, 295, 299],
'e': [1, 10, 15, 16, 25, 26, 35, 36, 45, 46, 55, 56, 61, 70, 71, 80, 81, 90, 95, 96,
105, 106, 115, 120, 121, 126, 131, 140, 141, 150, 151, 160, 161, 170, 175, 180, 181, 186, 195, 196,
205, 206, 211, 220, 221, 230, 231, 240, 245, 246, 255, 256, 265, 266, 275, 276, 285, 286, 291, 300]}

word_60 = dice_to_word(dice_60)
print(is_gofirst_fair(word_60))
print(is_place_fair(word_60))
print(is_permutation_fair(word_60))
