import numpy as np

from itertools import combinations, permutations, product

import smithnormalform as snf

from utils import rotl, insert

# ============================================================================

lut = {k: i for i, k in enumerate(permutations("abcd"))}
M_list = []
for x in permutations("abcd"):
    pairs = [(lut[rotl(x, i)], lut[rotl(x, i)[::-1]]) for i in range(4)]
    for ys in product(*pairs):
        indices = np.array(ys)
        row = np.zeros(24)
        row[indices] = 1
        M_list.append(row)
    pairs = [
        (lut[insert(x[1:], i, x[0])], lut[insert(x[1:], i, x[0])[::-1]])
        for i in range(4)
    ]
    for ys in product(*pairs):
        indices = np.array(ys)
        row = np.zeros(24)
        row[indices] = 1
        M_list.append(row)
    # M_list.append(np.ones(24))

M = np.matrix(M_list).astype(np.int)
rank = np.linalg.matrix_rank(M)
print(rank)

# N_list = []
# current_rank = 0
# for row in M_list:
#     temp = np.matrix(N_list + [row])
#     if np.linalg.matrix_rank(temp) > current_rank:
#         N_list.append(row)
#         current_rank += 1
#
# N = np.matrix(N_list)
N = np.matrix(M_list)


# ZM = [snf.z.Z(int(N[i,j])) for i in range(N.shape[0]) for j in range(N.shape[1])]
# original_matrix = snf.matrix.Matrix(N.shape[0], N.shape[1], ZM)
# prob = snf.snfproblem.SNFProblem(original_matrix)
# prob.computeSNF()
#
# S = np.matrix([x.a for x in prob.S.elements]).reshape((23,23))
# A = np.matrix([x.a for x in prob.A.elements]).reshape((23,24))
# T = np.matrix([x.a for x in prob.T.elements]).reshape((24,24))
# J = np.matrix([x.a for x in prob.J.elements]).reshape((23,24))



with open("snf_n.pickle", "rb") as handle:
    J = pickle.load(handle)

with open("snf_u.pickle", "rb") as handle:
    S = pickle.load(handle)

with open("snf_v.pickle", "rb") as handle:
    T = pickle.load(handle)


d = 12
C = (d**4 // 6) * np.ones(N.shape[0], dtype=np.int)

D = (S @ C)
E = np.array((D[0,:23] / np.diag(J)[:23]).tolist()[0] + [d**4 // 24])
X = T @ E

for i in range(-2**10, 2**10):
    E = np.array((D[0,:23] / np.diag(J)[:23]).tolist()[0] + [i])
    X = T @ E
    if np.all(X >= 0):
        print(i)
