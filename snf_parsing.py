import numpy as np
import pickle

# ============================================================================

with open("snf_u.txt", "r") as f:
    data = f.read()

lines = data.strip("\n[]").replace("\n", " ").split("] [")
rows = []
for line in lines:
    row = [int(r) for r in line.split()]
    rows.append(row)

U = np.matrix(rows)
with open('snf_u.pickle', 'wb') as handle:
    pickle.dump(U, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ============================================================================

with open("snf_n.txt", "r") as f:
    lines = f.readlines()

rows = []
for line in lines:
    row = [int(r) for r in line.strip("[]\n").split()]
    rows.append(row)

N = np.matrix(rows)
with open('snf_n.pickle', 'wb') as handle:
    pickle.dump(N, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ============================================================================

with open("snf_v.txt", "r") as f:
    lines = f.readlines()

rows = []
for line in lines:
    row = [int(r) for r in line.strip("[]\n").split()]
    rows.append(row)

V = np.matrix(rows)
with open('snf_v.pickle', 'wb') as handle:
    pickle.dump(V, handle, protocol=pickle.HIGHEST_PROTOCOL)
