import numpy as np

from itertools import permutations, product
from pysat.solvers import Minicard
from pysat.pb import PBEnc
from pysat.formula import IDPool

import string
import time

from tqdm import tqdm

from utils import *

# ============================================================================


def add_indicator_variable(var_name, source_variable_names, values, vpool, clauses):
    new_variable = vpool.id(var_name)
    source_variables = [
        vpool.id(x) if v == 1 else -vpool.id(x)
        for x, v in zip(source_variable_names, values)
    ]

    forward_clause = [-x for x in source_variables] + [new_variable]
    backward_clauses = [[-new_variable, x] for x in source_variables]

    new_clauses = forward_clause, backward_clauses
    clauses += [forward_clause] + backward_clauses


# ============================================================================


def gofirst_search(n, m, level, c=None):
    vpool = IDPool()
    xs = [vpool.id("{x}".format(x=i)) for i in range(m)]

    l = n - (level - 1)

    if c is None:
        c = np.zeros(m, dtype=int)

    clauses = []

    lits = [-x for x in xs] + xs

    r = np.arange(m, dtype=int)
    zero_weights = (r ** (n - l)) * (r ** c) * ((r + 1) ** ((l - 1) - c))
    one_weights = ((r + 1) ** (n - l)) * (r ** c) * ((r + 1) ** ((l - 1) - c))
    weights = list(zero_weights) + list(one_weights)

    bound = m ** n // n

    cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
    clauses += cnf.clauses

    bounds = [
        m // 2,
        m * (3 * m + 2) // 12,
        m ** 2 * (m + 1) // 6,
        m * (5 * m ** 2 * (3 * m + 4) - 4) // 120,
    ]
    for i, bound in enumerate(bounds[: (level - 1)]):
        lits = xs
        weights = [j ** i for j, x in enumerate(xs, 1)]
        cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
        clauses += cnf.clauses

    sat = Minicard()
    for clause in clauses:
        sat.add_clause(clause)

    is_solvable = True
    while is_solvable:
        is_solvable = sat.solve()
        if is_solvable:
            # print("Found a solution.")
            sat_solution = sat.get_model()[:m]
            yield sat_solution
            elimination_clause = [-x for x in sat_solution]
            sat.add_clause(elimination_clause)


# ============================================================================

n = 5
m = 60

start_time = time.time()

flag = 0
solutions = []
gofirst_hits = []
place_hits = []
permutation_hits = []
c5 = np.zeros(m, dtype=int)
level_5_generator = gofirst_search(n, m, 5, c5)
for solution5 in level_5_generator:
    c4 = c5 + (np.array(solution5) > 0).astype(int)
    level_4_generator = gofirst_search(n, m, 4, c4)
    for solution4 in level_4_generator:
        c3 = c4 + (np.array(solution4) > 0).astype(int)
        level_3_generator = gofirst_search(n, m, 3, c3)
        for solution3 in level_3_generator:
            c2 = c3 + (np.array(solution3) > 0).astype(int)
            level_2_generator = gofirst_search(n, m, 2, c2)
            for solution2 in level_2_generator:
                solution = solution2 + solution3 + solution4 + solution5
                solutions.append(solution)
                print("search hit %i" % len(solutions))
                bits = (np.array(solution) > 0).astype(int)
                array = bits_to_array(bits, m)
                letters = string.ascii_lowercase[:n]
                word = array_to_word(array, letters)
                if is_gofirst_fair(word):
                    gofirst_hits.append(solution)
                    print("  gofirst hit %i" % len(gofirst_hits))
                if is_place_fair(word):
                    place_hits.append(solution)
                    print("  place hit %i" % len(place_hits))
                    flag = 1
                if is_permutation_fair(word):
                    permutation_hits.append(solution)
                    print("  permutation hit %i" % len(permutation_hits))
                if flag:
                    break
            if flag:
                break
        if flag:
            break
    if flag:
        break


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
    if is_gofirst_fair(word):
        gofirst_hits.append(word)
    if is_place_fair(word):
        place_hits.append(word)
    if is_permutation_fair(word):
        permutation_hits.append(word)

print(len(gofirst_hits))
print(len(place_hits))
print(len(permutation_hits))

# ============================================================================


n = 4
m = 12

vpool = IDPool()

var_names = [["x_%i_%i" % (i, j) for j in range(m)] for i in range(n - 1)]
var_dict = dict()
for index, var_name in enumerate(sum(var_names, []), 1):
    vpool.id(var_name)

var_dict = {v: vpool.id(v) for v in sum(var_names, [])}

clauses = []
# bounds = [
#     m // 2,
#     m * (3 * m + 2) // 12,
#     m ** 2 * (m + 1) // 6,
#     m * (5 * m ** 2 * (3 * m + 4) - 4) // 120,
# ]
# for i, bound in enumerate(bounds):
#     for row in var_names[i:]:
#         lits = [var_dict[x] for x in row]
#         weights = [j ** i for j, x in enumerate(row, 1)]
#         cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
#         clauses += cnf.clauses

bound = m ** n // n

ys = [
    add_indicator_variable("y_%i" % i, ["x_%i_%i" % (n - 2, i)], [0], vpool, clauses)
    for i in range(m)
]
lits = [vpool.id("y_%i" % i) for i in range(m)] + [var_dict[x] for x in var_names[-1]]
weights = [i ** (n - 1) for i in range(m)] + [(i + 1) ** (n - 1) for i in range(m)]
cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
clauses += cnf.clauses

lits = []
weights = []
for u, v in product([0, 1], repeat=2):
    for i in range(m):
        add_indicator_variable(
            "x_%i%i_%i%i_%i" % (n - 3, n - 2, u, v, i),
            ["x_%i_%i" % (n - 3, i), "x_%i_%i" % (n - 2, i)],
            [u, v],
            vpool,
            clauses,
        )
    lits += [vpool.id("x_%i%i_%i%i_%i" % (n - 3, n - 2, u, v, i)) for i in range(m)]
    # if (u, v) == (0, 0):
    #     weights += [i ** (n - 2) * (i + 1) for i in range(m)]
    # elif (u, v) == (0, 1):
    #     weights += [i ** (n - 1) for i in range(m)]
    # elif (u, v) == (1, 0):
    #     weights += [(i + 1) ** (n - 1) for i in range(m)]
    # elif (u, v) == (1, 1):
    #     weights += [i * (i + 1) ** (n - 2) for i in range(m)]

    if u == 0:
        weights += [i ** (n - 2) * i ** v * (i + 1) ** (1 - v) for i in range(m)]
    else:
        weights += [(i + 1) ** (n - 2) * i ** v * (i + 1) ** (1 - v) for i in range(m)]

cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
clauses += cnf.clauses

lits = []
weights = []
for u, v, w in product([0, 1], repeat=3):
    for i in range(m):
        add_indicator_variable(
            "x_%i%i%i_%i%i%i_%i" % (n - 4, n - 3, n - 2, u, v, w, i),
            ["x_%i_%i" % (n - 4, i), "x_%i_%i" % (n - 3, i), "x_%i_%i" % (n - 2, i)],
            [u, v, w],
            vpool,
            clauses,
        )
    lits += [
        vpool.id("x_%i%i%i_%i%i%i_%i" % (n - 4, n - 3, n - 2, u, v, w, i))
        for i in range(m)
    ]
    if u == 0:
        weights += [
            i ** (n - 3) * i ** (v + w) * (i + 1) ** (2 - (v + w)) for i in range(m)
        ]
    else:
        weights += [
            (i + 1) ** (n - 3) * i ** (v + w) * (i + 1) ** (2 - (v + w))
            for i in range(m)
        ]

cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
clauses += cnf.clauses

lits = []
weights = []
for u, v, w, x in product([0, 1], repeat=4):
    for i in range(m):
        add_indicator_variable(
            "x_%i%i%i%i_%i%i%i%i_%i" % (n - 5, n - 4, n - 3, n - 2, u, v, w, x, i),
            [
                "x_%i_%i" % (n - 5, i),
                "x_%i_%i" % (n - 4, i),
                "x_%i_%i" % (n - 3, i),
                "x_%i_%i" % (n - 2, i),
            ],
            [u, v, w, x],
            vpool,
            clauses,
        )
    lits += [
        vpool.id("x_%i%i%i%i_%i%i%i%i_%i" % (n - 5, n - 4, n - 3, n - 2, u, v, w, x, i))
        for i in range(m)
    ]
    if u == 0:
        weights += [
            i ** (n - 4) * i ** (v + w + x) * (i + 1) ** (3 - (v + w + x))
            for i in range(m)
        ]
    else:
        weights += [
            (i + 1) ** (n - 4) * i ** (v + w + x) * (i + 1) ** (3 - (v + w + x))
            for i in range(m)
        ]

cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
clauses += cnf.clauses

# if n == 5:
#     print("Adding extra clauses")
#     row = var_names[0]
#     lits = [var_dict[x] for x in row]
#     weights = [j for j, x in enumerate(row, 1)]
#     bound = 915
#     cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
#     clauses += cnf.clauses
#
#     for j, k in combinations([1, 2, 3], 2):
#         for i in range(m):  # loop to build cardinality constraints for case 11
#             x = vpool.id("x_%i_%i" % (j, i))
#             y = vpool.id("x_%i_%i" % (k, i))
#
#             var_name = "x_%i%i_11_%i" % (j, k, i)
#             vpool.id(var_name)
#             var_dict[var_name] = vpool.id(var_name)
#             z = vpool.id(var_name)
#             clauses += [[-x, -y, z], [x, -z], [y, -z]]  # (x & y) <-> z
#
#             var_name = "x_%i%i_00_%i" % (j, k, i)
#             vpool.id(var_name)
#             var_dict[var_name] = vpool.id(var_name)
#             z = vpool.id(var_name)
#             clauses += [[x, y, z], [-x, -z], [-y, -z]]  # (-x & -y) <-> z
#
#             var_name = "x_%i%i_01_%i" % (j, k, i)
#             vpool.id(var_name)
#             var_dict[var_name] = vpool.id(var_name)
#             z = vpool.id(var_name)
#             clauses += [[x, -y, z], [-x, -z], [y, -z]]  # (-x & y) <-> z
#
#             var_name = "x_%i%i_10_%i" % (j, k, i)
#             vpool.id(var_name)
#             var_dict[var_name] = vpool.id(var_name)
#             z = vpool.id(var_name)
#             clauses += [[-x, y, z], [x, -z], [-y, -z]]  # (x & -y) <-> z
#
#         lits = [var_dict["x_%i%i_11_%i" % (j, k, i)] for i in range(m)]
#         weights = [1 for i in range(m)]
#         bound = m // 6
#         cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
#         clauses += cnf.clauses
#
#         lits = [var_dict["x_%i%i_00_%i" % (j, k, i)] for i in range(m)]
#         weights = [1 for i in range(m)]
#         bound = m // 6
#         cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
#         clauses += cnf.clauses
#
#         lits = [var_dict["x_%i%i_01_%i" % (j, k, i)] for i in range(m)]
#         weights = [1 for i in range(m)]
#         bound = m // 3
#         cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
#         clauses += cnf.clauses
#
#         lits = [var_dict["x_%i%i_10_%i" % (j, k, i)] for i in range(m)]
#         weights = [1 for i in range(m)]
#         bound = m // 3
#         cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
#         clauses += cnf.clauses
#
#     for k in [1, 2, 3]:
#         for i in range(m):
#             x = vpool.id("x_0_%i" % i)
#             y = vpool.id("x_%i_%i" % (k, i))
#
#             var_name = "x_0%i_11_%i" % (k, i)
#             vpool.id(var_name)
#             var_dict[var_name] = vpool.id(var_name)
#             z = vpool.id(var_name)
#             clauses += [[-x, -y, z], [x, -z], [y, -z]]  # (x & y) <-> z
#
#             var_name = "x_0%i_00_%i" % (k, i)
#             vpool.id(var_name)
#             var_dict[var_name] = vpool.id(var_name)
#             z = vpool.id(var_name)
#             clauses += [[x, y, z], [-x, -z], [-y, -z]]  # (x & y) <-> z
#
#             var_name = "x_0%i_01_%i" % (k, i)
#             vpool.id(var_name)
#             var_dict[var_name] = vpool.id(var_name)
#             z = vpool.id(var_name)
#             clauses += [[x, -y, z], [-x, -z], [y, -z]]  # (x & y) <-> z
#
#             var_name = "x_0%i_10_%i" % (k, i)
#             vpool.id(var_name)
#             var_dict[var_name] = vpool.id(var_name)
#             z = vpool.id(var_name)
#             clauses += [[-x, y, z], [x, -z], [-y, -z]]  # (x & y) <-> z
#
#         lits = [var_dict["x_0%i_11_%i" % (k, i)] for i in range(m)]
#         weights = [1 for i in range(m)]
#         bound = m // 4
#         cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
#         clauses += cnf.clauses
#
#         lits = [var_dict["x_0%i_00_%i" % (k, i)] for i in range(m)]
#         weights = [1 for i in range(m)]
#         bound = m // 4
#         cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
#         clauses += cnf.clauses
#
#         lits = [var_dict["x_0%i_01_%i" % (k, i)] for i in range(m)]
#         weights = [1 for i in range(m)]
#         bound = m // 4
#         cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
#         clauses += cnf.clauses
#
#         lits = [var_dict["x_0%i_10_%i" % (k, i)] for i in range(m)]
#         weights = [1 for i in range(m)]
#         bound = m // 4
#         cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
#         clauses += cnf.clauses


sat = Minicard()
for clause in clauses:
    sat.add_clause(clause)

start_time = time.time()

solutions = []
is_solvable = True
while is_solvable:
    if len(solutions) % 10000 == 0:
        print("%i solutions discovered so far." % len(solutions))
    is_solvable = sat.solve()
    if is_solvable:
        # print("Found a solution.")
        sat_solution = sat.get_model()[: m * (n - 1)]
        solutions.append(sat_solution)

        # bits = (np.array(sat_solution) > 0).astype(int)
        # array = bits_to_array(bits, m)
        # letters = string.ascii_lowercase[:n]
        # word = array_to_word(array, letters)
        # # if is_gofirst_fair(word): gofirst_hits.append(word)
        # if is_place_fair(word):
        #     print("Found a set of dice that is place fair.")
        #     break
        # # if is_permutation_fair(word): permutation_hits.append(word)

        elimination_clause = [-x for x in sat_solution]
        sat.add_clause(elimination_clause)

    else:
        print("Found %i solutions." % len(solutions))

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
    if is_gofirst_fair(word):
        gofirst_hits.append(word)
    if is_place_fair(word):
        place_hits.append(word)
    if is_permutation_fair(word):
        permutation_hits.append(word)

print(len(gofirst_hits))
print(len(place_hits))
print(len(permutation_hits))

# ============================================================================


dice_60 = {
    "a": [
        2,
        8,
        12,
        18,
        24,
        29,
        32,
        38,
        44,
        49,
        53,
        59,
        63,
        67,
        73,
        78,
        83,
        88,
        92,
        98,
        103,
        109,
        113,
        117,
        122,
        127,
        133,
        138,
        143,
        148,
        153,
        159,
        164,
        167,
        173,
        178,
        183,
        188,
        194,
        199,
        202,
        208,
        214,
        217,
        224,
        227,
        233,
        238,
        243,
        248,
        253,
        257,
        263,
        269,
        272,
        278,
        284,
        289,
        292,
        298,
    ],
    "b": [
        3,
        7,
        13,
        19,
        23,
        28,
        33,
        37,
        43,
        48,
        52,
        58,
        64,
        68,
        74,
        79,
        84,
        87,
        93,
        97,
        104,
        108,
        112,
        118,
        123,
        128,
        134,
        137,
        142,
        149,
        154,
        158,
        163,
        168,
        172,
        177,
        184,
        187,
        193,
        198,
        203,
        207,
        213,
        218,
        223,
        228,
        234,
        239,
        242,
        249,
        252,
        258,
        264,
        268,
        273,
        279,
        283,
        288,
        293,
        297,
    ],
    "c": [
        4,
        9,
        11,
        20,
        22,
        27,
        31,
        39,
        42,
        47,
        51,
        60,
        65,
        69,
        75,
        77,
        82,
        89,
        91,
        99,
        102,
        110,
        114,
        119,
        124,
        129,
        132,
        136,
        144,
        147,
        152,
        157,
        162,
        169,
        171,
        176,
        182,
        189,
        192,
        200,
        204,
        209,
        215,
        219,
        222,
        226,
        235,
        237,
        244,
        247,
        254,
        259,
        262,
        267,
        271,
        280,
        282,
        290,
        294,
        296,
    ],
    "d": [
        5,
        6,
        14,
        17,
        21,
        30,
        34,
        40,
        41,
        50,
        54,
        57,
        62,
        66,
        72,
        76,
        85,
        86,
        94,
        100,
        101,
        107,
        111,
        116,
        125,
        130,
        135,
        139,
        145,
        146,
        155,
        156,
        165,
        166,
        174,
        179,
        185,
        190,
        191,
        197,
        201,
        210,
        212,
        216,
        225,
        229,
        232,
        236,
        241,
        250,
        251,
        260,
        261,
        270,
        274,
        277,
        281,
        287,
        295,
        299,
    ],
    "e": [
        1,
        10,
        15,
        16,
        25,
        26,
        35,
        36,
        45,
        46,
        55,
        56,
        61,
        70,
        71,
        80,
        81,
        90,
        95,
        96,
        105,
        106,
        115,
        120,
        121,
        126,
        131,
        140,
        141,
        150,
        151,
        160,
        161,
        170,
        175,
        180,
        181,
        186,
        195,
        196,
        205,
        206,
        211,
        220,
        221,
        230,
        231,
        240,
        245,
        246,
        255,
        256,
        265,
        266,
        275,
        276,
        285,
        286,
        291,
        300,
    ],
}

word_60 = dice_to_word(dice_60)
print(is_gofirst_fair(word_60))
print(is_place_fair(word_60))
print(is_permutation_fair(word_60))
