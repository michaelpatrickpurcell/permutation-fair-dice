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


def build_grime_bounds_clauses(i, m, n, vpool):
    if n > 5:
        raise ValueError()
    grime_bounds = [
        m // 2,
        m * (3 * m + 2) // 12,
        m ** 2 * (m + 1) // 6,
        m * (5 * m ** 2 * (3 * m + 4) - 4) // 120,
    ]
    row = ["x_%i_%i" % (i, j) for j in range(m)]
    grime_bounds_clauses = []
    for k, bound in enumerate(grime_bounds[: (i + 1)]):
        # lits = [var_dict[x] for x in row]
        lits = [vpool.id(x) for x in row]
        weights = [j ** k for j, x in enumerate(row, 1)]
        cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, vpool=vpool)
        grime_bounds_clauses += cnf.clauses

    return grime_bounds_clauses


# ============================================================================


def build_gofirst_clauses(i, m, n, vpool):
    if n > 5:
        raise ValueError()

    gofirst_bound = m ** n // n

    lits = []
    weights = []

    reps = (n - i) - 1
    for bits in product([0, 1], repeat=reps):
        for j in range(m):
            add_indicator_variable(
                "y_%i_%s_%i" % ((i,) + (str(bits),) + (j,)),
                # ["x_%i_%i" % (i, j), "x_%i_%i" % (i + 1, j)],
                ["x_%i_%i" % (i + k, j) for k in range(reps)],
                bits,
                vpool,
                clauses,
            )
        lits += [
            vpool.id("y_%i_%s_%i" % ((i,) + (str(bits),) + (j,))) for j in range(m)
        ]

        u = bits[0]
        v = sum(bits[1:])
        if u == 0:
            weights += [
                j ** (i + 1) * j ** v * (j + 1) ** (((n - i) - 2) - v) for j in range(m)
            ]
        else:
            weights += [
                (j + 1) ** (i + 1) * j ** v * (j + 1) ** (((n - i) - 2) - v)
                for j in range(m)
            ]

    cnf = PBEnc.equals(lits=lits, weights=weights, bound=gofirst_bound, vpool=vpool)
    gofirst_clauses = cnf.clauses

    return gofirst_clauses


# ============================================================================

n = 5
m = 30

vpool = IDPool()

var_names = [["x_%i_%i" % (i, j) for j in range(m)] for i in range(n - 1)]
var_dict = dict()
for index, var_name in enumerate(sum(var_names, []), 1):
    vpool.id(var_name)

var_dict = {v: vpool.id(v) for v in sum(var_names, [])}

clauses = []
for i in range(2, n + 1):
    clauses += build_grime_bounds_clauses(n - i, m, n, vpool)
    clauses += build_gofirst_clauses(n - i, m, n, vpool)

# cnf = pysat.formula.CNF(from_clauses(clauses))
# cnf.to_file(filename)

sat = Minicard()
for clause in clauses:
    sat.add_clause(clause)

start_time = time.time()
solutions = []
is_solvable = True
while is_solvable:
    is_solvable = sat.solve()
    if is_solvable:
        sat_solution = sat.get_model()[: m * (n - 1)]
        print("Found a solution")
        bits = (np.array(sat_solution) > 0).astype(int)
        array = bits_to_array(bits, m)
        letters = string.ascii_lowercase[:n]
        word = array_to_word(array, letters)

        print(is_gofirst_fair(word))
        print(is_place_fair(word))
        print(is_permutation_fair(word))
        solutions.append(sat_solution)
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
