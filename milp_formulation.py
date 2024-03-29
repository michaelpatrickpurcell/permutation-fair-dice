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


# ============================================================================

letters = "abc"
word2 = letters + letters[::-1]

word3 = milp_search_concatenate(word2, 3, solution_len=6, verbose=True)

# search_result = milp_search_concatenate(word2, 3, verbose=True)
# if search_results:
#     used_perms, segments_3 = search_results
#     word3 = "".join(segments_3)

word4 = word3 + word3[::-1]

word5 = milp_search_concatenate(word4, 5, solution_len=10, verbose=True)

# word4 = milp_search_concatenate(word3, 4, verbose=True)
# search_results = milp_search_concatenate(word3, 4, solution_len=10, verbose=True)
# if search_results:
#     used_perms, segments_4 = search_results
#     word4 = "".join(segments_4)

# word_4b = "".join([permute_letters(word_4, rotl((0, 1, 2, 3, 4), i)) for i in range(5)])
# segments_4b = ["".join(rotl(segments_4, i)) for i in range(6)]
# word_4b = "".join(segments_4b)

search_results = milp_search_concatenate(
    word_4, order_len=5, solution_len=15, verbose=True
)
if search_results:
    used_perms, segments_5 = search_results
    word_5 = "".join(segments_5)


# Results
# n = 3
# letters = "abc"
# word_2 = "abccba"
# word_3 = "abccbabaccabcabbac"

# n = 4
# letters = "abcd"
# word_2 = "abcddcba"
# word_3 = "adbccbdabdcaacdbcdabbadc"
# word_4 = word_3 + word_3[::-1]

# n = 5
# letters = "abcde"
# word_2 = letters + letters[:-1]
# word_3 = "abecddcebaadecbbcedacbdaeeadbccedbaabdecdbeaccaebdebdaccadbe"
# word_4 = word_3 + word_3[::-1]

# word2 = 6*'e' + gofirst + 6*'e' # 'eeeeeeabcddcbadbaccabdcbaddabccbaddabcdbaccabdabcddcbaeeeeee'
# word3 = milp_search_concatenate(word2, 3)
# word4 = word3 + word3[::-1]
# word_5 = milp_search_concatenate(word4, 5)
# word5 = 'eeeeeedabccbadcadbbdacbadccdabbadccdabcadbbdacdabccbadeeeeeebbbbbbecaddacedceaaecdaceddecaaceddecadceaaecdecaddacebbbbbbaaaaaabcdeedcbecbddbcedcbeebcddcbeebcdecbddbcebcdeedcbaaaaaaddddddbcaeeacbecbaabceacbeebcaacbeebcaecbaabcebcaeeacbddddddccccccdabeebadeadbbdaebadeedabbadeedabeadbbdaedabeebadccccccccccccdabeebadeadbbdaebadeedabbadeedabeadbbdaedabeebadccccccddddddbcaeeacbecbaabceacbeebcaacbeebcaecbaabcebcaeeacbddddddaaaaaabcdeedcbecbddbcedcbeebcddcbeebcdecbddbcebcdeedcbaaaaaabbbbbbecaddacedceaaecdaceddecaaceddecadceaaecdecaddacebbbbbbeeeeeedabccbadcadbbdacbadccdabbadccdabcadbbdacdabccbadeeeeeeccccccebdaadbeabeddebadbeaaebddbeaaebdabeddebaebdaadbeccccccddddddcabeebaceacbbcaebaceecabbaceecabeacbbcaecabeebacddddddbbbbbbdaecceadcadeedaceadccdaeeadccdaecadeedacdaecceadbbbbbbeeeeeedabccbadcadbbdacbadccdabbadccdabcadbbdacdabccbadeeeeeeaaaaaaebdccdbecbeddebcdbeccebddbeccebdcbeddebcebdccdbeaaaaaaaaaaaaebdccdbecbeddebcdbeccebddbeccebdcbeddebcebdccdbeaaaaaaeeeeeedabccbadcadbbdacbadccdabbadccdabcadbbdacdabccbadeeeeeebbbbbbdaecceadcadeedaceadccdaeeadccdaecadeedacdaecceadbbbbbbddddddcabeebaceacbbcaebaceecabbaceecabeacbbcaecabeebacddddddccccccebdaadbeabeddebadbeaaebddbeaaebdabeddebaebdaadbeccccccbbbbbbceaddaecdecaacedaecddceaaecddceadecaacedceaddaecbbbbbbaaaaaabdeccedbcdbeebdcedbccbdeedbccbdecdbeebdcbdeccedbaaaaaaeeeeeeadcbbcdabdaccadbcdabbadccdabbadcbdaccadbadcbbcdaeeeeeeccccccadebbedabdaeeadbedabbadeedabbadebdaeeadbadebbedaccccccddddddceabbaecbecaacebaecbbceaaecbbceabecaacebceabbaecddddddddddddceabbaecbecaacebaecbbceaaecbbceabecaacebceabbaecddddddccccccadebbedabdaeeadbedabbadeedabbadebdaeeadbadebbedacccccceeeeeeadcbbcdabdaccadbcdabbadccdabbadcbdaccadbadcbbcdaeeeeeeaaaaaabdeccedbcdbeebdcedbccbdeedbccbdecdbeebdcbdeccedbaaaaaabbbbbbceaddaecdecaacedaecddceaaecddceadecaacedceaddaecbbbbbbaaaaaabdceecdbedbccbdecdbeebdccdbeebdcedbccbdebdceecdbaaaaaaccccccaedbbdeabeaddaebdeabbaeddeabbaedbeaddaebaedbbdeaccccccddddddcebaabecaecbbceabecaacebbecaacebaecbbceacebaabecddddddbbbbbbcedaadecaecddceadecaaceddecaacedaecddceacedaadecbbbbbbeeeeeebdcaacdbadbccbdacdbaabdccdbaabdcadbccbdabdcaacdbeeeeeeeeeeeebdcaacdbadbccbdacdbaabdccdbaabdcadbccbdabdcaacdbeeeeeebbbbbbcedaadecaecddceadecaaceddecaacedaecddceacedaadecbbbbbbddddddcebaabecaecbbceabecaacebbecaacebaecbbceacebaabecddddddccccccaedbbdeabeaddaebdeabbaeddeabbaedbeaddaebaedbbdeaccccccaaaaaabdceecdbedbccbdecdbeebdccdbeebdcedbccbdebdceecdbaaaaaaddddddacebbecabcaeeacbecabbaceecabbacebcaeeacbacebbecaddddddeeeeeedbcaacbdabdccdbacbdaadbccbdaadbcabdccdbadbcaacbdeeeeeeccccccebaddabedbeaaebdabeddebaabeddebadbeaaebdebaddabeccccccaaaaaaebcddcbedbeccebdcbeddebccbeddebcdbeccebdebcddcbeaaaaaabbbbbbaceddecadcaeeacdecaddaceecaddacedcaeeacdaceddecabbbbbbbbbbbbaceddecadcaeeacdecaddaceecaddacedcaeeacdaceddecabbbbbbaaaaaaebcddcbedbeccebdcbeddebccbeddebcdbeccebdebcddcbeaaaaaaccccccebaddabedbeaaebdabeddebaabeddebadbeaaebdebaddabecccccceeeeeedbcaacbdabdccdbacbdaadbccbdaadbcabdccdbadbcaacbdeeeeeeddddddacebbecabcaeeacbecabbaceecabbacebcaeeacbacebbecadddddd'

# word4 = 'e'*2 + gofirst[:24] + 'e'*8 + gofirst[24:] + 'e'*2
# word5 = milp_search_concatenate(word4, 5, solution_len=10, verbose=True)
# 'eeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecdd'


# n = 6
# letters = "abcdef"
# word_2 = letters + letters[::-1]
# word_3 = "aecfdbbdfceaafcebddbecfabcefaddafecbbfecdaadcefbdcfebaabefcddefcbaabcfed"
# word_4 = word_3 + word_3[::-1]


# p0 = 7/90
# p1 = 32/90
# p2 = 12/90
# word6 = 'f'*7*16 + word5*3 + 'f'*32*16 + word5*3 + 'f'*12*16 + word5*3 + 'f'*32*16 + word5*3 + 'f'*7*16
# word6 = 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecddeeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecddeeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecddffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecddeeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecddeeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecddffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecddeeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecddeeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecddffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecddeeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecddeeabcddcbadbaccabdcbaddabceeeeeeeecbaddabcdbaccabdabcddcbaeeeecbaddabcdbcaacbdabcddcbaeeeeeeeeabcddcbadbcaacbdcbaddabceeaaebcddcbedbeccebdcbeddebcaaaaaaaacbeddebcdbeccebdebcddcbeaaaacbeddebcdbceecbdebcddcbeaaaaaaaaebcddcbedbceecbdcbeddebcaabbaedccdeaceaddaecdeaccaedbbbbbbbbdeaccaedceaddaecaedccdeabbbbdeaccaedcedaadecaedccdeabbbbbbbbaedccdeacedaadecdeaccaedbbccbedaadebaebddbeadebaabedccccccccdebaabedaebddbeabedaadebccccdebaabedaedbbdeabedaadebccccccccbedaadebaedbbdeadebaabedccddaecbbceabeaccaebceabbaecddddddddceabbaecbeaccaebaecbbceaddddceabbaecbecaacebaecbbceaddddddddaecbbceabecaacebceabbaecddffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'


# n = 7
# letters = "abcdefg"
# word_2 = letters + letters[::-1]
# word_3 = "abedgcffcgdebaadgbfceecfbgdaaebfgdccdgfbeaafgedbccbdegfabgdefaccafedgbcdebfgaagfbedccfbedaggadebfccgbfeaddaefbgccgedbfaafbdegcdfegbcaacbgefdegfbdaccadbfgefdbgecaacegbdf"
# word_4 = word_3 + word_3[::-1]

# n = 8
# letters = "abcdefgh"
# word_2 =  "abcdefghhgfedcba"
# word_3 = "acedbhgffghbdecaahbdecgffgcedbhabcfhedgaagdehfcbbedafcghhgcfadebfcbadeghhgedabcffdehbcgaagcbhedfgdchbafeefabhcdggdfabhceechbafdggebhfacddcafhbeggecafhbddbhfaceghafedcbggbcdefahhcdefabggbafedch"
# word_4 = word_3 + word_3[::-1]

# n = 9
# letters = "abcdefghi"
# word_2 = letters + letters[::-1]
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
m = 10

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
k = 5
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

status = prob.solve(pulp.COIN(msg=0))

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
k = 5
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
letters = "abcde"
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
