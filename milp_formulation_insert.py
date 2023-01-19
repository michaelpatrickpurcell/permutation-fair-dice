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

letters = "abcd"  # efg"
word2 = letters + letters[::-1]

search_results = milp_search(word2, 3, verbose=True)
if search_results:
    used_perms, segments_3 = search_results
    word_3 = "".join(segments_3)

search_results = milp_search(word_3, 4, solution_len=5, verbose=True)
if search_results:
    used_perms, segments_4 = search_results
    word4 = "".join(segments_4)

# word_4b = "".join([permute_letters(word_4, rotl((0, 1, 2, 3, 4), i)) for i in range(5)])
# segments_4b = ["".join(rotl(segments_4, i)) for i in range(6)]
# word_4b = "".join(segments_4b)

search_results = milp_search(word_4, order_len=5, solution_len=15, verbose=True)
if search_results:
    used_perms, segments_5 = search_results
    word_5 = "".join(segments_5)


# ============================================================================

seed3 = "cabbcabaaccbcabbca"
three_perms = list(permutations([0, 1, 2], 3))

hits = []
for i, perms in enumerate(product(three_perms, repeat=5)):
    print(i)
    word3 = "".join([permute_letters(seed3, perm) for perm in perms])
    word4 = milp_search_insert(word3, palindrome=True, upBound=2)
    if word4:
        print("hit")
        print("secondary test")
        word5 = milp_search_insert(word4)
        if word5:
            print("!!!!!HIT!!!!!")
            hits.append((word5))
        else:
            print("so close")
    else:
        print("miss")
