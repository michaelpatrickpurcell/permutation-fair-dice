import copy
import random

import numpy as np

from tqdm import tqdm
from scipy.special import factorial, comb
from collections import Counter
from itertools import combinations, permutations, product

from utils import coverage_search, coverage_search2
from utils import permute_letters, score_orders
from utils import word_to_dice, dice_to_word
from utils import l0_norm, l1_norm, l2_norm, l4_norm, max_norm
from permutation_lists import perms_list_6a


# ============================================================================
# This section generates sets of dice that are:
#   3d2 that are 2/3 turn-order fair
#   3d6 that are 3/3 turn-order fair

n = 3
word_1 = "abc"
word_2 = word_1 + word_1[::-1]
used_perms = coverage_search(word_2, 3)
word_3 = ''.join([permute_letters(w, p) for w,p in used_perms])

# ============================================================================
# This section generates sets of dice that are:
#   4d2 that are 2/4 turn-order fair
#   4d6 that are 3/4 turn-order fair
#   4d12 that are 4/4 turn-order fair

n = 4
word_1 = "abcd"
word_2 = word_1 + word_1[::-1]
used_perms = coverage_search(word_2, 3)
word_3 = ''.join([permute_letters(w, p) for w,p in used_perms])
word_4 = word_3 + word_3[::-1]

# ============================================================================
# This section generates sets of dice that are:
#   5d2 that are 2/5 turn-order fair
#   5d12 that are 3/5 turn-order fair
#   5d24 that are 4/5 turn-order fair
#   5d2880 that are 5/5 turn order fair

n = 5
word_1 = "abcde"
word_2 = word_1 + word_1[::-1]
used_perms = coverage_search(word_2, 3, norm=l4_norm)
word_3 = ''.join([permute_letters(w, p) for w,p in used_perms])
word_4 = word_3 + word_3[::-1]
word_5 = ''.join([permute_letters(word_4, p) for p in permutations(range(5))])
