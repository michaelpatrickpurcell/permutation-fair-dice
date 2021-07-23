import copy
import random

import numpy as np

from tqdm import tqdm
from scipy.special import factorial, comb
from collections import Counter
from itertools import combinations, permutations, product

from utils import coverage_search, coverage_search2
from utils import permute_letters, score_orders
from utils import aggregate_scores, normalize_score
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
used_perms = coverage_search2(word_2, 3)
word_3 = ''.join([permute_letters(w, p) for w,p in used_perms])

# ============================================================================
# This section generates sets of dice that are:
#   4d2 that are 2/4 turn-order fair
#   4d6 that are 3/4 turn-order fair
#   4d12 that are 4/4 turn-order fair

n = 4
word_1 = "abcd"
word_2 = word_1 + word_1[::-1]
used_perms = coverage_search2(word_2, 3)
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


# ============================================================================
# This section generates sets of dice that are:
#   6d2 that are 2/6 perm fair
#   6d72 that are 3/6 perm fair
#   6d144 that are 4/6 perm_fair

n = 6
word_1 = "abcdef"
word_2 = word_1 + word_1[::-1]
used_perms = coverage_search(word_2, 3, norm=max_norm, max_len=3)
word2a = ''.join(permute_letters(w,p) for w,p in used_perms)
used_perms2 = coverage_search(word2a, 3, norm=max_norm, max_len=2)
word2b = ''.join(permute_letters(w,p) for w,p in used_perms2)
used_perms3 = coverage_search(word2b, 3, norm=l2_norm)
word_3 = ''.join(permute_letters(w,p) for w,p in used_perms3)
word_4 = word_3 + word_3[::-1]



# ============================================================================
# This section generates sets of dice that are:

n = 15
word_1 = "abcdefghijklmno"
word_2 = word_1 + word_1[::-1]
# This will crash your computer!
# used_perms = coverage_search(word_2, 3, norm=l2_norm)

# ============================================================================
# Playing with pyeasyga
from pyeasyga import pyeasyga

n = 5
word_1 = "abcde"
word_2 = word_1 + word_1[::-1]

all_permutations = list(permutations(range(n)))
all_keys = sorted(list(permutations(word_1, 3)))
data = []
for p in all_permutations:
    perm_word_2 = permute_letters(word_2, p)
    score = score_orders(perm_word_2, 3)
    nscore = normalize_score(score)
    data.append({k:score[k] for k in all_keys})

mu = np.mean(2**3 / 6)
epsilon = 2
threshold = 6*mu
ga = pyeasyga.GeneticAlgorithm(data)        # initialise the GA with data
ga.population_size = 2000                 # increase population size to 200 (default value is 50)
ga.generations = 1000
def fitness(individual, data):
    selected_data = [d for i,d in zip(individual, data) if i]
    score = aggregate_scores(*selected_data)
    # norm = l1_norm(nscore)# + n*(sum(individual) % 3) + n*sum(individual)
    ret = sum(score.values()) - np.exp(epsilon*abs(threshold - max(score.values())))
    # if max(score.values()) > threshold:
    #     ret = 0
    if (sum(individual) % 3) != 0:
        ret = 0
    return ret

ga.fitness_function = fitness               # set the GA's fitness function
ga.run()                                    # run the GA
print(ga.best_individual())                 # print the GA's best solution

winner = ga.best_individual()[1]

indices = [i for i,w in enumerate(winner) if w == 1]
word_3 = ''.join(permute_letters(word_2, all_permutations[i]) for i in indices)

print(score_orders(word_3, 3))
