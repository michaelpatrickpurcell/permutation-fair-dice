import random
import numpy as np

from tqdm import tqdm
from scipy.special import factorial, comb
from collections import Counter
from itertools import combinations, permutations, product
from pyeasyga import pyeasyga

from utils import coverage_search, coverage_search2
from utils import permute_letters, score_orders
from utils import aggregate_scores, normalize_score
from utils import word_to_dice, dice_to_word
from utils import l0_norm, l1_norm, l2_norm, l4_norm, max_norm
from permutation_lists import perms_list_6a

# ============================================================================
# Playing with pyeasyga

n = 5
word_1 = "abcde"
word_2 = word_1 + word_1[::-1]
used_perms = coverage_search(word_2, 3, norm=l4_norm)
word_3 = ''.join([permute_letters(w, p) for w,p in used_perms])
word_4 = word_3 + word_3[::-1]

all_permutations = list(permutations(range(n)))
all_keys = sorted(list(permutations(word_1, 5)))
data = []
for p in all_permutations:
    perm_word = permute_letters(word_4, p)
    score = score_orders(perm_word, 5)
    nscore = normalize_score(score)
    data.append({k:score[k] for k in all_keys})

mu = np.mean((len(word_4)//5)**5 / 120)
epsilon = 0.0001
threshold = 5*mu
ga = pyeasyga.GeneticAlgorithm(data)        # initialise the GA with data
ga.population_size = 200                 # increase population size to 200 (default value is 50)
ga.generations = 5000
def fitness(individual, data):
    selected_data = [d for i,d in zip(individual, data) if i]
    score = aggregate_scores(*selected_data)
    # norm = l1_norm(nscore)# + n*(sum(individual) % 3) + n*sum(individual)
    ret = sum(score.values()) - np.exp(epsilon*abs(threshold - max(score.values()))) - mu**2*(sum(individual) % 5)
    # if max(score.values()) > threshold:
    #     ret = 0
    # if (sum(individual) % 5) != 0:
    #     ret = 0
    return ret

ga.fitness_function = fitness               # set the GA's fitness function
ga.run()                                    # run the GA
print(ga.best_individual())                 # print the GA's best solution

winner = ga.best_individual()[1]

indices = [i for i,w in enumerate(winner) if w == 1]
word_5 = ''.join(permute_letters(word_4, all_permutations[i]) for i in indices)

score = score_orders(word_5, 5)
print(sum(winner))
print(len(set(score.values())))
print(set(score.values()))

# ============================================================================
# Playing with pyeasyga
from pyeasyga import pyeasyga
from collections import Counter

n = 3
order_len = 3
word_1 = "abc"
d = 6
data = list(word_1 * (n*d))

def fitness(individual, data):
    # segments = [individual[n*i:n*(i+1)] for i in range(d*n)]
    # segment_sums = np.array([sum(s) for s in segments])
    slices = [individual[i::n] for i in range(n)]
    slice_sums = np.array([sum(s) for s in slices])
    if np.any(slice_sums != d):
        ret = -np.inf
    else:
        selected_data = ''.join([d for i,d in zip(individual, data) if i])
        counter = Counter(selected_data)
        score = score_orders(selected_data, order_len)
        nscore = normalize_score(score)
        norm = l2_norm(nscore)
        ret = -norm
    # else:
    #     ret = -10000
    return ret

def create_individual(data):
    subindices = sum([[i]*d for i in range(n)], [])
    np.random.shuffle(subindices)
    individual = [0]* (n**2 * d)
    for i,x in enumerate(subindices):
        individual[n*i + x] = 1
    return individual

def mutate(individual):
    segments = [individual[n*i:n*(i+1)] for i in range(len(individual)//n)]
    i,j = np.random.randint(len(segments), size=2)
    segments[i], segments[j] = segments[j], segments[i]
    return sum(segments, [])

def crossover(parent_1, parent_2):
    segments_1 = [parent_1[n*i:n*(i+1)] for i in range(d*n)]
    segments_2 = [parent_2[n*i:n*(i+1)] for i in range(d*n)]
    indices_1 = []
    indices_2 = []
    m = d // 2
    for j in range(n):
        temp_1 = [i for i,s in enumerate(segments_1) if s[j] == 1]
        indices_1 += temp_1[:m]
        temp_2 = [i for i,s in enumerate(segments_2) if s[j] == 1]
        indices_2 += temp_2[:m]
    sorted_indices_1 = sorted(indices_1)
    sorted_indices_2 = sorted(indices_2)
    child_1 = [segments_1[i] for i in sorted_indices_1]
    child_2 = [segments_2[i] for i in sorted_indices_2]
    return sum(child_1 + child_2, [])


ga = pyeasyga.GeneticAlgorithm(data)        # initialise the GA with data
ga.population_size = 1000                    # increase population size to 200 (default value is 50)
ga.generations = 1000
ga.fitness_function = fitness               # set the GA's fitness function
ga.create_individual = create_individual
# ga.crossover = crossover
# ga.mutate = mutate
ga.run()                                    # run the GA
print(ga.best_individual())                 # print the GA's best solution

winner = ga.best_individual()[1]
word_3 = ''.join([d for i,d in zip(winner, data) if i])
score_orders(word_3, 3)
