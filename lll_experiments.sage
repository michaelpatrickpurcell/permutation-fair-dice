import numpy as np
from itertools import combinations, permutations, product

def score_orders2(word, k, verbose=False):
    l = len(word)
    dice_names = sorted(list(set(word)))
    row_lut = {(x,):i for i,x in enumerate(dice_names)}
    n = len(dice_names)
    indicator = np.zeros((n,l), dtype=np.int)
    for i,x in enumerate(word):
        indicator[row_lut[(x,)],i] = True
    accumulator = np.cumsum(indicator, axis=1)
    accumulators = [accumulator]
    row_luts = [row_lut]
    for m in range(2,k+1):
        keys = sorted(list(permutations(dice_names, m)))
        row_lut = {x:i for i,x in enumerate(keys)}
        accumulator = np.zeros((len(keys), l), dtype=np.int)
        for i,x in enumerate(keys):
            mask = indicator[row_luts[0][x[-1:]]]
            j = row_luts[-1][x[:-1]]
            accumulator[i] = np.cumsum(accumulators[-1][j] * mask)
        row_luts.append(row_lut)
        accumulators.append(accumulator)
    ret = {x:accumulators[-1][row_luts[-1][x]][-1] for x in keys}
    return ret

def permute_letters(string, permutation, relative=True):
    letter_set = set(string)
    if relative:
        pairs = [(string.index(letter), letter) for letter in letter_set]
        sorted_pairs = sorted(pairs)
        letters = "".join(l for i, l in sorted_pairs)
    else:
        letters = sorted(list(set(string)))
    subs = {s: letters[p] for s, p in zip(letters, permutation)}
    subs_string = "".join([subs[s] for s in string])
    return subs_string

def find_concat_words(words, k, verbose=False):
  letters = sorted(list(set(words[0])))
  scores_lists = []
  word_perms = []

  if verbose:
    print("Building scores matrix")

  for word in words:
    for i,perm in enumerate(permutations(range(len(letters)))):
      temp_word = permute_letters(word, perm)
      scores_dict = score_orders2(temp_word, int(k))
      scores_list = tuple(zip(*sorted(scores_dict.items())))[1]
      if not (scores_list in scores_lists):
        scores_lists.append(scores_list)
        word_perms.append((word, perm))

  M = np.array(scores_lists)
  n1 = M.shape[0]
  n2 = M.shape[1]

  row_sums = M.sum(1, keepdims=True)

  A = Matrix(ZZ, n1,n2, int(n2) * M - row_sums)

  if verbose:
    print("Computing kernel")

  C = A.left_kernel()
  D = C.basis_matrix()

  if verbose:
    print("Finding reduced basis")

  reduced_basis = np.matrix(D.LLL())

  concat_words = []
  for basis_vector in reduced_basis:
    temp = np.array(basis_vector)
    print(temp)
    if np.all(basis_vector >= 0):
      if verbose:
        print(temp)
      concat_word = ""
      for i,word_perm in enumerate(word_perms):
        for j in range(temp[0][i]):
          concat_word += permute_letters(word_perm[0], word_perm[1])
      if verbose:
        print(concat_word)
      concat_words.append(concat_word)
  return concat_words

#############################################################################

letters = "abcde"
word2 = letters + letters[::-1]
word3s = find_concat_words([word2], 3, verbose=True)
word3bs = sum([[word, word[::-1]] for word in word3s], [])
word4s = find_concat_words(word3bs, 4, verbose=True)
word4bs = sum([[word, word[::-1]] for word in word4s], []) + [word + word[::-1] for word in word3s]
word5s = find_concat_words(word4bs, 5, verbose=True)

letters = "abcdefg"
word2 = letters + letters[::-1]
k = 3

scores_lists = []
perms = []

print("Building scores matrix")
for i,perm in enumerate(permutations(range(len(letters)))):
  temp_word = permute_letters(word2, perm)
  scores_dict = score_orders2(temp_word, int(k))
  scores_list = tuple(zip(*sorted(scores_dict.items())))[1]
  if not (scores_list in scores_lists):
    scores_lists.append(scores_list)
    perms.append(perm)


M = np.array(scores_lists)
n1 = M.shape[0]
n2 = M.shape[1]

row_sums = M.sum(1, keepdims=True)

A = Matrix(ZZ, n1,n2, int(n2) * M - row_sums)

print("Computing kernel")
C = A.left_kernel()
D = C.basis_matrix()

print("Finding reduced basis")
reduced_basis = np.matrix(D.LLL())

for basis_vector in reduced_basis:
  if (np.all(basis_vector >= 0) or np.all(basis_vector <= 0)):
    temp = np.array(basis_vector)
    print(temp)
    word3 = ""
    for i,perm in enumerate(perms):
      for j in range(temp[0][i]):
        word3 += permute_letters(word2, perm)
    print(word3)



k = 4

scores_lists = []
perms = []

print("Building scores matrix")
for i,perm in enumerate(permutations(range(len(letters)))):
  temp_word = permute_letters(word3, perm)
  scores_dict = score_orders2(temp_word, int(k))
  scores_list = tuple(zip(*sorted(scores_dict.items())))[1]
  if not (scores_list in scores_lists):
    scores_lists.append(scores_list)
    perms.append(perm)

M = np.array(scores_lists)
n1 = M.shape[0]
n2 = M.shape[1]

row_sums = M.sum(1, keepdims=True)

A = Matrix(ZZ, n1,n2, int(n2) * M - row_sums)

print("Computing kernel")
C = A.left_kernel()
D = C.basis_matrix()

print("Finding reduced basis")
reduced_basis = np.matrix(D.LLL())

for basis_vector in reduced_basis:
  if (np.all(basis_vector >= 0) or np.all(basis_vector <= 0)):
    temp = np.array(basis_vector)
    print(temp)
    word4 = ""
    for i,perm in enumerate(perms):
      for j in range(temp[0][i]):
        word4 += permute_letters(word3, perm)
        print(word4)
