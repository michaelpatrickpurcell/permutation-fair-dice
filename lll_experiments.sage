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

def find_concat_words(words, k, noise_params=None, verbose=False):
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
  row_sums = M.sum(1, keepdims=True)
  M_prime = M.shape[1] * M - row_sums

  A = Matrix(ZZ, M_prime.shape[0], M_prime.shape[1], M_prime)

  if verbose:
    print("Computing kernel")

  C = A.left_kernel()
  D = C.basis_matrix()

  if noise_params is not None:
    n3, sigma, rho = noise_params
    cov = np.ones((D.nrows(),D.nrows()))
    mask = ~np.eye(D.nrows(), dtype=np.bool)
    cov[mask] *= rho
    cov *= sigma^2
    Z = np.round(scipy.stats.multivariate_normal.rvs(cov=cov, size=int(n3))).astype(np.int).transpose()
    D_prime = np.column_stack((D, Z))
  else:
    D_prime = np.matrix(D)


  if verbose:
    print("Finding reduced basis")

  reduced_basis = np.matrix(Matrix(ZZ, D_prime.shape[0], D_prime.shape[1], D_prime).LLL())

  concat_words = []
  for basis_vector in reduced_basis:
    temp = np.array(basis_vector)
    if np.all(basis_vector[:D.ncols()] >= 0):
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

letters = "abcdef"
word2 = letters + letters[::-1]
word3s = find_concat_words([word2], 3, verbose=True)
word3bs = sum([[word, word[::-1]] for word in word3s], [])
word4s = find_concat_words(word3bs, 4, verbose=True)
word4bs = sum([[word, word[::-1]] for word in word4s], []) + [word + word[::-1] for word in word3s]
word5s = find_concat_words(word4bs, 5, verbose=True)
