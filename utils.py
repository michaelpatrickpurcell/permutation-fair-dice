import numpy as np

from tqdm import tqdm
from itertools import combinations, permutations, product

def max_norm(score):
    return max([abs(score[x]) for x in score])


def l0_norm(score):
    return sum(True ^ np.isclose(np.array(list(score.values())), 0))


def l1_norm(score):
    return sum([abs(score[x]) for x in score])


def l2_norm(score):
    return np.sqrt(sum([score[x] ** 2 for x in score]))


def l4_norm(score):
    return (sum([score[x] ** 4 for x in score])) ** (1 / 4)


def rotl(x, i):
    return x[i:] + x[:i]


def rotr(x, i):
    return x[-i:] + x[:-i]


def insert(x, i, y):
    temp = list(x)
    temp.insert(i, y)
    return tuple(temp)


def dice_to_word(dice):
    dice_names = list(dice.keys())
    m = len(dice_names)
    d = len(dice[dice_names[0]])
    foo = [[(x, dice[x][i]) for i in range(d)] for x in dice_names]
    bar = sum(foo, [])
    ram = sorted(bar, key=lambda x: x[1])
    word = "".join([t[0] for t in ram])
    return word


def word_to_dice(word):
    dice_names = set(word)
    dice = dict()
    for i, w in enumerate(word):
        if w in dice:
            dice[w].append(i)
        else:
            dice[w] = [i]
    return dice


def apply_perm(perm, iterable):
    return [iterable[p] for p in perm]


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


def dice_to_constraints(dice, dtype=np.int):
    dice_names = list(dice.keys())
    d = len(dice[dice_names[0]])
    dice_pairs = list(permutations(dice_names, 2))
    n = len(dice_pairs)
    constraints = dict()
    for x, y in dice_pairs:
        foo = np.array(dice[x]).reshape(len(dice[x]), 1)
        bar = np.array(dice[y]).reshape(1, len(dice[y]))
        constraint = foo > bar
        constraints[(x, y)] = constraint.astype(dtype)
    return constraints


def score_orders(word, k, verbose=False):
    dice = word_to_dice(word)
    dice_names = list(dice.keys())
    d = len(dice[dice_names[0]])
    constraints = dice_to_constraints(dice, dtype=np.float)
    scores = dict()
    for x in permutations(dice_names, k):
        accum = np.eye(d, d)
        for y, z in zip(x[:-1], x[1:]):
            accum = accum @ constraints[(y, z)]
        score = np.sum(accum)
        scores[x] = score
        if verbose:
            print(score)
    return scores


def aggregate_scores(*args):
    aggregator = dict()
    for arg in args:
        for x in arg:
            if x in aggregator:
                aggregator[x] += arg[x]
            else:
                aggregator[x] = arg[x]
    return aggregator


def normalize_score(score):
    mu = np.mean(np.array(list(score.values())))
    normalized_score = {x: score[x] - mu for x in score}
    return normalized_score


def coverage_search(
    word, order_len, norm=l2_norm, roots=[], unique=True, max_len=None
):
    m = len(set(word))
    all_perms = list(permutations(range(m)))

    if roots == []:
        ext_roots = [word]
    else:
        ext_roots = roots

    perm_words = []
    candidates = dict()
    for root in tqdm(ext_roots):
        for perm in all_perms:
            perm_word = permute_letters(root, perm)
            perm_score = score_orders(perm_word, order_len)
            if perm_word not in perm_words:
                nscore = normalize_score(perm_score)
                candidates[(root, perm)] = nscore
                perm_words.append(perm_word)

    print(len(candidates))

    if unique:
        blacklist = [(word, tuple(range(m)))]
    else:
        blacklist = []
    used_perms = [(word, tuple(range(m)))]
    current_score = normalize_score(score_orders(used_perms[0][0], order_len))
    current_norm = norm(current_score)
    print(current_norm)
    while not np.isclose(current_norm, 0, atol=0.05):
        next_scores = dict()
        for k, v in candidates.items():
            if k not in blacklist:
                next_scores[k] = aggregate_scores(current_score, v)

        next_norms = {k: norm(v) for k, v in next_scores.items()}

        next_perms = sorted(next_norms.items(), key=lambda x: x[1])
        next_perm = next_perms[0]

        used_perms.append(next_perm[0])
        if unique:
            blacklist.append(next_perm[0])

        current_score = next_scores[next_perm[0]]
        current_norm = norm(current_score)
        print(current_norm)
        if (max_len is not None) and (len(used_perms) >= max_len):
            break

    return used_perms

def coverage_search2(
    word, order_len, norm=l2_norm, roots=[], unique=True, max_len=None
):
    m = len(set(word))
    all_perms = list(permutations(range(m)))

    if roots == []:
        ext_roots = [word]
    else:
        ext_roots = roots

    # Find a list of all candidates.
    # Here we score each candidate in isolation as per the
    # previous implementation.  We won't use those scores, we
    # just want to modify the interface as little as possible as
    # we run some initial experiments.
    perm_words = []
    candidates = dict()
    for root in tqdm(ext_roots):
        for perm in all_perms:
            perm_word = permute_letters(root, perm)
            perm_score = score_orders(perm_word, order_len)
            if perm_word not in perm_words:
                nscore = normalize_score(perm_score)
                candidates[(root, perm)] = nscore
                perm_words.append(perm_word)

    print(len(candidates))

    if unique:
        blacklist = [(word, tuple(range(m)))]
    else:
        blacklist = []

    used_perms = [(word, tuple(range(m)))]
    current_word = word
    current_score = normalize_score(score_orders(word, order_len))
    current_norm = norm(current_score)
    print(current_norm)
    while not np.isclose(current_norm, 0, atol=0.05):
        next_scores = dict()
        for k, v in candidates.items():
            if k not in blacklist:
                ext = permute_letters(k[0], k[1])
                score = score_orders(current_word + ext, order_len)
                next_scores[k] = normalize_score(score)

        next_norms = {k: norm(v) for k, v in next_scores.items()}

        next_perms = sorted(next_norms.items(), key=lambda x: x[1])
        next_perm = next_perms[0]

        used_perms.append(next_perm[0])
        if unique:
            blacklist.append(next_perm[0])

        ext = permute_letters(next_perm[0][0], next_perm[0][1])
        current_word = current_word + ext
        current_score = next_scores[next_perm[0]]
        current_norm = norm(current_score)
        print(current_norm)
        if (max_len is not None) and (len(used_perms) >= max_len):
            break

    return used_perms
