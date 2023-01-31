import numpy as np

from tqdm import tqdm
from itertools import combinations, permutations, product, accumulate

from tqdm import tqdm
from scipy.special import factorial, comb
from collections import Counter

import pulp


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
    constraints = dice_to_constraints(dice, dtype=np.float)
    scores = dict()
    for x in permutations(dice_names, k):
        d = len(dice[x[0]])
        accum = np.eye(d, d)
        for y, z in zip(x[:-1], x[1:]):
            accum = accum @ constraints[(y, z)]
        score = np.sum(accum)
        scores[x] = score
        if verbose:
            print(score)
    return scores


def score_orders2(word, k, verbose=False):
    l = len(word)
    dice_names = sorted(list(set(word)))
    row_lut = {(x,): i for i, x in enumerate(dice_names)}
    n = len(dice_names)
    indicator = np.zeros((n, l), dtype=np.int)
    for i, x in enumerate(word):
        indicator[row_lut[(x,)], i] = True
    accumulator = np.cumsum(indicator, axis=1)
    accumulators = [accumulator]
    row_luts = [row_lut]
    for m in range(2, k + 1):
        keys = sorted(list(permutations(dice_names, m)))
        row_lut = {x: i for i, x in enumerate(keys)}
        accumulator = np.zeros((len(keys), l), dtype=np.int)
        for i, x in enumerate(keys):
            mask = indicator[row_luts[0][x[-1:]]]
            j = row_luts[-1][x[:-1]]
            accumulator[i] = np.cumsum(accumulators[-1][j] * mask)
        row_luts.append(row_lut)
        accumulators.append(accumulator)
    ret = {x: accumulators[-1][row_luts[-1][x]][-1] for x in keys}
    return ret


def score_orders3(word, k, verbose=False):
    l = len(word)
    dice_names = sorted(list(set(word)))
    row_lut = {(x,): i for i, x in enumerate(dice_names)}
    n = len(dice_names)
    indicator = [[0 for i in range(l)] for j in range(n)]
    # indicator = np.zeros((n, l), dtype=np.uint64)
    for i, x in enumerate(word):
        indicator[row_lut[(x,)]][i] = True
    accumulator = [list(accumulate(indicator[j])) for j in range(n)]
    # accumulator = np.cumsum(indicator, axis=1)
    accumulators = [accumulator]
    row_luts = [row_lut]
    for m in range(2, k + 1):
        keys = sorted(list(permutations(dice_names, m)))
        row_lut = {x: i for i, x in enumerate(keys)}
        accumulator = [[0 for i in range(l)] for j in range(len(keys))]
        # accumulator = np.zeros((len(keys), l), dtype=np.uint64)
        for i, x in enumerate(keys):
            mask = indicator[row_luts[0][x[-1:]]]
            j = row_luts[-1][x[:-1]]
            temp = [int(a) * int(m) for a, m in zip(accumulators[-1][j], mask)]
            accumulator[i] = list(
                accumulate(temp)
            )  # np.cumsum(accumulators[-1][j] * mask)
        row_luts.append(row_lut)
        accumulators.append(accumulator)
    ret = {x: accumulators[-1][row_luts[-1][x]][-1] for x in keys}
    return ret


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


def coverage_search(word, order_len, norm=l2_norm, roots=[], unique=True, max_len=None):
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
            perm_score = score_orders2(perm_word, order_len)
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
    current_score = normalize_score(score_orders2(used_perms[0][0], order_len))
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


def compute_bigstring_index(dice_names, die, die_index):
    n = len(dice_names)
    all_perms = list(permutations(range(n), n))
    m = len(all_perms)
    perm_indices = number_to_base(die_index, m)
    face_index = sum([j * (m ** i) for i, j in enumerate(perm_indices[::-1])])
    die_index = dice_names.index(die)
    for i in perm_indices:
        die_index = all_perms[i].index(die_index)
    bigstring_index = face_index * n + die_index
    return bigstring_index


def create_bigstring(dice_names):
    temp = list(dice_names)
    n = len(dice_names)
    all_perms = list(permutations(range(n), n))
    for i in range(n - 1):
        temp = sum(
            [[dice_names[p[dice_names.index(x)]] for x in temp] for p in all_perms], []
        )
    bigstring = "".join(temp)
    return bigstring


def number_to_base(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def milp_search_concatenate(word, order_len, solution_len=None, verbose=False):
    letters = "".join(sorted(list(set(word))))
    n = len(letters)
    all_perms = list(permutations(range(n)))
    all_orders = list(permutations(letters, order_len))

    counts = []
    for i, p in tqdm(enumerate(all_perms), disable=~verbose):
        counts.append(score_orders2(permute_letters(word, p), order_len))

    xs = []
    for i in tqdm(range(len(all_perms)), disable=~verbose):
        xs.append(pulp.LpVariable("x%i" % i, lowBound=0, cat="Integer"))

    if solution_len is not None:
        m = solution_len
        target = m * sum([int(v) for v in counts[0].values()]) // len(all_orders)
        prob = pulp.LpProblem("myProblem", pulp.LpMaximize)
        prob += pulp.lpSum(xs) == m  # There is no objective for this formulation!
        for order in tqdm(all_orders, disable=~verbose):
            prob += pulp.lpSum([x * ct[order] for x, ct in zip(xs, counts)]) <= target
    else:
        ncounts = [normalize_score(ct) for ct in counts]
        prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
        prob += pulp.lpSum(xs)  # This is the objective!
        for order in tqdm(all_orders, disable=~verbose):
            prob += pulp.lpSum([x * nct[order] for x, nct in zip(xs, ncounts)]) == 0.0
        prob += pulp.lpSum(xs) >= 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=int(verbose)))

    if status == -1:
        ret = None
    else:
        print([pulp.value(x) for x in xs])
        used_perms = [p for p, x in zip(all_perms, xs) if pulp.value(x) > 0.0]
        multipliers = [int(pulp.value(x)) for x in xs if pulp.value(x) > 0.0]
        segments = [permute_letters(word, p) for p in used_perms]
        # ret = used_perms, segments
        ret = "".join([m * s for m, s in zip(multipliers, segments)])

    return ret


def milp_search_insert(
    word,
    order_len,
    solution_len=None,
    upBound=None,
    palindrome=False,
    positions=None,
    verbose=False,
):
    letters_list = list(word)
    last_letter = sorted(list(set(letters_list)))[-1]
    new_letter = chr(ord(last_letter) + 1)
    letters = "".join(sorted(list(set(letters_list))) + [new_letter])
    n = len(letters)
    if positions:
        all_positions = sorted(list(positions))
    else:
        all_positions = [i for i in range(len(word) + 1)]

    all_orders = list(permutations(letters, n))
    short_orders = list(permutations(letters, order_len))

    counts = []
    for i in tqdm(all_positions, disable=~verbose):
        temp = list(word)
        temp.insert(i, new_letter)
        counts.append(score_orders2(temp, n))

    xs = []
    for i in tqdm(all_positions, disable=~verbose):
        if upBound:
            xs.append(
                pulp.LpVariable("x%i" % i, lowBound=0, upBound=upBound, cat="Integer")
            )
        else:
            xs.append(pulp.LpVariable("x%i" % i, lowBound=0, cat="Integer"))

    m = len(word) // (n - 1)
    if solution_len is None:
        solution_len = m

    # target = ((m**n) // factorial(n, exact=True)) * factorial(n-order_len, exact=True) * comb(n, n-order_len, exact=True)
    # target = (m**n) // factorial(order_len, exact=True)
    target = (
        comb(n, n - order_len, exact=True)
        * factorial(n - order_len, exact=True)
        * np.mean(list(counts[0].values()))
        * solution_len
    )

    prob = pulp.LpProblem("myProblem", pulp.LpMaximize)
    prob += pulp.lpSum(xs) == solution_len

    for short_order in tqdm(short_orders, disable=~verbose):
        temp = []
        for order in tqdm(all_orders, disable=~verbose):
            temp_order = tuple([x for x in order if x in short_order])
            if short_order == temp_order:
                temp += [x * ct[order] for x, ct in zip(xs, counts)]
        prob += pulp.lpSum(temp) == target

    if palindrome:
        for x, y in zip(xs, xs[::-1]):
            prob += x == y

    status = prob.solve(pulp.PULP_CBC_CMD(msg=int(verbose)))

    if status == -1:
        ret = [""]
    else:
        solution = [pulp.value(x) for x in xs]
        ret = list(word)
        for i, mult in zip(all_positions[::-1], solution[::-1]):
            ret.insert(i, "".join(int(mult) * [new_letter]))

    return "".join(ret)


def milp_exhaust_binary(m, d, verbose=False):
    prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
    xs = []
    for i in range(m):
        xs.append(pulp.LpVariable("x%i" % i, lowBound=0, upBound=1, cat="Integer"))

    # vec = [2**i for i in range(m)]
    # print(vec)
    # temp = pulp.lpDot(vec, xs)
    # print(temp)
    # prob += temp

    if d >= 2:
        vec = [1 for i in range(m)]
        target2 = m // 2
        prob += pulp.lpDot(vec, xs) == target2
    if d >= 3:
        vec = [(i + 1) for i in range(m)]
        target3 = m * (3 * m + 2) // 12
        prob += pulp.lpDot(vec, xs) == target3
    if d >= 4:
        vec = [(i + 1) ** 2 for i in range(m)]
        target4 = (m ** 2 * (m + 1)) // 6
        prob += pulp.lpDot(vec, xs) == target4
    if d >= 5:
        vec = [(i + 1) ** 3 for i in range(m)]
        target5 = (m * (5 * (m ** 2) * (3 * m + 4) - 4)) // 120
        prob += pulp.lpDot(vec, xs) == target5

    status = 1
    solutions = []
    # while status != -1:
    #     status = prob.solve(pulp.PULP_CBC_CMD(msg=int(verbose)))
    #     if status == -1:
    #         pass
    #     else:
    #         solution = [pulp.value(x) for x in xs]
    #         solutions.append(solution)
    #         vec = [2 ** i for i in range(m)]
    #         target = pulp.lpDot(vec, solution) + 1
    #         print(target)
    #         prob += pulp.lpDot(vec, xs) >= target

    status = prob.solve(pulp.PULP_CBC_CMD(msg=int(verbose)))
    if status == -1:
        pass
    else:
        solution = [pulp.value(x) for x in xs]
        solutions.append(solution)

    return solutions


def bits_to_array(bits, m):
    n = (len(bits) // m) + 1
    temp = bits.reshape(n - 1, m)
    row0 = (n - 1) * np.ones(m, dtype=int)
    rows = [2 * (i + 1) * temp[i] + (n - i - 2) for i in range(n - 1)]
    array = np.row_stack([row0] + rows)
    return array


def array_to_word(array, letters):
    letter_dict = {l: i for i, l in enumerate(letters)}
    ret = []
    for j in range(array.shape[1]):
        ret += sorted(letters, key=lambda l: array[letter_dict[l], j])
    return "".join(ret)


def is_gofirst_fair(word):
    letters = set(word)
    n = len(letters)
    scores = score_orders2(word, n)
    target = sum(scores.values()) // n
    ret = True
    for letter in letters:
        check = sum([v for k, v in scores.items() if k[-1] == letter])
        if check != target:
            ret = False
            break
    return ret


def is_place_fair(word):
    letters = set(word)
    n = len(letters)
    scores = score_orders2(word, n)
    target = sum(scores.values()) // n
    ret = True
    for i in range(n):
        for letter in letters:
            check = sum([v for k, v in scores.items() if k[i] == letter])
            if check != target:
                ret = False
                break
        if not ret:
            break
    return ret


def is_permutation_fair(word):
    letters = set(word)
    n = len(letters)
    scores = score_orders2(word, n)
    ret = len(set(scores.values())) == 1
    return ret


def expand_to_lcm(s):
    c = Counter(s)
    lcm = math.lcm(*c.values())
    t = []
    for x in s:
        t.append(x * (lcm // c[x]))

    return "".join(t)
