from numpy_typing import np, ax, ax




# max wildcards in the fingerprint
MAX_SUB_FP_LEVEL = 5
def sub_fingerprint(fp:np.int8_1d[ax.time]) -> np.int8_2d[ax.sample, ax.time]:

    wildcard_loc = np.where(fp == 0)[0]
    if (len(wildcard_loc) > MAX_SUB_FP_LEVEL):
        return np.zeros((0, len(fp)), dtype=np.int8)

    if (len(wildcard_loc) == 0):
        return np.array([fp])

    nb_sub_fp = 2**len(wildcard_loc)
    res = np.tile(fp, (nb_sub_fp, 1))
    for i in range(nb_sub_fp):
        comb = [-1 if x == "0" else 1 for x in bin(i)[2:].zfill(len(wildcard_loc))]
        for j in range(len(wildcard_loc)):
            res[i, wildcard_loc[j]] = comb[j]

    return res




MAX_HASH = None
FP_LEN = None
POWERS = None
def init_hash(max_len:int) -> None:
    global MAX_HASH, FP_LEN, POWERS
    MAX_HASH = 2**(max_len-1)
    FP_LEN = max_len
    POWERS = [2**i for i in range(max_len+1)]


def __hash_one__(fp:np.int8_1d) -> int:
    v = 0
    for i in range(FP_LEN):
        if (fp[i] == 1):
            v += POWERS[i]
    if v >= MAX_HASH:
        v = POWERS[FP_LEN] - v - 1
    return v



def hash(fp:np.int8_2d[ax.sample, ax.time]) -> np.int32_1d:
    """
    Compute the hash of a fingerprint
    """
    hashes = np.zeros(fp.shape[0], dtype=np.int32)
    for s in range(len(fp)):
        hashes[s] = __hash_one__(fp[s])
    return hashes



def match(hashes:np.int64_1d, hashtable:"dict[int, list[str]]") -> "list[str]":
    """
    Match the hash with the hashtable
    """
    res = []
    for i in range(len(hashes)):
        matches = hashtable.get(hashes[i], [])
        res.extend(matches)

    return res

