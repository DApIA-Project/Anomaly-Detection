from   _Utils.numpy import np, ax
import _Utils.Limits as Limits
import _Utils.geographic_maths as GEO


# |====================================================================================================================
# | LOSS MATRIX FOR TRAJECTORY ASSOCIATION
# |====================================================================================================================


def loss_matrix(y:np.float64_2d[ax.sample, ax.feature], y_:np.float64_2d[ax.sample, ax.feature])\
            -> np.float64_2d[ax.sample, ax.sample]:

    mat = np.zeros((len(y), len(y_)), dtype=np.float64)
    for i in range(len(y)):
        for j in range(len(y_)):
            mat[i, j] = GEO.distance(y[i][0], y[i][1], y_[j][0], y_[j][1])
    return mat


def print_loss_matrix(mat:np.float64_2d[ax.sample, ax.sample]) -> None:

    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if (mat[i, j] == Limits.INT_MAX):
                print(str("X").rjust(3), end=" ")
            else:
                print(str(int(mat[i, j])).rjust(3), end=" ")
        print()
    print()
    print()



def eval_association(mat:np.float64_2d[ax.sample, ax.sample], combination:np.int64_1d[ax.sample])\
        -> np.float64:

    loss = 0
    for i in range(len(combination)):
        if (combination[i] != -1):
            loss += mat[i, combination[i]]
    return loss


# |====================================================================================================================
# | UTILS FUNCTION TO MAKE FIRST ASSOCIATIONS, AND THEN CONTINUE ONLY ON A SUB-SET OF THE NON-ASSOCIATED TRAJECTORIES
# |====================================================================================================================

def apply_sub_associations(
            assoc:np.int64_1d, mat:np.float64_2d[ax.sample, ax.sample],
            sub_assoc:np.int64_1d, sub_mat:np.float64_2d[ax.sample, ax.sample],
            remain_y:np.int64_1d, remain_y_:np.int64_1d,) -> "tuple[np.int64_1d, np.int64_1d]":

    # locs = np.where(assoc != -1)[0]
    # assoc[remain_y[locs]] = remain_y_[sub_assoc[locs]]
    for i in range(len(remain_y)):
        if (sub_assoc[i] != -1):
            assoc[remain_y[i]] = remain_y_[sub_assoc[i]]

    for i in range(len(remain_y)):
        for j in range(len(remain_y_)):
            mat[remain_y[i], remain_y_[j]] = sub_mat[i, j]

    return assoc, mat


def compute_remaining_loss_matrix(mat:np.float64_2d[ax.sample, ax.sample], assoc:np.int64_1d)\
        -> "tuple[np.float64_2d[ax.sample, ax.sample], np.int64_1d[ax.sample], np.int64_1d[ax.sample]]":

    remain_y = np.where(assoc == -1)[0]
    if (len(remain_y) == 0):
        return np.zeros((0, 0)), [], []
    remain_y_ = np.where(mat[remain_y[0], :] != Limits.INT_MAX)[0]
    return mat[remain_y, :][:, remain_y_], remain_y, remain_y_



# |====================================================================================================================
# | OTHERS UTILS FUNCTION
# |====================================================================================================================

def argmin2(vec:np.float64_1d) -> "tuple[int, int]":
    "return the two minimum values of the vector"
    min_i_1 = 0
    min_i_2 = 0
    min_val_1 = Limits.INT_MAX
    min_val_2 = Limits.INT_MAX

    for i in range(len(vec)):

        if (vec[i] < min_val_1):
            # transfer min_1 to min_2
            min_i_2 = min_i_1
            min_val_2 = min_val_1

            # update min_1
            min_i_1 = i
            min_val_1 = vec[i]

        elif (vec[i] < min_val_2):
            min_i_2 = i
            min_val_2 = vec[i]

    return min_i_1, min_i_2

def mat_argmin(mat:np.float64_2d) -> "tuple[int, int]":
    "return the index of the minimum value in the matrix"
    min_i = 0
    min_j = 0
    min_val = Limits.INT_MAX

    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if (mat[i, j] < min_val):
                min_i = i
                min_j = j
                min_val = mat[i, j]

    return min_i, min_j

def have_n_inf_to(vec:np.float64_1d[ax.sample], val:float, n=2) -> bool:
    for i in range(len(vec)):
        if (vec[i] < val):
            n -= 1
            if (n == 0):
                return True

    return False
