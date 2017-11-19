import numpy as np


# import the data
def import_data(filename):
    x_array = []
    y_array = []
    fr = open(filename)
    for line in fr.readlines():
        line_data = line.strip().split('\t')
        x_array.append([float(line_data[0]), float(line_data[1])])
        y_array.append(float(line_data[2]))
    return x_array, y_array

# x_array, y_array = import_data('testSet.txt')
# print(x_array)


class OptStructure:
    # initialize structure to store the needed parameters
    def __init__(self, x_matrix, y_matrix, c, tole):
        self.c = c
        self.x_matrix = x_matrix
        self.y_matrix = y_matrix.T
        self.tole = tole
        self.m = y_matrix.shape[1]
        self.alphas = np.matrix(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.matrix(np.zeros((self.m, 2)))  # why the dimension should be m*2 error Cache

def take_steps(os, i, c):
    # input(what elements of os)
    # the output: update alpha_i, alpha_j and b
    ei = calc_e(os, i)
    kkt_or_not = (os.alphas[i] > 0 and os.y_matrix[i] * ei > os.tole) \
                 or (os.alphas[i] < c and os.y_matrix[i] * ei < -os.tole) # why >> and <<
    # select the i which violates the KKT condition
    if kkt_or_not:
        j, ej = select_j(os, i, ei)
        kii = os.x_matrix[i, :] * (os.x_matrix[i, :]).T
        kij = os.x_matrix[i, :] * (os.x_matrix[j, :]).T
        kjj = os.x_matrix[j, :] * (os.x_matrix[j, :]).T
        eta = 2 * kij - kii - kjj
        if eta >= 0:
            print('eta>=0')
            return 0
        alpha_i_old = os.alphas[i].copy()
        alpha_j_old = os.alphas[j].copy()
        delta_alpha_j = os.y_matrix[j] / eta * (ej - ei)
        alpha_j_new = alpha_j_old + delta_alpha_j
        # calHL
        if os.y_matrix[i] == os.y_matrix[j]:
            h = min(alpha_j_old + alpha_i_old, c)
            l = max(0, alpha_j_old + alpha_i_old - c)
        else:
            h = min(c, c + alpha_j_old - alpha_i_old)
            l = max(0, alpha_j_old - alpha_i_old)
        # clip_alpha_j
        if h == l:
            print('h==l')
            return 0
        if alpha_j_new < l:
            alpha_j_new = l
        elif alpha_j_new > h:
            alpha_j_new = h
        # get alpha_i
        s = os.y_matrix[i] * os.y_matrix[j]
        delta_alpha_i = - s * (alpha_j_new - alpha_j_old)
        alpha_i_new = alpha_i_old + delta_alpha_i
        # update alpha_i, alpha_j
        os.alphas[i] = alpha_i_new
        os.alphas[j] = alpha_j_new
        update_e(os, i)  # since alpha has been changed, so will the os.e_cache. What about other os.e_cache
        update_e(os, j)
        # get b
        bi = os.b - ei\
             - delta_alpha_i * os.y_matrix[i] * (os.x_matrix[i, :] * (os.x_matrix[i, :]).T) \
             - delta_alpha_j * os.y_matrix[j] * (os.x_matrix[j, :] * (os.x_matrix[i, :]).T)
        bj = os.b - ej \
             - delta_alpha_i * os.y_matrix[i] * (os.x_matrix[i, :] * (os.x_matrix[j, :]).T) \
             - delta_alpha_j * os.y_matrix[j] * (os.x_matrix[j, :] * (os.x_matrix[j, :]).T)
        if 0 < os.alphas[i] < c:  # the alpha used here should new or old? the answer is new, why?
            b = bi
        elif 0 < os.alphas[j] < c:
            b = bj
        else:
            b = (bi + bj) / 2
        # update b
        os.b = b
        return 1
    else:
        return 0


def select_j(os, i, ei):
    # get the non-zeros of the alphas
    # this part need to be re-read since you almost follow what the MLiA tells
    # the input and the output
    # the key ideas of such selection
    max_delta_e = 0
    max_ej = 0  # the corresponding ej when delta_e is the max
    max_j = 0  # the corresponding j when delta_e is the max
    os.e_cache[i, :] = [1, ei]
    unbounded = np.nonzero(os.e_cache[:, 0].A)[0]  # special non-zero get way
    if len(unbounded) > 0:
        for k in unbounded:
            ek = calc_e(os, k)
            if abs(ek - ei) > max_delta_e:
                max_ej = ek
                max_j = k
        return max_j, max_ej
    else:
        j = select_j_random(os, i)
        ej = calc_e(os, j)
        return j, ej


def calc_e(os, k):
    w = np.multiply(os.alphas, os.y_matrix).T * os.x_matrix  # 1*n
    error_k = float(w * (os.x_matrix[k, :]).T - os.y_matrix[k])
    # why not update os.e_cache immediately? because
    return error_k


def select_j_random(os, i):
    # select a random j, which should not be i
    j = i
    while i == j:
        j = np.random.randint(os.m)
    return j


def update_e(os, k):
    ek = calc_e(os, k)
    os.e_cache[k, :] = [1, ek]


# the main function of the full SMO
def full_smo(x_matrix, y_matrix, c, tole, max_iter):
    os = OptStructure(x_matrix, y_matrix, c, tole)
    pairs_changed = 0
    examine_all = 1
    iters = 0
    while iters < max_iter and (pairs_changed > 0 or examine_all == 1):
        pairs_changed = 0
        if examine_all == 1:
            # if the entire data is not rounded through, then we choose i in order, and choose j in random
            for i in range(os.m):
                pairs_changed += take_steps(os, i, os.c)
            iters += 1
            print('examine_all, iter: %d, i: %d, pairs changed: %d' %(iters, i, pairs_changed))
        else:
            nonbound = np.nonzero((os.alphas.A > 0) * (os.alphas.A < c))[0]
            for i in nonbound:
                pairs_changed += take_steps(os, i, os.c)
            iters += 1
            print('nonbounds, iter: %d, i: %d, pairs changed: %d' % (iters, i, pairs_changed))
        if examine_all == 1:
            examine_all = 0
        elif pairs_changed == 0:
            examine_all = 1
    return os.alphas, os.b

dataArr, labelArr = import_data('testSet.txt')
alphas, b = full_smo(np.matrix(dataArr), np.matrix(labelArr), 0.6, 0.001, 40)
print(b)





