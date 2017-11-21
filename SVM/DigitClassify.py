import numpy as np


# import the images
def load_images(file_path):
    from os import listdir
    y_array = []
    file_names = listdir(file_path)  # list all the files in that file_path
    m = len(file_names)
    x_array = np.zeros((m, 1024))
    # get the label or y for the data
    for i in range(m):
        file_name = file_names[i].split('.')[0]  # get the file_name without .txt
        digit = int(file_name.split('_')[0])  # !!!get the label(注意string 到 int的转换)
        if digit == 9:  # !!! not 9 and 1, but -1 and 1
            y_array.append(-1)
        else:
            y_array.append(1)
        x_array[i, :] = img2vector( "%s/%s" % (file_path, file_names[i]))
    return x_array, y_array


# img2vector
def img2vector(filename):
    return_vect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            return_vect[0,32*i+j] = int(lineStr[j])
    return return_vect


# get kernel
def calc_kernel(x, xa, ktup):
    # kTup means (type, sigma)
    m, n = x.shape
    res = np.matrix(np.zeros((m, 1)))
    denominator = - 2 * (ktup[1]) ** 2
    if ktup[0] == "lin":
        res = x * xa.T
    elif ktup[0] == 'rbf':
        for i in range(m):
            res[i] = np.exp(((x[i, :] - xa) * (x[i, :] - xa).T)/denominator)
    else:
        raise NameError('the type of kernel is not recognized')
    return res


class OptStructure:
    # initialize structure to store the needed parameters
    def __init__(self, x_matrix, y_matrix, c, tole, ktup):
        self.c = c
        self.x_matrix = x_matrix
        self.y_matrix = y_matrix
        self.tole = tole
        self.m = self.y_matrix.shape[0]
        self.alphas = np.asmatrix(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.asmatrix(np.zeros((self.m, 2)))  # why the dimension should be m*2 error Cache
        self.k = np.matrix(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = calc_kernel(self.x_matrix, self.x_matrix[i, :], ktup)


def take_steps(os, i):
    # input(what elements of os)
    # the output: update alpha_i, alpha_j and b
    ei = calc_e(os, i)
    kkt_or_not = (os.alphas[i] > 0 and os.y_matrix[i] * ei > os.tole) or \
                 (os.alphas[i] < os.c and os.y_matrix[i] * ei < -os.tole)  # why >> and <<
    # select the i which violates the KKT condition
    if kkt_or_not:
        j, ej = select_j(os, i, ei)
        # j, ej = select_j(os, i, ei)
        # j, ej = select_j_random(os, i)
        kii = os.k[i,i]
        kij = os.k[i,j]
        kjj = os.k[j,j]
        eta = 2 * kij - kii - kjj
        if eta >= 0:
            print('eta>=0')
            return 0
        alpha_i_old = os.alphas[i].copy()
        alpha_j_old = os.alphas[j].copy()
        delta_alpha_j = os.y_matrix[j] * (ej - ei) / eta
        alpha_j_new = alpha_j_old + delta_alpha_j
        # calHL
        if os.y_matrix[i] == os.y_matrix[j]:
            h = min(alpha_j_old + alpha_i_old, os.c)
            l = max(0, alpha_j_old + alpha_i_old - os.c)
        else:
            h = min(os.c, os.c + alpha_j_old - alpha_i_old)
            l = max(0, alpha_j_old - alpha_i_old)
        # clip_alpha_j
        if h == l:
            print('h == l， %d， %d', (h, l))
            return 0
        if alpha_j_new < l:
            alpha_j_new = l
        elif alpha_j_new > h:
            alpha_j_new = h
        if abs(alpha_j_new - alpha_j_old) < 0.00001 :  # take care of this if it is the fullSMO
            print('j not moving enough')
            return 0
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
        bi = os.b - ei - \
             (alpha_i_new - alpha_i_old) * os.y_matrix[i] * os.k[i, i] - \
             (alpha_j_new - alpha_j_old) * os.y_matrix[j] * os.k[j, i]
        bj = os.b - ej \
             - (alpha_i_new - alpha_i_old) * os.y_matrix[i] * os.k[i, j] \
             - (alpha_j_new - alpha_j_old) * os.y_matrix[j] * os.k[j, j]
        if 0 < os.alphas[i] < os.c:  # the alpha used here should new or old? the answer is new, why?
            os.b = bi
        elif 0 < os.alphas[j] < os.c:
            os.b = bj
        else:
            os.b = (bi + bj) / 2
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
    if len(unbounded) > 1:
        for k in unbounded:
            ek = calc_e(os, k)
            if abs(ek - ei) > max_delta_e:
                max_ej = ek
                max_j = k
                max_delta_e = abs(ek - ei)
        return max_j, max_ej
    else:
        j, ej = select_j_random(os, i)
        return j, ej

# def selectJ(os, i,  Ei):  # this is the second choice -heurstic, and calcs Ej
#     maxK = -1
#     maxDeltaE = 0
#     Ej = 0
#     os.e_cache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
#     validEcacheList = np.nonzero(os.e_cache[:, 0].A)[0]
#     if (len(validEcacheList)) > 1:
#         for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
#             if k == i: continue  # don't calc for i, waste of time
#             Ek = calc_e(os, k)
#             deltaE = abs(Ei - Ek)
#             if (deltaE > maxDeltaE):
#                 maxK = k
#                 maxDeltaE = deltaE
#                 Ej = Ek
#         return maxK, Ej
#     else:  # in this case (first time around) we don't have any valid eCache values
#         j, Ej = select_j_random(os, i)
#     return j, Ej


def calc_e(os, k):
    fk = float(np.multiply(os.alphas, os.y_matrix).T * os.k[:, k] + os.b)  # 1*n
    error_k = fk - float(os.y_matrix[k])
    # why not update os.e_cache immediately? because
    return error_k


def select_j_random(os, i):
    # select a random j, which should not be i
    j = i
    while i == j:
        j = int(np.random.uniform(0, os.m))
    ej = calc_e(os, j)
    return j, ej


def update_e(os, k):
    ek = calc_e(os, k)
    os.e_cache[k, :] = [1, ek]


# the main function of the full SMO
def full_smo(x_matrix, y_matrix, c, tole, max_iter, ktup):
    os = OptStructure(x_matrix, y_matrix, c, tole, ktup)
    pairs_changed = 0
    examine_all = 1
    iterations = 0
    while iterations < max_iter and (pairs_changed > 0 or examine_all == 1):
        pairs_changed = 0
        if examine_all == 1:
            # if the entire data is not rounded through, then we choose i in order, and choose j in random
            for i in range(os.m):
                pairs_changed += take_steps(os, i)
                print('examine_all, iter: %d, i: %d, pairs changed: %d' % (iterations, i, pairs_changed))
            iterations += 1
            # # linear svm(simple smo)
            # if pairs_changed != 0:
            #     iterations = 0
            # else:
            #     iterations += 1
            #
        else:
            nonbound = np.nonzero((os.alphas.A > 0) * (os.alphas.A < os.c))[0]
            for i in nonbound:
                pairs_changed += take_steps(os, i)
                print('non-bounds, iter: %d, i: %d, pairs changed: %d' % (iterations, i, pairs_changed))
            iterations += 1
        print('iters: %d' % iterations)
        if examine_all == 1:
            examine_all = 0
        elif pairs_changed == 0:
            examine_all = 1
    return os.alphas, os.b

# dataArr, labelArr = import_data('testSetRBF.txt')
# alphas, b = full_smo(np.asmatrix(dataArr), np.asmatrix(labelArr).T, 0.6, 0.001, 40, ('rbf', 1))
# print(b)
# print(alphas[alphas>0])


def digit_classify(sigma):
    data_arr, label_arr = load_images('testDigits')
    x = np.asmatrix(data_arr)
    y = np.asmatrix(label_arr).T
    alphas, b = full_smo(x, y, 200, 0.0001, 10000,  ('rbf', sigma))
    # how many vectors
    sv_index = np.nonzero(alphas.A > 0)[0]
    num_vector = sv_index.shape
    svs = x[sv_index, :]  # !!! noted that svs will be used all the time without the change of the dataset
    ysv = y[sv_index]  # !!! it will also be used in any case, think about the generation of the w
    print('number of vectors: %d' % num_vector)
    # the accuracy of the predictions of the training set
    m, n = alphas.shape
    pred = np.matrix(np.zeros((m, 1)))
    error_training = 0
    for i in range(m):
        # since a lot of alphas are o, so we do not take those into w to save time for calculation
        pred[i] = np.multiply(alphas[sv_index], y[sv_index]).T * calc_kernel(svs, x[i, :], ('rbf', sigma)) + b
        if np.sign(pred[i]) != np.sign(y[i]):
            error_training += 1
    print('error rate training: %f' % float(error_training/m))

    # the accuracy of the predictions of the test set
    data_arr, label_arr = load_images('trainingDigits')
    x = np.asmatrix(data_arr)
    y = np.asmatrix(label_arr).T
    m, n = y.shape
    pred_test = np.matrix(np.zeros((m, 1)))
    error_test = 0
    for i in range(m):
        # !!! since a lot of alphas are o, so we do not take those into w to save time for calculation
        pred_test[i] = np.multiply(alphas[sv_index], ysv).T * calc_kernel(svs, x[i, :], ('rbf', sigma)) + b
        if np.sign(pred_test[i]) != np.sign(y[i]):
            error_test += 1
    print('error rate test: %f' % float(error_test/m))

digit_classify(10)
