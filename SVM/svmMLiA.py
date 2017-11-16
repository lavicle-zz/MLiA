import numpy as np


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArray = line.strip().split('\t')
        dataMat.append([float(lineArray[0]), float(lineArray[1])])
        labelMat.append(float(lineArray[2]))
    return dataMat, labelMat


def select_j(i, m):
    # this def is to selct alpha_j randomly based on alpha_i
    # m is the number of samples, the selected index must not exceed m
    j = i
    while i == j:  # prevent alpha_j= alpha_i
        j = np.random.random_integers(m)-1  # generate a random integer
        # print(j)
    return j


def clip_alpha_j(alpha_j, H, L):
    if alpha_j > H:
        alpha_j = H
    elif alpha_j < L:
        alpha_j = L
    return alpha_j


def cal_new_alphaij(alphas, x_i, x_j, y_i, y_j, s, i, j, C, Ei, Ej):
    # calculate kernelii,ij,jj
    # then calculate eta as a in -b/a(not 2a since 1/2*eta is the a)
    # calculate delta_alpha
    # dataMat m-n; alphas,LableMat m-1;
    kernel_ij = x_i*np.transpose(x_j)
    kernel_ii = x_i*np.transpose(x_i)
    kernel_jj = x_j*np.transpose(x_j)
    eta = 2*kernel_ij - kernel_ii - kernel_jj
    if eta>0: print("eta>0")
    delta_alpha_j = y_j * (Ej - Ei)/eta  # it is data(j) - data(i)
    alpha_newj = alphas[j] + delta_alpha_j  # get alpha_j

    if y_i == y_j:
        L = max(0, alphas[i] +alphas[j] - C)
        H = min(alphas[i] +alphas[j], C)
    else:
        L = max(0, alphas[j] - alphas[i])
        H = min(C, C + alphas[j] - alphas[i])

    if L == H: print("L == H")
    alpha_newj = clip_alpha_j(alpha_newj, H, L)  # clip alpha_j

    # get alpha_newi based on the constant equation yi*alphai +yj*alphaj =cons
    alpha_newi = alphas[i] + s * (alphas[j] - alpha_newj)
    return alpha_newj, alpha_newi


def cal_b(alpha_newj, alpha_newi, w_new, x_i, x_j, y_i, y_j, C):
    bj = y_j - w_new * x_j.T
    bi = y_i - w_new * x_i.T
    if 0 < alpha_newj< C:
        b = bj
    elif 0 < alpha_newi< C:
        b = bi
    else:
        b = (bi + bj)/2
    return b

def smo_simple(dataMat, labelMat, C, toler, maxIter):
    # this funtion calculate the alphas and b
    # w can be calculated with alphas, x(dataMat) and y(lableMat)
    dataMat = np.asmatrix(dataMat)# change array to matrix
    labelMat = np.asmatrix(labelMat).T
    m = labelMat.shape[0]  # m is the number of samples, n is the number of features
    alphas = np.asmatrix(np.zeros([m, 1])) # set alphas to be 0[m, 1]
    b = 0
    iters = 0
    while iters < maxIter:
        alphaPairchange = 0
        # loop until reaching maxIter to optimize alphas
        # print (iters)
        for i in range(m):
            # choose alpha_i index
            w = np.multiply(alphas, labelMat).T * dataMat  # get w 1*n
            Ei = float(w*dataMat[i, :].T + b - labelMat[i]) # ei is w*xi+b-yi Ei should be a number instead of a array
            # check whether alpha_i violates kkt condition
            i_break_kkt = ((float(alphas[i]) < C) and (float(Ei*labelMat[i]) < -toler)) or ( (alphas[i] > C) and (Ei*labelMat[i] > toler))
            if i_break_kkt:
                j = select_j(i, m)
                s = labelMat[i] * labelMat[j]
                Ej = float(w * dataMat[j, :].T + b - labelMat[j])
                x_i = dataMat[i, :]
                x_j = dataMat[j, :]
                y_i = labelMat[i]
                y_j = labelMat[j]
                alpha_newj, alpha_newi = cal_new_alphaij(alphas, x_i, x_j, y_i, y_j, s, i, j, C, Ei, Ej)
                alphas[j] = alpha_newj
                alphas[i] = alpha_newi  # update alpha
                w_new = np.transpose(np.multiply(alphas, labelMat)) * dataMat
                b = cal_b(alpha_newj, alpha_newi, w_new, x_i, x_j, y_i, y_j, C)  # update b
                alphaPairchange += 1
                print(alphaPairchange)
        if alphaPairchange != 0:
            iters = 0
        else:
            iters += 1
    return alphas, b

dataMat, labelMat = loadDataSet('testSet.txt')
alphas, b = smo_simple(np.array(dataMat), np.array(labelMat), 0.6, 0.001, 40)
print (alphas[alphas>0])
print (b)
# print (alphas[alphas>0])
# print (alphas)
# print (b)