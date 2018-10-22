from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import time
import multiprocessing as mul
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
D,Dist_matrix,K,M,N = 1,1,1,1,1
X,Y_in,Y_out = 1,1,1
W_in, W_out = 1,1
P,Q = 1,1
l1 = 1
E,U,G,F,V,H,R,S = 1, 1, 1, 1, 1, 1, 1, 1
K_begin,K_end = 1,1
rho, theta = 1, 1
L_sp, L_te = 1, 1
ORI_Y_in, ORI_Y_out = 1,1
loss_df = -1


l0 = 10.0
l1 = 5.0
l2 = 500.0
l3 = 10.0
l4 = 5.0
l5 = 500.0
l6 = 0.1
l7 = 0.1

def g_dist_f(i, j, dist, exponent=1):
    dist = dist[i][j]
    return max(dist, 1) ** (-exponent)


df_train = pd.read_csv('train_data.csv')

f = open('cluster_new.txt')
lines = f.readlines()
grid_cluster_dict = {}
cluster_dict = {}
for index, line in enumerate(lines):
    line = line.strip()
    line = line.replace('[', '')
    line = line.replace(']', '')
    line = line.split(', ')
    line = map(int, line)
    cluster_dict[index] = []
    for l in line:
        grid_cluster_dict[l] = index
        cluster_dict[index].append(l)


def cluster_check(grid):
    cluster = grid_cluster_dict[grid]
    return cluster_dict[cluster]


def sgn(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def init(reset_sign=False):
    # TODO: To normalize the Y_in, Y_out with Y_in/300 and Y_out/300
    global D, Dist_matrix, K, M, N
    global X, Y_in, Y_out
    global W_in, W_out
    global P, Q
    global l1
    global E, U, G, F, V, H, R, S
    global K_begin, K_end
    global rho
    global L_sp, L_te
    global ORI_Y_in, ORI_Y_out
    global loss_df
    print('init begin!!')
    time_list = []
    time_list.append(time.time())
    N = 784
    rho = float(1)
    D = 2
    M = df_train.shape[1] - 4
    if reset_sign == True:
        X = np.zeros((N, int(max(df_train.iloc[:, 1]) + 1), df_train.shape[1] - 4))
        Y_in = np.zeros((N, int(max(df_train.iloc[:, 1]) + 1), 1))
        Y_out = np.zeros((N, int(max(df_train.iloc[:, 1]) + 1), 1))
        for row in df_train.iterrows():
            X[int(row[1][0]), int(row[1][1])] = list(row[1][4:])
            Y_in[int(row[1][0]), int(row[1][1])] = int(row[1][2])
            Y_out[int(row[1][0]), int(row[1][1])] = int(row[1][3])
        ORI_Y_in = Y_in
        Y_in = Y_in / 300.0
        ORI_Y_out = Y_out
        Y_out = Y_out / 300.0
        X.tofile('X_np')
        Y_in.tofile('Y_in_np')
        Y_out.tofile('Y_out_np')
        ORI_Y_in.tofile('ORI_Y_in_np')
        ORI_Y_out.tofile('ORI_Y_out_np')
    else:
        X = np.fromfile('X_np').reshape(784, 240, 29)
        Y_in = np.fromfile('Y_in_np').reshape(784, 240, 1)
        Y_out = np.fromfile('Y_out_np').reshape(784, 240, 1)
        ORI_Y_in = np.fromfile('ORI_Y_in_np').reshape(784, 240, 1)
        ORI_Y_out = np.fromfile('ORI_Y_out_np').reshape(784, 240, 1)
    time_list.append(time.time())
    if reset_sign == True:
        W_in = np.random.rand(
            N, int(max(df_train.iloc[:, 1]) + 1), df_train.shape[1] - 4)
        W_out = np.random.rand(
            N, int(max(df_train.iloc[:, 1]) + 1), df_train.shape[1] - 4)
        for i in range(N):
            for j in range(int(max(df_train.iloc[:, 1]) + 1)):
                W_in[i][j] = [-1.01488813e-17, 9.50855590e-04, -9.78103650e-05,
                              5.20031761e-05, 8.96910818e-05, -2.60807921e-05,
                              1.11059862e-04, -6.07489850e-05, -4.31857881e-05,
                              2.86565637e-05, 1.36402575e-02, 1.58184137e-02,
                              -1.41914835e-02, 1.59217651e-03, -1.01809838e-03,
                              -1.13592072e-03, 2.98911799e-03, 3.06094257e-03,
                              -1.19614883e-03, 3.93401279e-03, -3.17651344e-03,
                              4.95637065e-03, 9.83877502e-04, 6.24353810e-03,
                              -3.92628967e-03, 6.41184129e-03, -2.30418121e-03,
                              -8.95705528e-03, 1.55242926e-03]
                W_out[i][j] = [-1.02584070e-17, -3.56129698e-04, 9.48073633e-06,
                               1.62879757e-04, -1.19515651e-05, -1.60936782e-04,
                               1.21312146e-04, 4.16144599e-05, 2.02909229e-04,
                               -2.32386746e-04, 1.32823465e-02, 1.58613484e-02,
                               -1.40722873e-02, 1.96989954e-03, -3.98221776e-04,
                               -1.29743159e-03, 3.27079896e-03, 3.62427427e-03,
                               -8.71975869e-05, 4.02984490e-03, -3.10176803e-03,
                               4.76147996e-03, 7.05186918e-04, 5.99663461e-03,
                               -4.13943890e-03, 6.12176380e-03, -3.23225780e-03,
                               -8.63732297e-03, 1.42380963e-03]
        W_in.tofile('W_in_np')
        W_out.tofile('W_out_np')
    else:
        W_in = np.fromfile('W_in_np').reshape(784, 240, 29)
        W_out = np.fromfile('W_out_np').reshape(784, 240, 29)
    time_list.append(time.time())
    Dist_matrix = np.asarray(pd.read_csv('grid_distance.csv', header=None))
    #     print(Dist_matrix.shape)
    K = D * 24
    K_begin = 0
    K_end = K_begin + K
    E = np.zeros((K, M, M))
    U = np.zeros((K, M, M))
    G = np.zeros((K, M, M))
    S = np.zeros((K, M, M))
    F = np.zeros((N, M, M))
    V = np.zeros((N, M, M))
    H = np.zeros((N, M, M))
    R = np.zeros((N, M, M))
    Z_te = np.zeros((K, K))
    D_te = np.zeros((K, K))
    for d in range(D - 1):
        for h in range(24):
            Z_te[(d + 1) * 24 + h, d * 24 + h] = 1
            Z_te[d * 24 + h, (d + 1) * 24 + h] = 1
    for i in range(K):
        for j in range(K):
            D_te[i, i] += Z_te[i, j]
    L_te = D_te - Z_te
    D_sp = np.zeros((N, N))
    Z_sp = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D_sp[i, i] += Dist_matrix[i, j]
            Z_sp[i, j] = Dist_matrix[i, j]
    L_sp = D_sp - Z_sp
    time_list.append(time.time())
    loss_df_columns = ['epoch', 'Total loss', 'In main loss', 'In spacial',
                       'In tempral', 'Out main loss', 'Out spacial', 'Out tempral',
                       'Unbalance loss', '2-norm loss', 'in mse', 'out mse', 'grad'
                       ]
    loss_df = pd.DataFrame(columns=loss_df_columns)
    task_list = ['Loading X,Y', 'Initing W', 'Initing EFG...']
    for i in range(len(time_list) - 1):
        print('%s using %.2f s' % (task_list[i], time_list[i + 1] - time_list[i]))


def norm1(x):
    return np.linalg.norm(x, ord=1)


def norm2(x):
    return np.linalg.norm(x)


def loss_compute():
    time_list = []
    time_list.append(time.time())
    l0_loss = 0
    # check in loss
    for k in range(K_begin, K_end):
        for n in range(N):
            l0_loss += float(
                np.dot(X[n, k], W_in[n][k]) - Y_in[n][k]) ** 2
    time_list.append(time.time())
    l1_loss = 0
    # check in spatial
    for k in range(K):
        l1_loss += 0.5 * E[k].trace()
        l1_loss += rho / 2 * norm2(
            np.dot(np.dot(W_in[:, k].T, L_sp), W_in[:, k]) - E[k] + U[k])
    time_list.append(time.time())
    l2_loss = 0
    # check in tempral
    for n in range(N):
        l2_loss += F[n].trace()
        l2_loss += rho / 2 * norm2(
            np.dot(np.dot(W_in[n, K_begin:K_end].T, L_te), W_in[n, K_begin:K_end])
            - F[n] + V[n])
    time_list.append(time.time())
    l3_loss = 0
    for k in range(K_begin, K_end):
        for n in range(N):
            l3_loss += float(
                np.dot(X[n, k], W_out[n][k]) - Y_out[n][k]) ** 2
    time_list.append(time.time())
    l4_loss = 0
    for k in range(K_begin, K_end):
        l4_loss += 0.5 * G[k].trace()
        l4_loss += rho / 2 * norm2(
            np.dot(np.dot(W_out[:, k].T, L_sp), W_out[:, k]) - G[k] + S[k])
    time_list.append(time.time())
    l5_loss = 0
    for n in range(N):
        l5_loss += H[n].trace()
        l5_loss += rho / 2 * norm2(
            np.dot(np.dot(W_out[n, K_begin:K_end].T, L_te), W_out[n, K_begin:K_end])
            - H[n] + R[n])
    time_list.append(time.time())
    l6_loss = 0
    for k in range(K_begin, K_end):
        for c in range(35):
            single_ub = 0
            for grid in cluster_dict[c]:
                single_ub += np.dot(
                    X[grid, k], W_in[grid, k] - W_out[grid, k])
            l6_loss += np.abs(single_ub)
    time_list.append(time.time())
    l7_loss = 0
    for n in range(N):
        for k in range(K_begin, K_end):
            l7_loss += norm2(W_in[n][k])
            l7_loss += norm2(W_out[n][k])
    l7_loss *= theta
    time_list.append(time.time())
    loss = np.sum([
        l0 * l0_loss, l1 * l1_loss, l2 * l2_loss,
        l3 * l3_loss, l4 * l4_loss, l5 * l5_loss,
        l6 * l6_loss, l7 * l7_loss])
    #     for i in range(len(time_list) -1):
    #         print('l%d cousume %.2f s' %(i, time_list[i+1]-time_list[i]))
    return (loss, l0 * l0_loss, l1 * l1_loss, l2 * l2_loss,
            l3 * l3_loss, l4 * l4_loss, l5 * l5_loss,
            l6 * l6_loss, l7 * l7_loss)


def grad_W_in_para(l):
    n = l[0]
    k = l[1]
    # grad = W_in[n, k] - W_in[n, k]
    l0_grad = 2 * np.dot(float(np.dot(X[n][k], W_in[n][k]) - Y_in[n][k]), X[n][k].T)
    WLW_temp = np.dot(np.dot(W_in[:, k].T, L_sp), W_in[:, k])
    l1_grad = rho * np.dot(
        np.dot(WLW_temp - E[k] + U[k], W_in[:, k].T), L_sp)[:, n]
    l1_grad += rho * np.dot(
        np.dot(WLW_temp - E[k].T + U[k].T, W_in[:, k].T), L_sp)[:, n]
    WLW_temp = np.dot(np.dot(W_in[n, K_begin:K_end].T, L_te), W_in[n, K_begin:K_end])
    l2_grad = rho * np.dot(
        np.dot(WLW_temp - F[n] + V[n], W_in[n, K_begin:K_end].T), L_te)[:, k]
    l2_grad += rho * np.dot(
        np.dot(WLW_temp - F[n].T + V[n].T, W_in[n, K_begin:K_end].T), L_te)[:, k]
    l7_grad = 2 * theta * W_in[n, k]
    clu_set = cluster_check(n)
    cluster_unbal = 0
    for i in clu_set:
        cluster_unbal += float(np.dot(
            X[i, k], W_in[i, k] - W_out[i, k]))
    l6_grad = sgn(cluster_unbal) * X[n][k].T
    grad = (
        l0 * l0_grad + l1 * l1_grad +
        l2 * l2_grad + l7 * l7_grad + l6 * l6_grad
    )
    #     grad = l0*l0_grad + l7*l7_grad
    return grad


def grad_W_out_para(l):
    n = l[0]
    k = l[1]
    # grad = W_in[n, k] - W_in[n, k]
    l3_grad = 2 * np.dot(float(np.dot(X[n][k], W_out[n][k]) - Y_out[n][k]), X[n][k].T)
    WLW_temp = np.dot(np.dot(W_out[:, k].T, L_sp), W_out[:, k])
    l4_grad = rho * np.dot(
        np.dot(WLW_temp - G[k] + S[k], W_out[:, k].T), L_sp)[:, n]
    l4_grad += rho * np.dot(
        np.dot(WLW_temp - G[k].T + S[k].T, W_out[:, k].T), L_sp)[:, n]
    WLW_temp = np.dot(np.dot(W_out[n, K_begin:K_end].T, L_te), W_out[n, K_begin:K_end])
    l5_grad = rho * np.dot(
        np.dot(WLW_temp - H[n] + R[n], W_out[n, K_begin:K_end].T), L_te)[:, k]
    l5_grad += rho * np.dot(
        np.dot(WLW_temp - H[n].T + R[n].T, W_out[n, K_begin:K_end].T), L_te)[:, k]
    l7_grad = 2 * theta * W_out[n, k]
    clu_set = cluster_check(n)
    cluster_unbal = 0
    for i in clu_set:
        cluster_unbal += float(np.dot(
            X[i, k], W_in[i, k] - W_out[i, k]))
    l6_grad = -sgn(cluster_unbal) * X[n][k].T
    grad = (
        l3 * l3_grad + l4 * l4_grad +
        l5 * l5_grad + l7 * l7_grad + l6 * l6_grad
    )
    #     grad = l3*l3_grad + l7*l7_grad
    return grad


import random


def sgd_mini_batch(steps, gamma_in, gamma_out, verbose=0, batch=48, loss_verbose=1):
    global W_in, W_out
    global loss_df
    print('SGD BEGIN!!!')
    begin_time = time.time()
    loss_list = []
    epo_list = []
    # pool = mul.Pool(batch)
    for j in range(K_begin, K_end):
        for i in range(N):
            epo_list.append((i, j))
    index = 0
    for s in range(steps):
        time_list = []
        time_list.append(time.time())
        #         random.shuffle(epo_list)
        gamma_mult = 1.0
        gamma_mult = 1.0 / ((s + 1) ** 0.5)
        # if s >100:
        #     gamma_mult = 10.0 / ((s+1))
        # if gamma_mult < 1.0/10:
        #     gamma_mult = 1.0/10 /np.log(s+1)
        #         if s > 50:
        #             gamma_mult = 0.5
        # if s > 50:
        #     gamma_mult = 5.0
        gamma_in_diminish = gamma_in * gamma_mult
        gamma_out_diminish = gamma_out * gamma_mult
        grad_norm_list = []
        # time_list.append(time.time())
        for b in range(0, len(epo_list), batch):
            t0 = time.time()
            gd_list = epo_list[b:min(b + batch, len(epo_list) - 1)]
            # print(b, min(b + batch, len(epo_list) - 1))
            # grad_list = pool.map(grad_W_in_para, gd_list)
            # grad_list = map(grad_W_in_para, gd_list)
            # for j,grad in enumerate(grad_list):
            #     gamma_vle = gamma_diminish
            #     grad_mean = np.max(np.abs(gamma_vle * grad))
            #     if grad_mean > 10**-4:
            #         gamma_vle = 1.0*gamma_diminish / (10**(4+np.log10(grad_mean)))
            #         # print('over')
            #         # continue
            #
            #     gd = gd_list[j]
            #     grad_norm_list.append(grad_mean)
            #     # print(grad_mean)
            #     W_in[gd[0]][gd[1]] -= gamma_vle * grad
            for i, gd in enumerate(gd_list):
                gamma_vle = gamma_in_diminish
                grad_in = grad_W_in_para(gd)
                grad_in_max = np.max(np.abs(grad_in))
                if grad_in_max > 10 ** -2:
                    gamma_vle = 1.0 * gamma_in_diminish / (10 ** (2 + np.log10(grad_in_max)))
                W_in[gd[0]][gd[1]] = W_in[gd[0]][gd[1]] - gamma_vle * grad_in
                index += 1
                #                 if index <10:
                #                     print(gamma_vle)
                #                     print(grad_in)
                #                     print(gd)
                #                     print(W_in[gd[0]][gd[1]]-W_in_bak[gd[0]][gd[1]])
                gamma_vle = gamma_out_diminish
                grad_out = grad_W_out_para(gd)
                grad_out_max = np.max(np.abs(grad_out))
                if grad_out_max > 10 ** -2:
                    gamma_vle = 1.0 * gamma_out_diminish / (10 ** (2 + np.log10(grad_out_max)))
                W_out[gd[0]][gd[1]] = W_out[gd[0]][gd[1]] - gamma_vle * grad_out
                grad_norm_list.append(max(grad_in_max, grad_out_max))
            t1 = time.time()
        time_list.append(time.time())
        loss = loss_compute()
        single_loss = [s]
        single_loss.extend(loss[:])
        in_mse, out_mse = compute_mse()
        single_loss.extend([in_mse, out_mse])
        single_loss.append(np.mean(grad_norm_list))
        loss_df.loc[loss_df.shape[0]] = single_loss
        if s % (int(10 ** verbose)) == 0:
            print('epoch %d' % s)
            print('Total Loss:%.2f' % loss[0])
            for i in range(1, len(loss), 2):
                print('L%dloss: %.2f' % (i - 1, loss[i]), end='\t')
                if i < len(loss) - 1:
                    print('L%dloss: %.2f' % (i, loss[i + 1]))
            print('Average step %.7E' % (np.mean(grad_norm_list)))
            # print(grad_norm_list)
            print('Max step %.7E' % (np.max(grad_norm_list)))
            # for i,l in loss:
            #     print('\tloss%d: %.5E' %())
            print('In MSE:%.2f \tOut MSE:%.2f' % (in_mse, out_mse))
            print('Cost %.2f s' % (time_list[1] - time_list[0]), end='\t')
            print('Total Cost %.2f s' % (time_list[1] - begin_time))
    plt.figure(figsize=(16, 9))
    for i in range(1, 10):
        plt.plot(loss_df.iloc[:, 0], loss_df.iloc[:, i], label=loss_df.columns[i])
    plt.legend()
    plt.figure()
    plt.plot(loss_df.iloc[:, 0], loss_df['grad'])
    plt.savefig('grad')


#     return loss_df


def compute_mse():
    true_in_list = []
    predict_in_list = []
    true_out_list = []
    predict_out_list = []

    def l(x):
        try:
            return x * 300.0
        except:
            print('%.2f too large for PWD' % x)
            return 0

    for n in range(N):
        for k in range(K_begin, K_end):
            predict_in_list.append(l(float(np.dot(X[n, k], W_in[n][k]))))
            predict_out_list.append(l(float(np.dot(X[n, k], W_out[n][k]))))
            true_in_list.append(float(ORI_Y_in[n, k]))
            true_out_list.append(float(ORI_Y_out[n, k]))
    in_mse = mse(true_in_list, predict_in_list)
    out_mse = mse(true_out_list, predict_out_list)
    # print('Check out predict mse: %.2f' %(in_mse))
    # print('Check in predict mse: %.2f' % (out_mse))
    return in_mse, out_mse

    # init()
    # time0 = time.time()
    # df = sgd_mini_batch(steps=300, gamma=0.00001,verbose=0,batch=8,loss_verbose=1)
    # time1 = time.time()
    # print('using %.2f s' %(time1-time0))