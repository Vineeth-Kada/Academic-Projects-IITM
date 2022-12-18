# Common Libraries


import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal as mvn
import random
import pandas as pd
from sklearn.metrics import DetCurveDisplay
from math import sqrt
import math
import matplotlib.pyplot as plt

# K - Means

# Vineeth
def kMeans(X, K, k_iterations):
    N = len(X)
    D = X[0].shape[0]
    
    # Randomly Choose k - points as the means
    mu = [ X[k] for k in range(K) ]
    cls = [ i for i in range(N) ]
    
    for _ in range(k_iterations):
        isSame = True
        
        # Assign Data points
        for j in range(N):
            mn = float('inf')
            currCls = 0
            for k in range(K):
                currDist = np.linalg.norm(mu[k] - X[j])
                if(currDist < mn):
                    mn = currDist
                    currCls = k
            if(cls[j] != currCls):
                isSame = False
                cls[j] = currCls
        
        
        mu = [ np.zeros(D) for k in range(K) ]
        cnt = [ 0 for k in range(K) ]
        for i in range(N):
            mu[cls[i]] += X[i]
            cnt[cls[i]] += 1
        
        for i in range(K):
            mu[i] /= cnt[i]
        
        if(isSame): break
        
    return cls, mu

# GMM EM Algorithm

# Vineeth
# E Step
# Input: 
# Output: 
def e_step(pi, mu, sigma, X):
    N = len(X)
    K = len(mu)
    D = X[0].shape[0]
    
    gamma = np.zeros([N, K])
    for n in range(N):
        denom = 0.0
        for j in range(K):
            denom += pi[j] * mvn(mu[j],sigma[j], allow_singular=True).pdf(X[n])
        for k in range(K):
            gamma[n][k] = pi[k] * mvn(mu[k],sigma[k], allow_singular=True).pdf(X[n]) / denom
            
    return gamma

# M Step
# Input:
    # gamma - 2D numpy array N * K => N points, K mixtures
    # x - N length list of numpy arrays of dim([D, 1]) => N points, D features
# Output:
    # theta_new = [pi, mu, sigma]
def m_step(gamma, X):
    N = gamma.shape[0]
    K = gamma.shape[1]
    D = X[0].shape[0]
    
    # Compute Nk for every class
    Nk = np.zeros(K)
    for k in range(K):
        for n in range(N):
            Nk[k] += gamma[n][k]

    # Compute pi
    pi = np.zeros(K)
    for k in range(K):
        pi[k] = Nk[k] / N

    # Compute mu
    mu = [ np.zeros(D) for k in range(K) ]
    for k in range(K):
        for n in range(N):
            mu[k] += gamma[n][k] * X[n]
        mu[k] /= Nk[k]
        
    # Compute sigma
    sigma = [ np.matrix(np.zeros((D, D))) for k in range(K) ]
    for k in range(K):
        for n in range(N):
            sigma[k] += gamma[n][k] * (np.matrix(X[n] - mu[k]).T @ np.matrix(X[n] - mu[k]))
        sigma[k] /= Nk[k]
    
    return pi, mu, sigma

# GMM
def GMM(X, K, iterations, k_iterations):
    N = len(X)
    D = X[0].shape[0]
    
    # Initialise pi, mu, sigma using k-Means
    cls, mu = kMeans(X, K , k_iterations)
    print("K-Means Initialisation Done")
    pi = [ 0 for i in range(K) ]
    sigma = [ np.matrix(np.zeros([D, D])) for k in range(K) ]
    Nk = [ 0 for i in range(K) ]
    
    for i in range(N):
        currCls = cls[i]
        pi[currCls] += 1
        Nk[currCls] += 1
        sigma[currCls] += np.matrix(X[i] - mu[currCls]).T @ np.matrix(X[i] - mu[currCls])
        
    for i in range(K):
        pi[i] /= N
        sigma[i] /= Nk[i]
    
    for _ in range(iterations):
        # Repeat E & M steps until convergence
        gamma = e_step(pi, mu, sigma, X)
        print("E STEP")
        pi_new, mu_new, sigma_new = m_step(gamma, X)
        print("M STEP\n\n")
        
    return pi, mu, sigma

# Synthetic Data

# Hakesh


def multivariate_gaussian(x, mu, sigma):
    size = len(x)
    det = np.linalg.det(sigma)

    norm_const = 1.0/ ( math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2) )
    x_mu = np.matrix(x - mu)
    inv = sigma.I        
    result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
    return (norm_const * result)

# Convert normal matrix to diagonal matrix
def make_dcov(cov):
    dim = len(cov)
    dcov = np.matrix(np.zeros(shape = (dim,dim)))
    for i in range(dim) : dcov[i,i] = cov[i,i]
    return dcov


# Calculate likelihood probability value for given feature vector
def likelihood(fvec,pis,means,covs,K):
    prob = 0
    for i in range(K):
        prob += pis[i] * mvn(means[i],covs[i], allow_singular=True).pdf(fvec)
#        prob += pis[i] * multivariate_gaussian(fvec,means[i],covs[i])
    return prob


# Hakesh
# Plotting contour-plots of all GMM's, decision surface for synthetic data
def contour_decision_plot(all_pis,all_means,all_covs,K,nClasses,path):
    
    # Contour plots
    for c in range(nClasses):
        for k in range(K):

            mean = all_means[c][k]
            
            count = 100
            x = np.linspace(mean[0] - 3, mean[0] + 3, count) 
            y = np.linspace(mean[1] - 3, mean[1] + 3, count)
            X,Y = np.meshgrid(x,y)
    
            cov = all_covs[c][k]

            Z = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i,j] = mvn(mean,cov, allow_singular=True).pdf(np.array([X[i,j],Y[i,j]]))
            plt.scatter([mean[0]], [mean[1]])
            plt.contour(X, Y, Z, levels = [0.2, 0.3, 0.4, 0.5, 0.6], cmap = plt.get_cmap('magma'))
    
    # Decision Surface
    # Assuming nClasses := 2
    count = 100
    x = np.linspace(-16, -1, count) 
    y = np.linspace(-10, 6, count)
    X,Y = np.meshgrid(x,y)
    
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i,j]
            y = Y[i,j]

            fvec = np.array([x,y])
            p1 = likelihood(fvec,all_pis[0],all_means[0],all_covs[0],K)
            p2 = likelihood(fvec,all_pis[1],all_means[1],all_covs[1],K)
            
            if(p1 >= p2) : Z[i,j] = 1
            else : Z[i,j] = 2
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axix')
    plt.title(f'Decision and Contour plot for K = {K}')
    plt.contourf(X,Y,Z,cmap=plt.cm.jet, alpha=.5)   
    plt.savefig(path,dpi = 150)
#     plt.show()
    
# ROC Plotting
def ROC(likelihood, prior, nTests, nClasses, Groundtruth):
    S = [[likelihood[j][i] * prior[j] for j in range(nClasses)] for i in range(nTests)]
    
    for i in range(nTests):
        sum = 0
        for j in range(nClasses):
            sum += S[i][j]
        for j in range(nClasses):
            S[i][j] = S[i][j] / sum

    TPR = []; FPR = []
    for threshold in np.linspace(np.amin(S), np.amax(S), 100):
        TP = FP = TN = FN = 0.0
        for i in range(nTests):
            for j in range(nClasses):
                if(S[i][j] >= threshold):
                    if Groundtruth[i] == j:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if Groundtruth[i] == j:
                        FN += 1
                    else:
                        TN += 1
        TPR.append(TP/(TP + FN))
        FPR.append(FP/(FP + TN))

    FPR, TPR = zip(*sorted(zip(FPR, TPR)))
    plt.plot(FPR, TPR)
    return np.trapz(TPR, FPR)



def DET(likelihood, prior, nTests, nClasses, Groundtruth):
    S = [[likelihood[i][j] * prior[j] for j in range(nClasses)] for i in range(nTests)]
    
#     for i in range(nTests):
#         sum = 0
#         for j in range(nClasses):
#             sum += S[i][j]
#         for j in range(nClasses):
#             S[i][j] = S[i][j] / sum

    l = []
    for i in range(nTests):
        for j in range(nClasses):
            l.append(S[i][j])
    l = list(set(l))

    FNR = []; FPR = []
#     for threshold in np.linspace(np.amin(S), np.amax(S), 100):
    for threshold in l:
        TP = FP = TN = FN = 0.0
        for i in range(nTests):
            for j in range(nClasses):
                if(S[i][j] >= threshold):
                    if Groundtruth[i] == j:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if Groundtruth[i] == j:
                        FN += 1
                    else:
                        TN += 1
        FNR.append(FN/(TP + FN))
        FPR.append(FP/(FP + TN))

    return FPR, FNR

# Hakesh
X = []
N = []
nClasses = 0

with open('Synthetic/train.txt') as train_syn:
    lines = train_syn.readlines()
    
    for line in lines:
        _, _, c = line.strip().split(',')
        nClasses = max(nClasses, int(c))
        
    N = [ 0 for _ in range(len(lines)) ]
    X = [ [] for _ in range(nClasses) ]
    for line in lines:
        x, y, c = line.strip().split(',')
        x, y, c = float(x), float(y), int(c) - 1
        X[c].append(np.array([x, y]))
        N[c] += 1
        
"""Scatter Plot visualization of training data """      
for c in range(nClasses):
    x = []
    y = []
    for a,b in X[c]:
        x.append(a)
        y.append(b)
    plt.scatter(x,y)
plt.show()

X_dev = []
groundtruth_dev = []
nTests = 0
N_dev = []
nClasses_dev = 0

with open('Synthetic/dev.txt') as dev_syn:
    lines = dev_syn.readlines()
    
    for line in lines:
        _, _, c = line.strip().split(',')
        nClasses_dev = max(nClasses_dev, int(c))
        
    N_dev = [0 for c in range(nClasses_dev)]
    for line in lines:
        nTests += 1
        x, y, c = line.strip().split(',')
        x, y, c = float(x), float(y), int(c) - 1
        X_dev.append(np.array([x, y]))
        groundtruth_dev.append(c)
        N_dev[c] += 1

"""Scatter Plot visualization of developement data """      
for c in range(nClasses_dev):
    x = []
    y = []
    for [a,b] in X_dev:
        x.append(a)
        y.append(b)
    plt.scatter(x,y)
plt.show()





# Hakesh
"""WARNING : RUNNING OF THIS CELL WILL TAKE TOO LONG TIME """
""" Main part of program """
gmm_all_pis = []
gmm_all_means = []
gmm_all_covs = []
gmm_all_dcovs = []
K_list = [5,11,16,25]

for K in K_list:
    all_pis = []
    all_means = []
    all_covs = []
    all_dcovs = []

    for c in range(nClasses):
        pis,means,covs = GMM(X[c],K,4,20)
        dcovs = []
        for cov in covs:dcovs.append(make_dcov(cov))
        all_pis.append(pis)
        all_means.append(means)
        all_covs.append(covs)
        all_dcovs.append(dcovs)

    gmm_all_pis.append(all_pis)
    gmm_all_means.append(all_means)
    gmm_all_covs.append(all_covs)
    gmm_all_dcovs.append(all_dcovs)
    
    

# Hakesh
""" ROC PLOTS """

# Directory to save images
dir_path = 'partA_images/synthetic_png/'

leg = []
for idx in range(len(K_list)):
    K = K_list[idx]

    
    all_pis = gmm_all_pis[idx]
    all_means = gmm_all_means[idx]
    all_covs = gmm_all_covs[idx]
    
    
    
    all_likelihood_dev = [[] for c in range(nClasses)]
    all_priors_dev = [0.5,0.5]
    for c in range(nClasses):
        for [x,y] in X_dev:
            fvec = np.array([x,y])
            all_likelihood_dev[c].append(likelihood(fvec,all_pis[c],all_means[c],all_covs[c],K))
            
    area = ROC(all_likelihood_dev,all_priors_dev,nTests,nClasses_dev,groundtruth_dev)
    leg.append(f'Area of ROC for K = {K} is : {area}')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curves for GMM with different K')
plt.legend(leg)

path = dir_path + 'ROC.svg'
plt.savefig(path)
plt.show()
plt.clf()


plt.clf()
################################################################################################################
"""DET PLOT"""

# def DET(likelihood, prior, nTests, nClasses, Groundtruth):

#     S = [[likelihood[i][j] * prior[j] for j in range(nClasses)] for i in range(nTests)]
    
#     for i in range(nTests):
#         sum = 0
#         for j in range(nClasses):
#             sum += S[i][j]
#         for j in range(nClasses):
#             S[i][j] = S[i][j] / sum

#     FNR = []; FPR = []
#     for threshold in np.linspace(np.amin(S), np.amax(S), 100):
#         TP = FP = TN = FN = 0.0
#         for i in range(nTests):
#             for j in range(nClasses):
#                 if(S[i][j] >= threshold):
#                     if Groundtruth[i] == j:
#                         TP += 1
#                     else:
#                         FP += 1
#                 else:
#                     if Groundtruth[i] == j:
#                         FN += 1
#                     else:
#                         TN += 1
#         FNR.append(FN/(TP + FN))
#         FPR.append(FP/(FP + TN))

#     return FPR, FNR


ax = plt.gca()
for idx in range(len(K_list)):
    K = K_list[idx]

    
    all_pis = gmm_all_pis[idx]
    all_means = gmm_all_means[idx]
    all_covs = gmm_all_covs[idx]
    
    
    
    all_likelihood_dev = [[0 for c in range(nClasses)] for n in range(nTests)]
    all_priors_dev = [0.5,0.5]
    for c in range(nClasses):
        for i in range(len(X_dev)):
            x = X_dev[i][0]
            y = X_dev[i][1]
            fvec = np.array([x,y])
            all_likelihood_dev[i][c] = likelihood(fvec,all_pis[c],all_means[c],all_covs[c],K)
            
    FPR,FNR = DET(all_likelihood_dev,all_priors_dev,nTests,nClasses_dev,groundtruth_dev)
    DetCurveDisplay(fpr = FPR, fnr = FNR, estimator_name = 'K = ' + str(K)).plot(ax)
#     leg.append(f'Area of DET for K = {K} is : {area}')
plt.xlabel('FPR')
plt.ylabel('FNR')
plt.title('DET Curves for GMM with different K')
# plt.legend(leg)

path = dir_path + 'DET.svg'
plt.savefig(path)
plt.show()
plt.clf()

# Hakesh 
"""DECISION-CONTOUR PLOTS"""
for idx in range(len(K_list)):
    K = K_list[idx]
    
    all_pis = gmm_all_pis[idx]
    all_means = gmm_all_means[idx]
    all_covs = gmm_all_covs[idx]
    
    path = dir_path + 'K' + str(K) + '_contour_decision.png'
    contour_decision_plot(all_pis, all_means, all_covs, K, nClasses,path)
    plt.clf()
    



# # Hakesh
# """To see scatter plot of means of different classes (for syn data only)"""
# """For some particular K only ( gmm_all_means[0] represented K)"""
# x1 = []
# x2 = []
# y2 = []
# y1 = []

# for i in range(len(gmm_all_means[0])):
#     x1.append(gmm_all_means[0][0][i][0])
#     y1.append(gmm_all_means[0][0][i][1])
    
# for i in range(len(all_means[1])):
#     x2.append(gmm_all_means[0][1][i][0])
#     y2.append(gmm_all_means[0][1][i][1])

# plt.scatter(x1,y1)
# plt.scatter(x2,y2)
# plt.show()




# Hakesh
""" Printing/Saving plots for diagonal covariance matrix case """


# Hakesh
""" ROC PLOTS """

leg = []
for idx in range(len(K_list)):
    K = K_list[idx]
    
    all_pis = gmm_all_pis[idx]
    all_means = gmm_all_means[idx]
    all_dcovs = gmm_all_dcovs[idx]
    
    all_likelihood_dev = [[] for c in range(nClasses)]
    all_priors_dev = [0.5,0.5]
    for c in range(nClasses):
        for [x,y] in X_dev:
            fvec = np.array([x,y])
            all_likelihood_dev[c].append(likelihood(fvec,all_pis[c],all_means[c],all_dcovs[c],K))
    area = ROC(all_likelihood_dev,all_priors_dev,nTests,nClasses_dev,groundtruth_dev)
    leg.append(f'Area of ROC for K = {K} is : {area}')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curves for Diagonal covariance GMM with different K')
plt.legend(leg)

path = dir_path + 'diag_ROC.svg'
plt.savefig(path)
plt.show()
plt.clf()

#######################################

ax = plt.gca()
for idx in range(len(K_list)):
    K = K_list[idx]

    
    all_pis = gmm_all_pis[idx]
    all_means = gmm_all_means[idx]
    all_dcovs = gmm_all_dcovs[idx]
    
    
    
    all_likelihood_dev = [[0 for c in range(nClasses)] for n in range(nTests)]
    all_priors_dev = [0.5,0.5]
    for c in range(nClasses):
        for i in range(len(X_dev)):
            x = X_dev[i][0]
            y = X_dev[i][1]
            fvec = np.array([x,y])
            all_likelihood_dev[i][c] = likelihood(fvec,all_pis[c],all_means[c],all_dcovs[c],K)
            
    FPR,FNR = DET(all_likelihood_dev,all_priors_dev,nTests,nClasses_dev,groundtruth_dev)
    DetCurveDisplay(fpr = FPR, fnr = FNR, estimator_name = 'K = ' + str(K)).plot(ax)
#     leg.append(f'Area of DET for K = {K} is : {area}')
plt.xlabel('FPR')
plt.ylabel('FNR')
plt.title('DET Curves for Diagonal Covariances GMM with different K')
# plt.legend(leg)

path = dir_path + 'diag_DET.svg'
plt.savefig(path)
plt.show()
plt.clf()

######################################


"""DECISION-CONTOUR PLOTS"""
for idx in range(len(K_list)):
    K = K_list[idx]
    
    all_pis = gmm_all_pis[idx]
    all_means = gmm_all_means[idx]
    all_dcovs = gmm_all_dcovs[idx]
    
    path = dir_path + 'diag_K' + str(K) + '_contour_decision.png'
    contour_decision_plot(all_pis, all_means, all_dcovs, K, nClasses,path)
    plt.clf()


# Vineeth
# print(all_covs)

# # Vineeth
# all_likelihood_dev = [[] for c in range(nClasses)]
# all_priors_dev = [0.5,0.5]
# for c in range(nClasses):
#     for [x,y] in X_dev:
#         fvec = np.array([x,y])
#         all_likelihood_dev[c].append(likelihood(fvec,all_pis[c],all_means[c],all_covs[c],K))
# ROC(all_likelihood_dev,all_priors_dev,nTests,nClasses_dev,groundtruth_dev)

# Vineeth

# Real Data

# Hakesh

# Hakesh

# Hakesh

# Vineeth

# Vineeth

# Vineeth