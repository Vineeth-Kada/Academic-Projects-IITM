import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay, ConfusionMatrixDisplay, confusion_matrix
from math import sqrt
import math
import time
import os
import random
from numba import njit
import numba as nb

# K - Means
@njit
def kMeans(X, K, k_iterations):
    N = len(X)
    D = X[0].shape[0]
    
    mu = [ X[i] for i in range(K) ]
    cls = [ i for i in range(N) ]
    
    for _ in range(k_iterations):

        # Assign Data points
        for j in range(N):
            mn = 0.0
            currCls = 0
            for k in range(K):
                currDist = np.linalg.norm(mu[k] - X[j])
                if(k == 0 or currDist < mn):
                    mn = currDist
                    currCls = k
            cls[j] = currCls
        
        mu = [ np.zeros(D, dtype = np.float64) for k in range(K) ]
        cnt = [ 0 for k in range(K) ]
        for i in range(N):
            mu[cls[i]] += X[i]
            cnt[cls[i]] += 1
        
        for i in range(K):
            mu[i] /= cnt[i]
        
    return cls, mu


# GMM EM Algorithm
@njit
def my_mvn(mu, sigma, X):
    D = sigma.shape[0]
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    
    const = 1.0 / ((2.0 * math.pi) ** (D/2.0)) / (det ** 0.5)
    exp = np.exp(-0.5 * ((X-mu).T @ inv @ (X-mu)))

    return const * exp

# E Step
# Output: gamma
@njit
def e_step(pi, mu, sigma, X):
    N = len(X)
    K = len(mu)
    D = X[0].shape[0]
    
    gamma = np.zeros((N, K), dtype = np.float64)
    for n in range(N):
        denom = 0.0
        for j in range(K):
            denom += pi[j] * my_mvn(mu[j],sigma[j], X[n])
        for k in range(K):
            gamma[n][k] = pi[k] * my_mvn(mu[k],sigma[k],X[n]) / denom
            
    return gamma

# M Step
# Input:
    # gamma - 2D numpy array N * K => N points, K mixtures
    # x - N length list of numpy arrays of dim([D, 1]) => N points, D features
# Output:
    # theta_new = [pi, mu, sigma]
@njit
def m_step(gamma, X):
    N = gamma.shape[0]
    K = gamma.shape[1]
    D = X[0].shape[0]
    
    # Compute Nk for every class
    Nk = np.zeros(K, dtype = np.float64)
    for k in range(K):
        Nk[k] = np.sum(gamma[:, k])

    # Compute pi
    pi = Nk / N

    # Compute mu
    mu = [ np.zeros(D, dtype = np.float64) for k in range(K) ]
    for k in range(K):
        for n in range(N):
            mu[k] += gamma[n][k] * X[n]
        mu[k] /= Nk[k]
        
    # Compute sigma
    sigma = [ np.zeros((D, D), dtype = np.float64) for k in range(K) ]
    for k in range(K):
        for n in range(N):
            sigma[k] += gamma[n][k] * np.outer(X[n] - mu[k], X[n] - mu[k])
        sigma[k] /= Nk[k]
    
    return pi, mu, sigma

# GMM
@njit
def GMM(X, K, iterations, k_iterations):
    N = len(X)
    D = X[0].shape[0]
    
    # Initialise pi, mu, sigma using k-Means
    cls, mu = kMeans(nb.typed.List(X), K , k_iterations)
    print("K-Means Initialisation Done")
    pi = np.zeros(K, dtype = np.float64)
    sigma = [ np.zeros((D, D), dtype = np.float64) for k in range(K) ]
    Nk = np.zeros(K, dtype = np.float64)
    
    for i in range(N):
        currCls = cls[i]
        pi[currCls] += 1
        Nk[currCls] += 1
        sigma[currCls] += np.outer(X[i] - mu[currCls], X[i] - mu[currCls])
        
    for i in range(K):
        pi[i] /= N
        sigma[i] /= Nk[i]
    
    for _ in range(iterations):
        gamma = e_step(nb.typed.List(pi), mu, sigma, nb.typed.List(X))
        print("E STEP")
        
        pi_new, mu_new, sigma_new = m_step(gamma, nb.typed.List(X))
        print("M STEP")
        
    return pi, mu, sigma

# Convert normal matrix to diagonal matrix
def make_dcov(cov):
    dim = len(cov)
    dcov = np.matrix(np.zeros(shape = (dim,dim)))
    for i in range(dim) : dcov[i,i] = cov[i,i]
    return dcov

# Data Extraction from files
def extractImgData(dir):
    files = os.listdir(dir)
    images = []
    for file in files:
        with open(dir + file, 'r') as f:
            blocks = []
            for line in f.readlines():
                block = np.array(line.split(), dtype=np.float64)
                blocks.append(block)
            images.append(blocks)
    return images


# Calculate likelihood probability value for given image
def log_likelihood(image,pis,means,covs,K):
    log_likelihood = 0.0
    for block in image:
        prob_block = 0.0
        for i in range(K):
            mvn_out = my_mvn(means[i],covs[i], block)
            prob_block += pis[i] * mvn_out
        log_likelihood += np.log(prob_block + np.exp(-500))
    return log_likelihood


exec(open("../../sharedFunctions.py").read())

trainData = []
classList = ['coast', 'forest', 'highway', 'mountain', 'opencountry']
nClasses_Real = len(classList)

for cls in classList:
    images = extractImgData('RealData/' + cls + '/train/')
    trainData.append(images)
    
# Normalisation
# Step1: Find Mean
allBlocks = []
Dim = 23
mean = np.zeros(23, dtype = np.float64)
cnt_train1 = 0
for i in range(nClasses_Real):
    for image in trainData[i]:
        for block in image:
            mean += block
            cnt_train1 += 1
            allBlocks.append(block)
mean /= cnt_train1

# Step2: Subtract Mean & Find range
mn = np.array([float('inf') for i in range(Dim)])
mx = np.array([float('-inf') for i in range(Dim)])
for block in allBlocks:
    block = block - mean
    mn = np.minimum(mn, block)
    mx = np.maximum(mx, block)

# Step3: Modify original data
mult_factor = 100
for cls in range(len(classList)):
    for img_idx in range(len(trainData[cls])):
        for blk_idx in range(len(trainData[cls][img_idx])):
            trainData[cls][img_idx][blk_idx] -= mean
            trainData[cls][img_idx][blk_idx] /= (mx - mn) / mult_factor

X_dev = []
nTests = 0
groundtruth_dev = []
for i in range(len(classList)):
    cls = classList[i]
    images = extractImgData('RealData/' + cls + '/dev/')
    for image in images:
        for blk_idx in range(len(image)):
            image[blk_idx] -= mean
            image[blk_idx] /= (mx - mn) / mult_factor
        nTests += 1
        X_dev.append(image)
        groundtruth_dev.append(i)

gmm_all_pis = []
gmm_all_means = []
gmm_all_covs = []
gmm_all_dcovs = []

K_list = [5, 12, 16]
for K in K_list:
    all_pis = []
    all_means = []
    all_covs = []
    all_dcovs = []

    for c in range(nClasses_Real):
        X = []
        for image in trainData[c]:
            for block in image:
                X.append(block)
        pis,means,covs = GMM(nb.typed.List(X),K,10,50)
#         pis,means,covs = GMM(nb.typed.List(X),K,7,10)
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

dir_path = 'RealData_Results/10_50_5.12.16_RD'


""" Confusion Matrix """
for idx in range(len(K_list)):
    K = K_list[idx]

    all_pis = gmm_all_pis[idx]
    all_means = gmm_all_means[idx]
    all_covs = gmm_all_covs[idx]
    all_dcovs = gmm_all_dcovs[idx]
    
    pred_class = []
    cnt, accuracy = 0, 0
    for image in X_dev:
        LL = np.zeros(nClasses_Real)
        for c in range(nClasses_Real):
            LL[c] = log_likelihood(image,all_pis[c],all_means[c],all_covs[c],K)
        trueCls, predCls = groundtruth_dev[cnt], np.argmax(LL)
        pred_class.append(predCls)
        if(trueCls == predCls): accuracy += 1
        cnt += 1

    ConfusionMatrixDisplay.from_predictions(groundtruth_dev, pred_class, display_labels=classList, xticks_rotation=45).plot()
    plt.title('Confusion Matrix for K = ' + str(K))
    plt.savefig(dir_path + 'CMat' + str(K) + '.svg')
    plt.clf()
    print('*' * 25); print(accuracy / cnt * 100); print('*' * 25);
    
    pred_class = []
    cnt, accuracy = 0, 0
    for image in X_dev:
        LL = np.zeros(nClasses_Real)
        for c in range(nClasses_Real):
            LL[c] = log_likelihood(image,all_pis[c],all_means[c],all_dcovs[c],K)
        trueCls, predCls = groundtruth_dev[cnt], np.argmax(LL)
        pred_class.append(predCls)
        if(trueCls == predCls): accuracy += 1
        cnt += 1

    ConfusionMatrixDisplay.from_predictions(groundtruth_dev, pred_class, display_labels=classList, xticks_rotation=45).plot()
    plt.title('Confusion Matrix for Diagonal Cov. K = ' + str(K))
    plt.savefig(dir_path + 'Diag_CMat' + str(K) + '.svg')
    plt.clf()
    print('*' * 25); print(accuracy / cnt * 100); print('*' * 25);

""" ROC PLOTS """
ROC_legend = []
for idx in range(len(K_list)):
    K = K_list[idx]

    all_pis = gmm_all_pis[idx]
    all_means = gmm_all_means[idx]
    all_covs = gmm_all_covs[idx]
    all_dcovs = gmm_all_dcovs[idx]
    
    all_likelihood_dev = [[] for c in range(nClasses_Real)]
    all_priors_dev = [1 / nClasses_Real for c in range(nClasses_Real)]
    for c in range(nClasses_Real):
        for image in X_dev:
            all_likelihood_dev[c].append(log_likelihood(image,all_pis[c],all_means[c],all_covs[c],K))
    ROC(all_likelihood_dev,all_priors_dev,nTests,nClasses_Real,groundtruth_dev)
    ROC_legend.append('Normal K=' + str(K))
    
    all_likelihood_dev = [[] for c in range(nClasses_Real)]
    all_priors_dev = [1 / nClasses_Real for c in range(nClasses_Real)]
    for c in range(nClasses_Real):
        for image in X_dev:
            all_likelihood_dev[c].append(log_likelihood(image,all_pis[c],all_means[c],all_dcovs[c],K))
    ROC(all_likelihood_dev,all_priors_dev,nTests,nClasses_Real,groundtruth_dev)
    ROC_legend.append('Diagonal K=' + str(K))

plt.xlabel('FPR'); plt.ylabel('TPR')
plt.legend(ROC_legend)
plt.title('ROC Curves for GMM with different K')
plt.savefig(dir_path + 'ROC.svg')
plt.clf()

ax = plt.gca()
for idx in range(len(K_list)):
    K = K_list[idx]

    all_pis = gmm_all_pis[idx]
    all_means = gmm_all_means[idx]
    all_covs = gmm_all_covs[idx]
    all_dcovs = gmm_all_dcovs[idx]
    
    all_likelihood_dev = [[] for n in range(nClasses_Real)]
    all_priors_dev = [1 / nClasses_Real for c in range(nClasses_Real)]
    for c in range(nClasses_Real):
        for image in X_dev:
            all_likelihood_dev[c].append(log_likelihood(image,all_pis[c],all_means[c],all_covs[c],K))
    FPR,FNR = DET(all_likelihood_dev,all_priors_dev,nTests,nClasses_Real,groundtruth_dev)
    DetCurveDisplay(fpr = FPR, fnr = FNR, estimator_name = 'Normal K=' + str(K)).plot(ax)
    
    all_likelihood_dev = [[] for c in range(nClasses_Real)]
    all_priors_dev = [1 / nClasses_Real for c in range(nClasses_Real)]
    for c in range(nClasses_Real):
        for image in X_dev:
            all_likelihood_dev[c].append(log_likelihood(image,all_pis[c],all_means[c],all_dcovs[c],K))
    FPR,FNR = DET(all_likelihood_dev,all_priors_dev,nTests,nClasses_Real,groundtruth_dev)
    DetCurveDisplay(fpr = FPR, fnr = FNR, estimator_name = 'Diagonal K=' + str(K)).plot(ax)

plt.title('DET Curves for GMM with different K')
plt.savefig(dir_path + 'DET.svg')
plt.clf()