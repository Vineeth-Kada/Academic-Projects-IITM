# Part B: Bayes Classifier

# Common
# Libraries

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.metrics import DetCurveDisplay

# Read data given a file path
def read_data(path):
    data = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines :
            data.append(list(map(float,line.split(','))))
        f.close()
    return data

# Hakesh
# Plots : Contours , Gaussian , Decision Surfaces


# means = [mean1 , mean2 , mean3] where meani = 1 x 2 matrix
# covs = [cov1 , cov2 , cov3] where covi = covariance matrix of ith class with 2 x 2 shape
# X , Y each is a 2 x 2 matrix representing (X[i,j],Y[i,j]) grid points
# X , Y are useful for plotting only and can be random.
def gaussian_plot(means , covs , X , Y):
    ax = plt.axes(projection='3d')

    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            max_prob = 0
            for k in range(3):
                mean = means[k]
                cov = covs[k]
                max_prob = max(max_prob , multivariate_normal(mean.A1,cov).pdf([X[i,j],Y[i,j]]))
            Z[i,j] = max_prob
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,cmap=cm.viridis)
    
    ax.view_init(27, -21)

    
# data = [ [x1, x2 , class] ...] raw data (just for scatter plot)
# means = [mean1 , mean2 , mean3] where meani = 1 x 2 matrix
# covs = [cov1 , cov2 , cov3] where covi = covariance matrix of ith class with 2 x 2 shape
# X , Y each is a 2 x 2 matrix representing (X[i,j],Y[i,j]) grid points
# X , Y are useful for plotting only and can be random.
def decision_plot(data , means , covs , X , Y):
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i,j]
            y = Y[i,j]

            p1 = multivariate_normal(means[0].A1,covs[0]).pdf([x,y])
            p2 = multivariate_normal(means[1].A1,covs[1]).pdf([x,y])
            p3 = multivariate_normal(means[2].A1,covs[2]).pdf([x,y])
            
            if(p1 > p2 and p1 > p3):Z[i,j] = 1
            if(p2 > p1 and p2 > p3):Z[i,j] = 2
            if(p3 > p1 and p3 > p2) : Z[i,j] = 3

    plt.contourf(X,Y,Z)

    clr = ['green','red','blue']
    for i in range(3):
        x = []
        y = []
        for x1,x2,cls in data:
            if(cls != i+1) : continue
            x.append(x1)
            y.append(x2)
        plt.scatter(x,y,color=clr[i])

        
# means = [mean1 , mean2 , mean3] where meani = 1 x 2 matrix
# covs = [cov1 , cov2 , cov3] where covi = covariance matrix of ith class with 2 x 2 shape
# X , Y each is a 2 x 2 matrix representing (X[i,j],Y[i,j]) grid points
# X , Y are useful for plotting only and can be random.
def contour_plot(means , covs , X , Y):
    for i in range(3):
        mean = means[i]
        cov = covs[i]
        
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = multivariate_normal(mean.A1,cov).pdf([X[i,j],Y[i,j]])
        
        plt.scatter([mean[0]], [mean[1]])
        plt.contour(X, Y, Z, colors='black')
    

        eigen_values, eigen_vectors = np.linalg.eig(np.array(cov))

        origin = mean

        eig_vec1 = eigen_vectors[:,0]
        eig_vec2 = eigen_vectors[:,1]

        plt.quiver(*origin, *eig_vec1, color=['r'], scale=8)
        plt.quiver(*origin, *eig_vec2, color=['b'], scale=8)

# ROC Plotting
def ROC(likelihood, prior, nTests, nClasses, Groundtruth):
    S = [[likelihood[i][j] * prior[j] for j in range(nClasses)] for i in range(nTests)]
    
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

# DET Plotting
def DET(likelihood, prior, nTests, nClasses, Groundtruth):
    S = [[likelihood[i][j] * prior[j] for j in range(nClasses)] for i in range(nTests)]
    
    for i in range(nTests):
        sum = 0
        for j in range(nClasses):
            sum += S[i][j]
        for j in range(nClasses):
            S[i][j] = S[i][j] / sum

    FNR = []; FPR = []
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
        FNR.append(FN/(TP + FN))
        FPR.append(FP/(FP + TN))

    return FPR, FNR

# Vineeth
def ConfusionMatrix(likelihood, nTests, groundTruth):
    cmat = np.zeros((3, 3))
    for i in range(nTests):
        act = int(groundTruth[i])
        pred = int(np.argmax(likelihood[i]))
        cmat[pred, act] += 1

    rows = ['C' + str(i) for i in range(1, 4)]
    cols = ['C' + str(i) for i in range(1, 4)]
    df = pd.DataFrame(cmat, cols, rows)
    print(df)

# Hakesh

# Case1: Bayes Classifier with C same for all classes
def case1(dataType):
    data = read_data('2_' + dataType + '/train.txt')

    means = [[0.0 for j in range(2)] for i in range(3)]
    cnt = [0 for i in range(3)]
    nTrain = 0

    for line in data:
        x, y, c = line; c = int(c)-1
        cnt[c] += 1
        means[c][0] += x
        means[c][1] += y
        nTrain += 1
    
    for i in range(3):
        means[i][0] /= cnt[i]
        means[i][1] /= cnt[i]


    means2D = [np.matrix([means[i][0], means[i][1]], dtype=np.float64).T for i in range(3)]
    common_cov = np.zeros((2,2))
    for x,y,c in data:
        c = int(c)-1
        diff = np.matrix([x,y]).T - means2D[c]
        common_cov = common_cov + ((diff) @ diff.T) / nTrain
    covs = [common_cov , common_cov, common_cov]
    return means2D, covs, data

# Hakesh
# Case2: Bayes Classifier with C different for all classes
def case2(dataType):
    data = read_data('2_' + dataType + '/train.txt')

    means = [[0.0 for j in range(2)] for i in range(3)]
    cnt = [0 for i in range(3)]

    for line in data:
        x, y, c = line; c = int(c)-1
        cnt[c] += 1
        means[c][0] += x
        means[c][1] += y
    
    for i in range(3):
        means[i][0] /= cnt[i]
        means[i][1] /= cnt[i]

    covs = [np.matrix([[0.0,0.0],[0.0,0.0]], dtype=np.float64) for c in range(3)]
    means2D = [np.matrix([means[i][0], means[i][1]], dtype=np.float64).T for i in range(3)]
    
    for x,y,c in data:
        c = int(c)-1
        diff = np.matrix([x,y]).T - means2D[c]
        covs[c] = covs[c] + ((diff) @ diff.T) / cnt[c]
        
    return means2D, covs, data

# Vineeth
# Case3: Naive Bayes Classifier with C = sigma^2 . I

def case3(dataType):
    data = read_data('2_' + dataType + '/train.txt')

    means = [[0.0 for j in range(2)] for i in range(3)]
    cnt = [0 for i in range(3)]
    nTrain = 0

    for line in data:
        x, y, c = line; c = int(c)-1
        cnt[c] += 1
        means[c][0] += x
        means[c][1] += y
    
    for i in range(3):
        means[i][0] /= cnt[i]
        means[i][1] /= cnt[i]
        nTrain += cnt[i]

    var = 0.0
    for line in data:
        x, y, c = line; c = int(c)-1
        var += (x - means[c][0])**2
        var += (y - means[c][1])**2
        
    var /= 2 * nTrain

    covs = [np.matrix([[var,0.0],[0.0,var]], dtype=np.float64) for i in range(3)]
    means2D = [np.matrix([means[i][0], means[i][1]], dtype=np.float64).T for i in range(3)]
    return means2D, covs, data

# Vineeth
# Case4: Naive Bayes Classifier with C same for all classes

def case4(dataType):
    data = read_data('2_' + dataType + '/train.txt')

    means = [[0.0 for j in range(2)] for i in range(3)]
    cnt = [0 for i in range(3)]
    nTrain = 0

    for line in data:
        x, y, c = line; c = int(c)-1
        cnt[c] += 1
        means[c][0] += x
        means[c][1] += y
    
    for i in range(3):
        means[i][0] /= cnt[i]
        means[i][1] /= cnt[i]
        nTrain += cnt[i]

    var1, var2 = 0.0, 0.0
    for line in data:
        x, y, c = line; c = int(c)-1
        var1 += (x - means[c][0])**2
        var2 += (y - means[c][1])**2
        
    var1 /= nTrain
    var2 /= nTrain

    covs = [np.matrix([[var1,0.0],[0.0,var2]], dtype=np.float64) for i in range(3)]
    means2D = [np.matrix([means[i][0], means[i][1]], dtype=np.float64).T for i in range(3)]
    return means2D, covs, data

# Vineeth
# Case5: Naive Bayes Classifier with C different for all classes

def case5(dataType):
    data = read_data('2_' + dataType + '/train.txt')

    means = [[0.0 for j in range(2)] for i in range(3)]
    cnt = [0 for i in range(3)]
    nTrain = 0

    for line in data:
        x, y, c = line; c = int(c)-1
        cnt[c] += 1
        means[c][0] += x
        means[c][1] += y
    
    for i in range(3):
        means[i][0] /= cnt[i]
        means[i][1] /= cnt[i]
        nTrain += cnt[i]

    var = [[0.0 for j in range(2)] for i in range(3)]
    for line in data:
        x, y, c = line; c = int(c)-1
        var[c][0] += (x - means[c][0])**2
        var[c][1] += (y - means[c][1])**2
        
    for c in range(3):
        var[c][0] /= cnt[c]
        var[c][1] /= cnt[c]

    covs = [np.matrix([[var[c][0],0.0],[0.0,var[c][1]]], dtype=np.float64) for c in range(3)]
    means2D = [np.matrix([means[i][0], means[i][1]], dtype=np.float64).T for i in range(3)]
    return means2D, covs, data

# Multivariate Normal
def multivariate_normal_custom(mean, cov, X):
    n = len(X)
    det = np.linalg.det(2 * math.pi * cov)
    const = math.pow(det,-1.0/2)
    x_mean = np.matrix(X - mean)
    rest = math.pow(math.e, -0.5 * (x_mean * cov.I * x_mean.T))
    return const * rest

for dataType in ["LinearlySeparable", "NonLinearlySeparable", "RealData"]:
    means = [0 for i in range(5)]
    covs = [0 for i in range(5)]
    means[0], covs[0], data = case1(dataType)
    means[1], covs[1], data = case2(dataType)
    means[2], covs[2], data = case3(dataType)
    means[3], covs[3], data = case4(dataType)
    means[4], covs[4], data = case5(dataType)
    
    test_data = []
    with open('2_' + dataType + '/dev.txt') as f:
        lines = f.readlines()
        for line in lines :
            test_data.append(list(map(float,line.split(','))))
        f.close()
    
    data = []
    with open('2_' + dataType + '/train.txt') as f:
        lines = f.readlines()
        for line in lines :
            data.append(list(map(float,line.split(','))))
        f.close()

    prior = [1/3, 1/3, 1/3]
    likelihood = [[] for i in range(5)]
    for caseNo in range(5):
        for i in range(len(test_data)):
            x1 = test_data[i][0]
            x2 = test_data[i][1]
            tmp = []
            for j in range(3):
                tmp.append(multivariate_normal_custom(means[caseNo][j].A1, covs[caseNo][j], [x1,x2]))
            likelihood[caseNo].append(tmp)

    ground_truth = []
    for [x1,x2,z] in test_data:
        ground_truth.append(z-1)

    nTests = len(test_data); nClasses = 3
    
#     # Generating mesh in X-Y plane specific to dataType
#     count = 100
#     if(dataType == "LinearlySeparable") : x = np.linspace(-4,16,count) ; y = np.linspace(-4,16,count)
#     elif(dataType == "NonLinearlySeparable") : x = np.linspace(-12,42,count) ; y = np.linspace(-2,42,count)
#     else : x = np.linspace(100,1500,count) ; y = np.linspace(250,3000,count)
#     X,Y = np.meshgrid(x,y)

#     # Contours + Eigen grphs , Gaussian Surfaces , Decision Surfaces
#     plt.clf(); plt.figure(figsize=(35, 5))
#     plt.rcParams.update({'font.size': 18})
#     for i in range(5):
#         plt.subplot(1, 5, i+1)
#         contour_plot(means[i],covs[i],X,Y);
#     plt.savefig(dataType +'_contour.svg'); plt.clf();

#     # Gaussian Plot
#     plt.clf(); plt.figure(figsize=(35, 5))
#     plt.rcParams.update({'font.size': 18})
#     for i in range(5):
#         gaussian_plot(means[i],covs[i],X,Y);
#         plt.savefig(dataType +'_case' + str(i) + '_gaussian.svg'); plt.clf();
    
#     # Decision plot
#     plt.clf(); plt.figure(figsize=(35, 5))
#     plt.rcParams.update({'font.size': 18})
#     for i in range(5):
#         plt.subplot(1, 5, i+1)
#         decision_plot(data,means[i],covs[i],X,Y);
#     plt.savefig(dataType +'_decision.svg'); plt.clf();
        
    # Generate confusion matrix
#     for i in range(5):
#         print("\n\nCase " + str(i+1) + ': ' + dataType + '\n')
#         ConfusionMatrix(likelihood[i], nTests, ground_truth)
    
    # Plot ROC
    fig = plt.figure(figsize = (14, 8))
    plt.rcParams.update({'font.size': 20})
    legend = []
    for i in range(5):
        aoc = ROC(likelihood[i], prior, nTests, 3, ground_truth)
        legend.append("case" + str(i+1) + " (AUC = " + str(aoc) + " )")
    plt.legend(legend, loc ="lower right")
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.savefig(dataType + '_ROC.svg')
    plt.clf()
    
    # Plot DET
#     fig = plt.figure(figsize = (14, 8))
#     plt.rcParams.update({'font.size': 20})
#     ax = plt.gca()
#     for i in range(5):
#         FPR, FNR = DET(likelihood[i], prior, nTests, 3, ground_truth)
#         DetCurveDisplay(fpr = FPR, fnr = FNR, estimator_name = 'case' + str(i+1)).plot(ax)
#     plt.savefig(dataType + '_DET.svg')
#     plt.clf()

