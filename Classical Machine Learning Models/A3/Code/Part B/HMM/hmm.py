# Libraries

#Libraries

import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import random
import pandas as pd
from sklearn.metrics import DetCurveDisplay
import os
import sy

# Utility Functions

# Vineeth
def kMeans(X, K, k_iterations):
    N = len(X)
    D = X[0].shape[0]
    
    # Randomly Choose k - points as the means
    # mu = [ X[k] for k in range(K) ]
    # X.sort()
    mu = list(random.sample(X,K))
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

# ROC Plotting
def ROC(likelihood, prior, nTests, nClasses, Groundtruth):
    S = [[likelihood[i][j] * prior[j] for j in range(nClasses)] for i in range(nTests)]
    

    TPR = []; FPR = []

    l = []
    for i in range(nTests):
        for j in range(nClasses):
            l.append(S[i][j])
    l = list(set(l))
        
        
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
        TPR.append(TP/(TP + FN))
        FPR.append(FP/(FP + TN))

    FPR, TPR = zip(*sorted(zip(FPR, TPR)))
    plt.plot(FPR, TPR)
    return np.trapz(TPR, FPR)


def DET(likelihood, prior, nTests, nClasses, Groundtruth):
    S = [[likelihood[i][j] * prior[j] for j in range(nClasses)] for i in range(nTests)]
    

    FNR = []; FPR = []
    l = []
    for i in range(nTests):
        for j in range(nClasses):
            l.append(S[i][j])
    l = list(set(l))
        
        
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
    FPR, FNR = zip(*sorted(zip(FPR, FNR)))
    return FPR, FNR

os.system('cd HMM-Code;touch test.hmm.seq')

# Generic Functions

""" Data Reading Functions """

def read_all_train(title,class_names,nClasses):
    all_X = []
    
    for c in range(nClasses):
        name = class_names[c]
        path = title + '_data/' + name + '/train/*.mfcc'
        all_filePaths = list(glob.glob(path))

        for file in all_filePaths:
            with open(file) as f:
                lines = f.readlines()
                for line in lines[1:]:
                    fvec = np.array( list(map(float, line.split())) ,dtype = np.float64)
                    all_X.append(fvec)
                    
    return all_X

                
                
def read_train(title,class_names,nClasses):
    X = [[] for _ in range(nClasses)]

    for c in range(nClasses):
        name = class_names[c]
        path = title + '_data/' + name + '/train/*.mfcc'
        all_filePaths = list(glob.glob(path))

        for file in all_filePaths:
            with open(file) as f:
                lines = f.readlines()
                tmp = []
                for line in lines[1:]:
                    fvec = np.array( list(map(float, line.split()))  ,dtype = np.float64)
                    tmp.append(fvec)
                X[c].append(tmp)
        
    return X

def read_dev(title,class_names,nClasses):    
    X_dev = []
    groundtruth = []
    for c in range(nClasses):
        name = class_names[c]
        path = title + '_data/' + name + '/dev/*.mfcc'
        all_filePaths = list(glob.glob(path))

        for file in all_filePaths:
            with open(file) as f:
                lines = f.readlines()

                tmp = []
                for line in lines[1:]:
                    fvec = np.array( list(map(float, line.split()))  ,dtype = np.float64)
                    tmp.append(fvec)

                X_dev.append(tmp)
                groundtruth.append(c)
                
    return X_dev , groundtruth



""" Vector Quantization - Converting Vectors to Symbols for Speech Recognition """
def vector_quantization(x,mu):
    sym = -1
    mini = 1000000
    for idx in range(len(mu)):
        dist = math.sqrt((x-mu[idx]) @ (x - mu[idx]))
        if(dist < mini):
            mini = dist
            sym = idx
    return sym

def train_VQ(X,mu,nClasses):
    X_sym = [[] for _ in range(nClasses)]
    for c in range(nClasses):
        for file in X[c]:
            tmp = []
            for x in file:
                sym = vector_quantization(x,mu)
                tmp.append(sym)
            X_sym[c].append(tmp)
    return X_sym
    
def dev_VQ(X_dev,mu,nClasses):
    X_sym_dev = []
    for file in X_dev:
        sym_seq = []
        for fvec in file:
            sym_seq.append( vector_quantization(fvec,mu) )
        X_sym_dev.append(sym_seq)
    return X_sym_dev
        

""" HMM CODE - ESTIMATION OF PROBABILITIES """

def hmm_code(X_sym,nClasses,states,symbols,seed,pmin):
    # all_A[class_id][state_id][recur/right] = real number between 0 to 1
    all_A = [[[0 for j in range(2)] for i in range(states[c])] for c in range(nClasses)]

    # all_B[class_id][state_id][recur/right][symbol_id] = real number between 0 to 1
    all_B = [[[[0 for k in range(symbols)] for j in range(2)] for i in range(states[c])] for c in range(nClasses)]


    for c in range(nClasses):
        path = 'HMM-Code/test.hmm.seq'
        with open(path,'w') as f:
            for file in X_sym[c]:
                string = ''
                for sym in file : string += str(sym) + ' '
                string += '\n'
                f.write(string)
            f.close()


        os.system(f"cd HMM-Code ; ./train_hmm test.hmm.seq {seed} {states[c]} {symbols} {pmin}")

        with open('HMM-Code/test.hmm.seq.hmm') as f:
            lines = f.readlines()
            row = 2
            state = 0
            while(row < len(lines)):
                l1 = list(map(float,lines[row].split()))
                if(len(l1) == 0) : row += 1 ; continue
                all_A[c][state][0] = l1[0]
                for i in range(1,len(l1)):
                    all_B[c][state][0][i-1] = l1[i]
                row += 1

                l2 = list(map(float,lines[row].split()))
                all_A[c][state][1] = l2[0]
                for i in range(1,len(l2)):
                    all_B[c][state][1][i-1] = l2[i]

                row += 2
                state += 1
            f.close()
        
    return all_A, all_B

""" Find Likelihood for given sequence(seq) wrt model of state probability matrix 'A', symbol probability matrix B"""
def find_likelihood(seq,A,B):
    n = len(seq)
    states = len(A)
    symbols = len(B[0][0])
    
    # dp[seq_len][states]
    dp = [[0 for i in range(states)] for _ in range(n)]
    
    for i in range(n):
        for j in range(states):
            # Computing dp[i][j]
            # act1 : new state transtition probabililty 
            act1 = 0
            if(j != 0) : act1 = A[j-1][1] * B[j-1][1][seq[i]]
                
            # act2 : new state transtition probabililty 
            act2 = A[j][0] * B[j][0][seq[i]]
            
            p1 = 0
            if(i != 0) : p1 = dp[i-1][j-1]
            
            p2 = 1
            if(i != 0) : p2 = dp[i-1][j]
                
            dp[i][j] = p1 * act1 + p2 * act2 
    
    ans = 0
    for j in range(states) : ans += dp[n-1][j]
    return ans

""" HMM's - ROC  DET  CONFUSION-MATRIX """
""" Compute Likelihood matrix for ROC,DET Graphs"""
def likelihood_matrix(X_sym_dev,all_A,all_B,nClasses,title):
    nTests = len(X_sym_dev)
    likelihood = [[0 for i in range(nClasses)] for _ in range(nTests)]
    for i in range(nTests):
        for j in range(nClasses):
            for x in X_sym_dev[i]:
                likelihood[i][j] = find_likelihood(X_sym_dev[i],all_A[j],all_B[j])
    return likelihood

def hmm_roc(title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols):
    area = ROC(likelihood,prior,nTests,nClasses,groundtruth)
    states_str = ''
    for i in range(nClasses):states_str += str(states[i]) + '|'
    states_str = states_str[:-1]
    plt.title(f'States : {states_str} Symbols : {symbols}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend([f'Area : {area}'])
    path = f'HMM_{title}_plots/sym{symbols}_{states_str}_ROC.svg'
    plt.savefig(path)
    plt.clf()
    return area


def hmm_det(title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols):
    FPR,FNR = DET(likelihood,prior,nTests,nClasses,groundtruth)
    states_str = ''
    for i in range(nClasses):states_str += str(states[i]) + '|'
    states_str = states_str[:-1]
    # , estimator_name = f'nSym = {symbols} states = {states_str}'
    ax = plt.gca()
    DetCurveDisplay(fpr = FPR, fnr = FNR).plot(ax)
    
    plt.xlabel('FPR')
    plt.ylabel('FNR')
    plt.title(f'DET Curve for nSym : {symbols} states : {states_str}')

    path = 'HMM_' + title + '_plots' + f'/sym{symbols}_{states_str}_DET.svg'
    plt.savefig(path)
    plt.clf()

    
def hmm_confusionMatrix(title,prefix,likelihood, nTests, nClasses, groundTruth,classlist,states,symbols):
    cmat = np.zeros((nClasses, nClasses))
    cfm1 = np.zeros((nClasses, nClasses))
    for i in range(nTests):
        act = int(groundTruth[i])
        pred = int(np.argmax(likelihood[i]))
        cmat[pred, act] += 1
    cfm = cmat
    for i in range(0,len(classlist)):
        for j in range(0,len(classlist)):
            cfm1[i,j] = cfm[i][j]/nTests
    
    fig, ax = plt.subplots(figsize=(6,6))

    ax.matshow(cmat)
    for i in range(cfm.shape[0]):
        for j in range(cfm.shape[1]):
            ax.text(x=j, y=i,s=str(cfm[i, j])+'\n\n'+'{:.2%}'.format(cfm1[i,j]), va='center', ha='center',color='white')
    states_str = ''
    for i in range(nClasses):states_str += str(states[i]) + '|'
    states_str = states_str[:-1]
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    
    names = class_names
    ax.set_xticks(np.arange(len(names))), ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names), ax.set_yticklabels(names)
    plt.title(f'Confusion Matrix for nSym : {symbols} states : {states_str}')
    plt.savefig(f'HMM_{title}_plots/{prefix}_sym{symbols}_state{states_str}_confusionMatrix.svg')
    plt.clf()



# Speech Recognition 

title = 'speech'
prefix = 'HSP'
class_names = ['1','4','5','7','o']
nClasses = len(class_names)
all_X = read_all_train(title, class_names, nClasses)
X = read_train(title, class_names, nClasses)
X_dev,groundtruth = read_dev(title, class_names, nClasses)
nTests = len(X_dev)

states_list = [[5,5,5,5,5],[10,10,10,10,10],[5,5,5,10,10]]
symbols_list = [25,50]
# states_list = [[5,5,5,5,5]]
# symbols_list = [25]
grph_params = []

for i in range(len(states_list)):
    for j in range(len(symbols_list)):
        states = states_list[i]
        symbols = symbols_list[j]
        if(i == 2) : symbols += 25
        
        cls,mu = kMeans(all_X,symbols,10)

        X_sym = train_VQ(X,mu,nClasses)
        X_sym_dev = dev_VQ(X_dev,mu, nClasses)

        all_A, all_B = hmm_code(X_sym,nClasses,states,symbols,1234,.01)

        likelihood = likelihood_matrix(X_sym_dev,all_A,all_B,nClasses,title)

        prior = [(1/nClasses) for _ in range(nClasses)]

        grph_params.append([title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols])
        hmm_confusionMatrix(title,prefix,likelihood, nTests, nClasses, groundtruth,class_names,states,symbols)
        
    

plt.clf()
leg = []
for [title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols] in grph_params:
    area = ROC(likelihood,prior,nTests,nClasses,groundtruth)
    states_str = ''
    for i in range(nClasses):states_str += str(states[i]) + '|'
    states_str = states_str[:-1]
    plt.title('ROC Curve for Different Symbols and States(per class)')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    leg.append(f'nSym : {symbols} nState : {states_str}')
plt.legend(leg)
plt.savefig(f'HMM_{title}_plots/HSP_ROC.svg')

plt.clf()
leg = []

for [title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols] in grph_params:
    FPR,FNR = DET(likelihood,prior,nTests,nClasses,groundtruth)

    states_str = ''
    for i in range(nClasses):states_str += str(states[i]) + '|'
    states_str = states_str[:-1]
    # , estimator_name = f'nSym = {symbols} states = {states_str}'
    ax = plt.gca()
    DetCurveDisplay(fpr = FPR, fnr = FNR).plot(ax)

    leg.append(f'nSym : {symbols} nState : {states_str}')

plt.xlabel('FPR')
plt.ylabel('FNR')
plt.title(f'DET Curve for Different Symbols and States(per class)')

plt.legend(leg, loc='upper right')
plt.savefig(f'HMM_{title}_plots/HSP_DET.svg')
plt.clf()


# Character Recognition

def extractHandWritingData(dir):
    files = os.listdir(dir)
    
    templates = []
    for file in files:
        with open(dir + file, 'r') as f:
            curr = np.array(f.readline().split()[1:], dtype=np.float64).reshape(-1,2)
            
            # Position Invariant
            for i in range(2):
                mn = np.min(curr[:, i])
                mx = np.max(curr[:, i])
                curr[:, i] = curr[:, i] - (mn + mx) / 2
            
            # Scale Invariant
            mn = np.array([np.min(curr[:, 0]), np.min(curr[:, 1])])
            mx = np.array([np.max(curr[:, 0]), np.max(curr[:, 1])])
            curr = curr / (mx - mn)
            templates.append(curr)

    return templates



        

title = 'character'
prefix = 'HCP'
class_names = ['ai','bA','dA','lA','tA']
nClasses = len(class_names)

all_X = []
X = [[] for _ in range(nClasses)]
X_dev = []
groundtruth = []


for i in range(nClasses):
    name = class_names[i]
    dir_path = title + '_data/'+name+'/train/'
    data = extractHandWritingData(dir_path)

    X[i] = list(data)
    for file in list(data):
        for fvec in file:
            all_X.append(fvec)

    dir_path = title + '_data/'+name+'/dev/'
    data = extractHandWritingData(dir_path)
    for file in data:
        fvec = []
        for x in file:
            fvec.append(x)
        X_dev.append(fvec)
        groundtruth.append(i)

nTests = len(X_dev)

# # """                                KMeans VQ Experiment                                             """

# states_list = [[states for _ in range(nClasses)] for states in [10,5,12,9,12]]
# pieces_list = [5,6,6,7,9]
# symbols_list = [pieces * pieces for pieces in pieces_list]
# grph_params = []

# seed = 59
# pmin = 0.01

# maxa = 0
# m_sym = 0
# m_states = 0

# for i in range(len(states_list)):
#     for j in range(len(symbols_list)):

#         states = states_list[i]
#         symbols = symbols_list[j]

#         kmeans = KMeans(n_clusters=symbols, init='random', max_iter=300, n_init=10, random_state=0)
#         kmeans.fit(np.array(all_X))
#         mu = list(kmeans.cluster_centers_)
        
#         X_sym = train_VQ(X,mu,nClasses)
#         X_sym_dev = dev_VQ(X_dev,mu, nClasses)
        
#         all_A, all_B = hmm_code(X_sym,nClasses,states,symbols,seed,pmin)


#         likelihood = likelihood_matrix(X_sym_dev,all_A,all_B,nClasses,title)

#         prior = [1 for _ in range(nClasses)]
# #         area = hmm_roc(title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols)
# #         if(area > maxa):
# #             maxa = area
# #             m_sym = symbols
# #             m_states = states[0]
# #         hmm_det(title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols)
# #         grph_params.append([title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols])
#         hmm_confusionMatrix(title,prefix,likelihood, nTests, nClasses, groundtruth,class_names,states,symbols)

# # print('#############################################################################################################')
# # print(maxa,m_sym,m_states)

# plt.clf()
# leg = []
# for [title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols] in grph_params:
#     area = ROC(likelihood,prior,nTests,nClasses,groundtruth)
#     states_str = ''
#     for i in range(nClasses):states_str += str(states[i]) + '|'
#     states_str = states_str[:-1]
#     plt.title('ROC Curve for Different Symbols and States(per class)')
#     plt.xlabel('FPR')
#     plt.ylabel('TPR')
#     leg.append(f'nSym : {symbols} nState : {states_str}')
# plt.legend(leg)
# plt.savefig(f'HMM_{title}_plots_kmeans/ROC.svg')

# plt.clf()
# leg = []

# for [title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols] in grph_params:
#     FPR,FNR = DET(likelihood,prior,nTests,nClasses,groundtruth)

#     states_str = ''
#     for i in range(nClasses):states_str += str(states[i]) + '|'
#     states_str = states_str[:-1]
#     # , estimator_name = f'nSym = {symbols} states = {states_str}'
#     ax = plt.gca()
#     DetCurveDisplay(fpr = FPR, fnr = FNR).plot(ax)

#     leg.append(f'nSym : {symbols} nState : {states_str}')

# plt.xlabel('FPR')
# plt.ylabel('FNR')
# plt.title(f'DET Curve for Different Symbols and States(per class)')

# plt.legend(leg, loc='upper right')
# plt.savefig(f'HMM_{title}_plots_kmeans/DET.svg')
# plt.clf()


"""                                      Grid VQ Experiments                                              """

def vector_quantization(fvec,pieces):
    x = fvec[0]
    y = fvec[1]

    x_idx = math.floor((x + 0.5) * pieces)
    if(x_idx < 0):x_idx = 0
    if(x_idx >= pieces):x_idx = pieces-1
    y_idx = math.floor((y + 0.5) * pieces)
    if(y_idx < 0):y_idx = 0
    if(y_idx >= pieces):y_idx = pieces-1
    sym = x_idx * pieces + y_idx
    if(sym == pieces * pieces):print(fvec,sym) ; sys.exit()
    return sym

def train_VQ(X,pieces,nClasses):
    X_sym = [[] for _ in range(nClasses)]
    for c in range(nClasses):
        for file in X[c]:
            tmp = []
            for x in file:
                sym = vector_quantization(x,pieces)
                tmp.append(sym)
            X_sym[c].append(tmp)
    return X_sym
    
def dev_VQ(X_dev,pieces,nClasses):
    X_sym_dev = []
    for file in X_dev:
        sym_seq = []
        for fvec in file:
            sym_seq.append( vector_quantization(fvec,pieces) )
        X_sym_dev.append(sym_seq)
    return X_sym_dev


states_list = [[states for _ in range(nClasses)] for states in [10,5,12,9,12]]
pieces_list = [5,6,6,7,9]
grph_params = []

seed = 59
pmin = 0.01

maxa = 0
m_sym = 0
m_states = 0

for i in range(len(states_list)):
    for j in range(len(pieces_list)):
        if(i != j) : continue
        pieces = pieces_list[j]
        states = states_list[i]
        symbols = pieces * pieces

        X_sym = train_VQ(X,pieces,nClasses)
        X_sym_dev = dev_VQ(X_dev,pieces, nClasses)
        
        all_A, all_B = hmm_code(X_sym,nClasses,states,symbols,seed,pmin)


        likelihood = likelihood_matrix(X_sym_dev,all_A,all_B,nClasses,title)

        prior = [1 for _ in range(nClasses)]
# #         area = hmm_roc(title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols)
#         if(area > maxa):
#             maxa = area
#             m_sym = symbols
#             m_states = states[0]
#         hmm_det(title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols)
        grph_params.append([title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols])
        hmm_confusionMatrix(title,prefix,likelihood, nTests, nClasses, groundtruth,class_names,states,symbols)

print('#############################################################################################################')
print(maxa,m_sym,m_states)


plt.clf()
leg = []
for [title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols] in grph_params:
    area = ROC(likelihood,prior,nTests,nClasses,groundtruth)
    states_str = ''
    for i in range(nClasses):states_str += str(states[i]) + '|'
    states_str = states_str[:-1]
    plt.title('ROC Curve for Different Symbols and States(per class)')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    leg.append(f'nSym : {symbols} nState : {states_str}')
plt.legend(leg)
plt.savefig(f'HMM_{title}_plots/ROC.svg')

plt.clf()
leg = []

for [title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols] in grph_params:
    FPR,FNR = DET(likelihood,prior,nTests,nClasses,groundtruth)

    states_str = ''
    for i in range(nClasses):states_str += str(states[i]) + '|'
    states_str = states_str[:-1]
    # , estimator_name = f'nSym = {symbols} states = {states_str}'
    ax = plt.gca()
    DetCurveDisplay(fpr = FPR, fnr = FNR).plot(ax)

    leg.append(f'nSym : {symbols} nState : {states_str}')

plt.xlabel('FPR')
plt.ylabel('FNR')
plt.title(f'DET Curve for Different Symbols and States(per class)')

plt.legend(leg, loc='upper right')
plt.savefig(f'HMM_{title}_plots/DET.svg')
plt.clf()


# """                                   Direction of Vectors Experiment                                     """
# def modify_X(X):
#     X_new = []
#     for i in range(1,len(X)):
#         [x0,y0] = X[i-1]
#         [x1,y1] = X[i]
#         x = x1 - x0
#         y = y1 - y0
#         d = np.linalg.norm([x,y])
#         x = (100 * x)/d
#         y = (100 * y)/d
#         X_new.append(np.array([x,y]))
#     return X_new


            
# all_X = []
# X = [[] for _ in range(nClasses)]
# X_dev = []
# groundtruth = []

# # cnt = 0
# # plt.figure(figsize=(35, 250))
# for i in range(nClasses):
#     name = class_names[i]
#     dir_path = title + '_data/'+name+'/train/'
#     data = extractHandWritingData(dir_path)

#     X[i] = list(data)
#     for file in list(data):
#         for fvec in file:
#             all_X.append(fvec)

#     dir_path = title + '_data/'+name+'/dev/'
#     data = extractHandWritingData(dir_path)
#     for file in data:
#         fvec = []
#         for x in file:
#             fvec.append(x)
#         X_dev.append(fvec)
#         groundtruth.append(i)

        
# all_X = []    
# for i in range(len(X)):
#     for j in range(len(X[i])):
#         X[i][j] = modify_X(X[i][j])
#         all_X += X[i][j]
# print(all_X[:2])            
        
# for i in range(len(X_dev)):
#     X_dev[i] = modify_X(X_dev[i])

            

# """                                KMeans VQ Experiment                                             """
# # states_list = [[5,5,5,5,5],[10,10,10,10,10],[5,5,5,10,10]]
# # symbols_list = [25,50]
# nTests = len(X_dev)


# nStates = [3,5,10]
# states_list = [[states for _ in range(nClasses)] for states in nStates]
# symbols_list = [75]
# grph_params = []

# seed = 59
# pmin = 0.01

# maxa = 0
# m_sym = 0
# m_states = 0

# for i in range(len(states_list)):
#     for j in range(len(symbols_list)):

#         states = states_list[i]
#         symbols = symbols_list[j]


        
#         cls,mu = kMeans(all_X,symbols,10)
        
#         X_sym = train_VQ(X,mu,nClasses)
#         X_sym_dev = dev_VQ(X_dev,mu, nClasses)
        
#         all_A, all_B = hmm_code(X_sym,nClasses,states,symbols,seed,pmin)


#         likelihood = likelihood_matrix(X_sym_dev,all_A,all_B,nClasses,title)

#         prior = [1 for _ in range(nClasses)]
#         area = hmm_roc(title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols)
#         if(area > maxa):
#             maxa = area
#             m_sym = symbols
#             m_states = states[0]
# #         hmm_det(title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols)
# #         grph_params.append([title,likelihood,prior,nTests,nClasses,groundtruth,states,symbols])
# #         hmm_confusionMatrix(title,likelihood, nTests, nClasses, groundtruth,class_names,states,symbols)

# print('#############################################################################################################')
# print(maxa,m_sym,m_states)

