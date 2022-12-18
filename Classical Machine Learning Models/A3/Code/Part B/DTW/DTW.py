# Common Function & Libraries for both data sets

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay, ConfusionMatrixDisplay, confusion_matrix
from numba import njit

@njit
def DTW(a, b, w):
    n, m = len(a), len(b)
    dtw = np.zeros((n + 1, m + 1), dtype = np.float64)
    dtw.fill(np.inf)
    
    # We are taking the least size window possible i.e., abs(n - m)
    # But it shouldn't be too low so we are using atleast w passed by user
    w = max(w, abs(n - m))
    
    dtw[0][0] = 0.0

    # DTW calculation
    for i in range(1, n+1):
        lb = max(1, i - w)
        ub = min(m+1, i + w + 1)
        for j in range(lb, ub):
            dtw[i][j] = 0.0
            cost = np.linalg.norm(a[i-1] - b[j-1])
            dtw[i][j] = cost + min(min(dtw[i][j-1], dtw[i-1][j]), dtw[i-1][j-1])

    return dtw[n][m]

# Data Extraction - Isolated Spoken-Digit Dataset
def extractDigitData(dir):
    files = os.listdir(dir)
    
    templates = []; NC = 0
    for file in files:
        currFile = []
        with open(dir + file, 'r') as f:
            NC, NF = list(map(int, f.readline().split()))
            for line in f.readlines():
                currFile.append(np.array(line.split(), dtype = np.float64))
        templates.append(np.array(currFile))

    return NC, templates

# Data Extraction - Online Handwritten-Character dataset
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
            diff = mx - mn
            curr[:, 0] = curr[:, 0] / diff[0]
            curr[:, 1] = curr[:, 1] / diff[1]
                
            templates.append(curr)

    return templates

# Main DTW Function
def DTWMain(inp):
    Handwriting = True
    classList = []
    
    if(inp == 'Spoken'):
        Handwriting = False 
        classList = ['1', '4', '5', '7', 'o']
        nClasses = len(classList)

        train, dev = [[] for i in range(nClasses)], [[] for i in range(nClasses)]
        NC = 0

        for cls in range(nClasses):
            NC, train[cls] = extractDigitData("./Isolated_Digits_Data/" + classList[cls] + "/train/")
            NC, dev[cls] = extractDigitData("./Isolated_Digits_Data/" + classList[cls] + "/dev/")

    elif(inp == 'HandWritten'):
        Handwriting = True
        classList = ['ai', 'bA', 'dA', 'lA', 'tA']
        nClasses = len(classList)

        train, dev = [[] for i in range(nClasses)], [[] for i in range(nClasses)]

        for cls in range(nClasses):
            train[cls] = extractHandWritingData("./Handwriting_Data/" + classList[cls] + "/train/")
            dev[cls] = extractHandWritingData("./Handwriting_Data/" + classList[cls] + "/dev/")

    # Testing Zone
    if(Handwriting): dir_path = 'DTW_Results/HandWritten_'
    else: dir_path = 'DTW_Results/SpokenDigit_'

    if(Handwriting): windowRange = range(5, 50, 5)
    else: windowRange = range(5, 50, 5)

    window_value = []
    window_acc = []
    for window in windowRange:
        window_value.append(window)
        print("WINDOW:", window)
        scores = []
        for actual_cls in range(nClasses):
            print("Class" + str(actual_cls))
            for d_i in range(len(dev[actual_cls])):
                currAvgScoreList = []
                for cmp_cls in range(nClasses):
                    thisClassScores = np.zeros(len(train[cmp_cls]))
                    for t_i in range(len(train[cmp_cls])):
                        thisClassScores[t_i] = DTW(train[cmp_cls][t_i], dev[actual_cls][d_i], window)
                    currAvgScoreList.append(thisClassScores)
                scores.append(currAvgScoreList)

        mx_acc = 0
        k_value = []
        k_acc = []

        for top_k in range(5, 95):
            k_value.append(top_k)
            pred_values = []
            ground_truth = []
            likelihood = [[] for i in range(nClasses)]
            nTests, accuracy = 0, 0
            for actual_cls in range(nClasses):
                for d_i in range(len(dev[actual_cls])):

                    ground_truth.append(actual_cls)

                    currAvgScoreList = scores[nTests]
                    currAvgScoreListNew = []
                    for cmp_cls in range(nClasses):

                        thisClassScores = currAvgScoreList[cmp_cls]

                        k = top_k * len(thisClassScores) // 100
                        idx = np.argpartition(thisClassScores, k)
                        best_k_avg = np.sum(thisClassScores[idx[:k]]) / k

                        likelihood[cmp_cls].append(1 / best_k_avg)
                        currAvgScoreListNew.append(best_k_avg)

                    pred_value = np.argmin(currAvgScoreListNew)
                    if(pred_value == actual_cls): accuracy += 1
                    pred_values.append(pred_value)
                    nTests += 1
            accuracy /= nTests
            if(accuracy * 100 > mx_acc):
                max_k = top_k
                mx_acc = accuracy * 100
            k_acc.append(accuracy * 100)

            if (top_k == 20 and accuracy * 100 > 98):
                prior = [1 / nClasses for _ in range(nClasses)]
                ROC(likelihood, prior, nTests, nClasses, ground_truth)
                plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
                plt.title('ROC for top k = ' + str(top_k) + ' and window = ' + str(window))
                plt.savefig(dir_path + 'w' + str(window) + '_k' + str(top_k) + '_ROC.svg') 
                plt.clf()

                prior = [1 / nClasses for _ in range(nClasses)]
                FPR, FNR = DET(likelihood, prior, nTests, nClasses, ground_truth)
                ax = plt.gca()
                DetCurveDisplay(fpr = FPR, fnr = FNR).plot(ax)
                plt.title('DET for top k = ' + str(top_k) + ' and window = ' + str(window))
                plt.savefig(dir_path + 'w' + str(window) + '_k' + str(top_k) + '_DET.svg') 
                plt.clf()
                
                ConfusionMatrixDisplay.from_predictions(ground_truth, pred_values, display_labels=classList, xticks_rotation=45).plot()
                plt.title('Confusion Matrix for top k = ' + str(top_k) + ' and window = ' + str(window))
                plt.savefig(dir_path + 'w' + str(window) + '_k' + str(top_k) + '_Confusion_Matrix.svg')
                plt.clf()

        window_acc.append(mx_acc)
        plt.plot(k_value, k_acc)
        plt.xlabel('Least K Scores')
        plt.ylabel('Accuracy Percentage') 
        plt.title('Least K scores vs. Max Accuracy for window = ' + str(window))
        plt.savefig(dir_path + 'w' + str(window) + '_k_vs_accuracy.svg')
        plt.clf()

        print("*" * 25)
        print("Accuracy is", mx_acc)
        print("*" * 25)

    plt.plot(window_value, window_acc)
    plt.xlabel('Window Size')
    plt.ylabel('Accuracy Percentage')
    plt.title('Window vs. Max Accuracy')
    plt.savefig(dir_path + 'window_vs_max_accuracy.svg')
    plt.clf()

exec(open("../../sharedFunctions.py").read())
# All the results will be stored in Results folder
DTWMain('Spoken')
DTWMain('HandWritten')