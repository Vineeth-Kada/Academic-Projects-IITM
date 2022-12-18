def ROC(likelihood, prior, nTests, nClasses, Groundtruth):
    S = [[likelihood[j][i] * prior[j] for j in range(nClasses)] for i in range(nTests)]
    
    for i in range(nTests):
        Sum = 0
        for j in range(nClasses):
            Sum += S[i][j]
        for j in range(nClasses):
            if(Sum > 0): S[i][j] /= Sum
            else: S[i][j] /= -Sum

    TPR = []; FPR = []
    for threshold in np.linspace(np.amin(S), np.amax(S), 1000):
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
    S = [[likelihood[j][i] * prior[j] for j in range(nClasses)] for i in range(nTests)]

    for i in range(nTests):
        Sum = 0
        for j in range(nClasses):
            Sum += S[i][j]
        for j in range(nClasses):
            if(Sum > 0): S[i][j] /= Sum
            else: S[i][j] /= -Sum

    FNR = []; FPR = []
    for threshold in np.linspace(np.amin(S), np.amax(S), 1000):
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