def ROC(score, nTests, nClasses, Groundtruth):
    S = [[score[i][j] for j in range(nClasses)] for i in range(nTests)]
    
    for i in range(nTests):
        Sum = 0
        for j in range(nClasses):
            Sum += S[i][j]
        for j in range(nClasses):
            if(Sum < 0): S[i][j] /= -Sum
            else: S[i][j] /= Sum

    TPR = []; FPR = []
    thresholds = list(np.linspace(np.amin(S), np.amax(S), 1000))
#     for i in range(nTests):
#         for j in range(nClasses):
#             thresholds.append(S[i][j])
#     thresholds += list(np.linspace(-0.1,0.1,100))
#     thresholds += list(np.linspace(0.9,1,100))
    for threshold in thresholds:
        TP = FP = TN = FN = 0.0
        for i in range(nTests):
            for j in range(nClasses):
                if(S[i][j] > threshold):
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
    return TPR, FPR

def DET(score, nTests, nClasses, Groundtruth, legendName):
    S = [[score[i][j] for j in range(nClasses)] for i in range(nTests)]

    for i in range(nTests):
        Sum = 0
        for j in range(nClasses):
            Sum += S[i][j]
        for j in range(nClasses):
            if(Sum < 0): S[i][j] /= -Sum
            else: S[i][j] /= Sum

    FNR = []; FPR = []
    thresholds = list(np.linspace(np.amin(S), np.amax(S), 1000))
#     thresholds += list(np.linspace(-0.1,0.1,100))
#     thresholds += list(np.linspace(0.9,1,100))
    for threshold in thresholds:
        TP = FP = TN = FN = 0.0
        for i in range(nTests):
            for j in range(nClasses):
                if(S[i][j] > threshold):
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
    return DetCurveDisplay(fpr = FPR, fnr = FNR, estimator_name = legendName)

# Confusion Matrix
def confusionMatrix(title,path,likelihood, nTests, nClasses, groundTruth,classlist):
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
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    
    names = classlist
    ax.set_xticks(np.arange(len(names))), ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names), ax.set_yticklabels(names)
    plt.title(title)
    plt.savefig(path)
    plt.clf()
