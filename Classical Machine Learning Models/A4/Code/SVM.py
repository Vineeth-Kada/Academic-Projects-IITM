exec(open("./libraries.py").read())
exec(open("./read_input.py").read())
exec(open("./roc_det.py").read())
exec(open("./lda_pca.py").read())

# SVM - Synthetic Data

prefix = 'SVM_Synthetic_'
likelihoods = []
expName = []
accuracies = []
Y_dev_cum = []
X_train_raw, T_train, X_dev_raw, T_dev, nClasses = Synthetic()

# Experiment 1 - Normalised Data - 99.8% Accuracy
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('Normalised Data')
Y_dev_cum.append(Y_dev)

# Experiment 2 - PCA Alone - 89.5% Accuracy
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

directions = PCA(X_train, 1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA (d = 1)')
Y_dev_cum.append(Y_dev)

# Experiment 3 - LDA Alone - 92% Accuracy
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

directions = LDA(X_train,T_train, nClasses, nClasses - 1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('LDA (d = 1)')
Y_dev_cum.append(Y_dev)

for i in range(len(expName)):
    TPR, FPR = ROC(likelihoods[i], len(X_dev), nClasses, T_dev)
    plt.plot(FPR, TPR)
plt.legend(expName)
plt.title('ROC curves')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.savefig(prefix + 'ROC.svg')
plt.clf()

ax = plt.gca()
for i in range(len(expName)):
    DET(likelihoods[i], len(X_dev), nClasses, T_dev, expName[i]).plot(ax)
plt.title('DET curves')
plt.legend(loc = 'upper right')
plt.savefig(prefix + 'DET.svg')
plt.clf()

for i in range(len(expName)):
    title = expName[i] + ' Accuracy = ' + str(round(accuracies[i] * 100, 2)) + '%'
    confusionMatrix(title, prefix + 'CM_' + str(i) + '.svg', likelihoods[i], len(X_dev), nClasses, T_dev, ['0', '1'])

# SVM - Real Images

prefix = 'SVM_Real_'
likelihoods = []
expName = []
accuracies = []
Y_dev_cum = []
X_train_raw, T_train, X_dev_raw, T_dev, nClasses = RealData()

# Experiment 1 - No Normalisation
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('RAW Data')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))


# Experiment 2 - Normalisation
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('Normalised Data')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 3 - PCA Alone
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

pca_dim1 = 30
directions = PCA(X_train, pca_dim1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA (d = ' + str(pca_dim1) + ')')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 4 - LDA Alone
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

lda_dim1 = nClasses - 1
directions = LDA(X_train,T_train, nClasses, lda_dim1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('LDA (d = ' + str(lda_dim1) + ')')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 5 - PCA then LDA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

pca_dim2 = 40
directions = PCA(X_train, pca_dim2)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

lda_dim2 = nClasses - 1
directions = LDA(X_train,T_train, nClasses, lda_dim2)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA (d = ' + str(pca_dim2) + ') + LDA (d = ' + str(lda_dim2) + ')')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))

for i in range(len(expName)):
    TPR, FPR = ROC(likelihoods[i], len(X_dev), nClasses, T_dev)
    plt.plot(FPR, TPR)
plt.legend(expName)
plt.title('ROC curves')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.savefig(prefix + 'ROC.svg')
plt.clf()

ax = plt.gca()
for i in range(len(expName)):
    DET(likelihoods[i], len(X_dev), nClasses, T_dev, expName[i]).plot(ax)
plt.title('DET curves')
plt.legend(loc = 'upper right')
plt.savefig(prefix + 'DET.svg')
plt.clf()

for i in range(len(expName)):
    title = expName[i] + ' Accuracy = ' + str(round(accuracies[i] * 100, 2)) + '%'
    confusionMatrix(title, prefix + 'CM_' + str(i) + '.svg', likelihoods[i], len(X_dev), nClasses, T_dev, ['0', '1', '2', '3', '4'])

# SVM - Speech Data

prefix = 'SVM_Speech_'
likelihoods = []
expName = []
accuracies = []
Y_dev_cum = []
X_train_raw, T_train, X_dev_raw, T_dev, nClasses = speech_data()

# Experiment 1 - No Normalisation
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('RAW Data')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))


# Experiment 2 - Normalisation
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('Normalised Data')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 3 - PCA Alone
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

pca_dim1 = 40
directions = PCA(X_train, pca_dim1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA (d = ' + str(pca_dim1) + ')')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 4 - LDA Alone
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

lda_dim1 = nClasses - 1
directions = LDA(X_train,T_train, nClasses, lda_dim1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('LDA (d = ' + str(lda_dim1) + ')')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 5 - PCA then LDA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

pca_dim2 = 40
directions = PCA(X_train, pca_dim2)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

lda_dim2 = nClasses - 1
directions = LDA(X_train,T_train, nClasses, lda_dim2)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA (d = ' + str(pca_dim2) + ') + LDA (d = ' + str(lda_dim2) + ')')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))

for i in range(len(expName)):
    TPR, FPR = ROC(likelihoods[i], len(X_dev), nClasses, T_dev)
    plt.plot(FPR, TPR)
plt.legend(expName)
plt.title('ROC curves')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.savefig(prefix + 'ROC.svg')
plt.clf()

ax = plt.gca()
for i in range(len(expName)):
    DET(likelihoods[i], len(X_dev), nClasses, T_dev, expName[i]).plot(ax)
plt.title('DET curves')
plt.legend(loc = 'upper right')
plt.savefig(prefix + 'DET.svg')
plt.clf()

for i in range(len(expName)):
    title = expName[i] + ' Accuracy = ' + str(round(accuracies[i] * 100, 2)) + '%'
    confusionMatrix(title, prefix + 'CM_' + str(i) + '.svg', likelihoods[i], len(X_dev), nClasses, T_dev, ['0', '1', '2', '3', '4'])

# SVM - HandWritten

prefix = 'SVM_Character_'
likelihoods = []
expName = []
accuracies = []
Y_dev_cum = []
X_train_raw, T_train, X_dev_raw, T_dev, nClasses = character_data()

# Experiment 1 - No Normalisation
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('RAW Data')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))


# Experiment 2 - Normalisation
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('Normalised Data')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 3 - PCA Alone
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
# X_train, X_dev = normalise(X_train, X_dev)

pca_dim1 = 10
directions = PCA(X_train, pca_dim1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA (d = ' + str(pca_dim1) + ')')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 4 - LDA Alone
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
# X_train, X_dev = normalise(X_train, X_dev)

lda_dim1 = nClasses - 1
directions = LDA(X_train,T_train, nClasses, lda_dim1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('LDA (d = ' + str(lda_dim1) + ')')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 5 - PCA then LDA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
# X_train, X_dev = normalise(X_train, X_dev)

pca_dim2 = 30
directions = PCA(X_train, pca_dim2)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

lda_dim2 = nClasses - 1
directions = LDA(X_train,T_train, nClasses, lda_dim2)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)

np.random.seed(123456789)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, T_train)
likelihoods.append(clf.predict_proba(X_dev))
Y_dev = []
for i in range(len(X_dev)):
    Y_dev.append(np.argmax(likelihoods[-1][i]))
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA (d = ' + str(pca_dim2) + ') + LDA (d = ' + str(lda_dim2) + ')')
Y_dev_cum.append(Y_dev)
print(expName[-1] + ': ' + str(currAcc))

for i in range(len(expName)):
    TPR, FPR = ROC(likelihoods[i], len(X_dev), nClasses, T_dev)
    plt.plot(FPR, TPR)
plt.legend(expName)
plt.title('ROC curves')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.savefig(prefix + 'ROC.svg')
plt.clf()

ax = plt.gca()
for i in range(len(expName)):
    DET(likelihoods[i], len(X_dev), nClasses, T_dev, expName[i]).plot(ax)
plt.title('DET curves')
plt.legend(loc = 'upper right')
plt.savefig(prefix + 'DET.svg')
plt.clf()

for i in range(len(expName)):
    title = expName[i] + ' Accuracy = ' + str(round(accuracies[i] * 100, 2)) + '%'
    confusionMatrix(title, prefix + 'CM_' + str(i) + '.svg', likelihoods[i], len(X_dev), nClasses, T_dev, ['0', '1', '2', '3', '4'])