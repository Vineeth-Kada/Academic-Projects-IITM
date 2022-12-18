exec(open("./libraries.py").read())
exec(open("./read_input.py").read())
exec(open("./roc_det.py").read())
exec(open("./lda_pca.py").read())


# ANN

prefix = 'ANN_Character_'
likelihoods = []
expName = []
accuracies = []
Y_dev_cum = []
X_train_raw, T_train, X_dev_raw, T_dev, nClasses = character_data()



iters = 1000
layer = (128,512)
all_X = []

# Without normalization
all_X.append([X_train_raw,X_dev_raw])
expName.append('Raw Data')

# Normalization
X_train,X_dev = normalise(X_train_raw, X_dev_raw)
all_X.append( [X_train,X_dev] )
expName.append('Normalized Data')

# PCA
pca_dim1 = 10
directions = PCA(X_train, pca_dim1)
X_train_PCA = project_data(X_train, directions)
X_dev_PCA = project_data(X_dev, directions)
all_X.append([X_train_PCA,X_dev_PCA])
expName.append('PCA (d = ' + str(pca_dim1) + ')')

# LDA
lda_dim1 = nClasses - 1
directions = LDA(X_train,T_train, nClasses, lda_dim1)
X_train_LDA = project_data(X_train, directions)
X_dev_LDA = project_data(X_dev,  directions)
all_X.append([X_train_LDA,X_dev_LDA])
expName.append('LDA (d = ' + str(lda_dim1) + ')')


# PCA + LDA
pca_dim2 = 30
directions = PCA(X_train, pca_dim2)
X_train_PCA_LDA = project_data(X_train, directions)
X_dev_PCA_LDA = project_data(X_dev, directions)

lda_dim2 = nClasses - 1
directions = LDA(X_train_PCA_LDA,T_train, nClasses, lda_dim2)
X_train_PCA_LDA = project_data(X_train_PCA_LDA, directions)
X_dev_PCA_LDA = project_data(X_dev_PCA_LDA,  directions)
all_X.append([X_train_PCA_LDA,X_dev_PCA_LDA])
expName.append('PCA (d = ' + str(pca_dim2) + ') + LDA (d = ' + str(lda_dim2) + ')')

for i in range(len(all_X)):
    X_train,X_dev = all_X[i]
    np.random.seed(123456789)
    clf = MLPClassifier(activation = 'logistic', hidden_layer_sizes=layer, max_iter = iters, random_state=1, tol = 10**-6)
    clf.fit(X_train, T_train)

    likelihoods.append(clf.predict_proba(X_dev))
    Y_dev = []
    for j in range(len(X_dev)):
        Y_dev.append(np.argmax(likelihoods[-1][j]))
    currAcc = accuracy_score(T_dev, Y_dev)
    accuracies.append(currAcc)

    Y_dev_cum.append(Y_dev)

    print(expName[i] + ': ' + str(currAcc))




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

# ANN - HandWritten

prefix = 'ANN_Speech_'
likelihoods = []
expName = []
accuracies = []
Y_dev_cum = []
X_train_raw, T_train, X_dev_raw, T_dev, nClasses = speech_data()



iters = 2000
layer = (128,512)
all_X = []

# Without normalization
all_X.append([X_train_raw,X_dev_raw])
expName.append('Raw Data')

# Normalization
X_train,X_dev = normalise(X_train_raw, X_dev_raw)
all_X.append( [X_train,X_dev] )
expName.append('Normalized Data')

# PCA
pca_dim1 = 10
directions = PCA(X_train, pca_dim1)
X_train_PCA = project_data(X_train, directions)
X_dev_PCA = project_data(X_dev, directions)
all_X.append([X_train_PCA,X_dev_PCA])
expName.append('PCA (d = ' + str(pca_dim1) + ')')

# LDA
lda_dim1 = nClasses - 1
directions = LDA(X_train,T_train, nClasses, lda_dim1)
X_train_LDA = project_data(X_train, directions)
X_dev_LDA = project_data(X_dev,  directions)
all_X.append([X_train_LDA,X_dev_LDA])
expName.append('LDA (d = ' + str(lda_dim1) + ')')


# PCA + LDA
pca_dim2 = 30
directions = PCA(X_train, pca_dim2)
X_train_PCA_LDA = project_data(X_train, directions)
X_dev_PCA_LDA = project_data(X_dev, directions)

lda_dim2 = nClasses - 1
directions = LDA(X_train_PCA_LDA,T_train, nClasses, lda_dim2)
X_train_PCA_LDA = project_data(X_train_PCA_LDA, directions)
X_dev_PCA_LDA = project_data(X_dev_PCA_LDA,  directions)
all_X.append([X_train_PCA_LDA,X_dev_PCA_LDA])
expName.append('PCA (d = ' + str(pca_dim2) + ') + LDA (d = ' + str(lda_dim2) + ')')

for i in range(len(all_X)):
    X_train,X_dev = all_X[i]
    np.random.seed(123456789)
    clf = MLPClassifier(activation = 'logistic', hidden_layer_sizes=layer, max_iter = iters, random_state=1, tol = 10**-6)
    clf.fit(X_train, T_train)

    likelihoods.append(clf.predict_proba(X_dev))
    Y_dev = []
    for j in range(len(X_dev)):
        Y_dev.append(np.argmax(likelihoods[-1][j]))
    currAcc = accuracy_score(T_dev, Y_dev)
    accuracies.append(currAcc)

    Y_dev_cum.append(Y_dev)

    print(expName[i] + ': ' + str(currAcc))




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

# ANN - HandWritten

prefix = 'ANN_Real_'
likelihoods = []
expName = []
accuracies = []
Y_dev_cum = []
X_train_raw, T_train, X_dev_raw, T_dev, nClasses = RealData()



iters = 1000
layer = (128,512)
all_X = []

# Without normalization
all_X.append([X_train_raw,X_dev_raw])
expName.append('Raw Data')

# Normalization
X_train,X_dev = normalise(X_train_raw, X_dev_raw)
all_X.append( [X_train,X_dev] )
expName.append('Normalized Data')

# PCA
pca_dim1 = 10
directions = PCA(X_train, pca_dim1)
X_train_PCA = project_data(X_train, directions)
X_dev_PCA = project_data(X_dev, directions)
all_X.append([X_train_PCA,X_dev_PCA])
expName.append('PCA (d = ' + str(pca_dim1) + ')')

# LDA
lda_dim1 = nClasses - 1
directions = LDA(X_train,T_train, nClasses, lda_dim1)
X_train_LDA = project_data(X_train, directions)
X_dev_LDA = project_data(X_dev,  directions)
all_X.append([X_train_LDA,X_dev_LDA])
expName.append('LDA (d = ' + str(lda_dim1) + ')')


# PCA + LDA
pca_dim2 = 30
directions = PCA(X_train, pca_dim2)
X_train_PCA_LDA = project_data(X_train, directions)
X_dev_PCA_LDA = project_data(X_dev, directions)

lda_dim2 = nClasses - 1
directions = LDA(X_train_PCA_LDA,T_train, nClasses, lda_dim2)
X_train_PCA_LDA = project_data(X_train_PCA_LDA, directions)
X_dev_PCA_LDA = project_data(X_dev_PCA_LDA,  directions)
all_X.append([X_train_PCA_LDA,X_dev_PCA_LDA])
expName.append('PCA (d = ' + str(pca_dim2) + ') + LDA (d = ' + str(lda_dim2) + ')')

for i in range(len(all_X)):
    X_train,X_dev = all_X[i]
    np.random.seed(123456789)
    clf = MLPClassifier(activation = 'logistic', hidden_layer_sizes=layer, max_iter = iters, random_state=1, tol = 10**-6)
    clf.fit(X_train, T_train)

    likelihoods.append(clf.predict_proba(X_dev))
    Y_dev = []
    for j in range(len(X_dev)):
        Y_dev.append(np.argmax(likelihoods[-1][j]))
    currAcc = accuracy_score(T_dev, Y_dev)
    accuracies.append(currAcc)

    Y_dev_cum.append(Y_dev)

    print(expName[i] + ': ' + str(currAcc))




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

# ANN - HandWritten

prefix = 'ANN_Syn_'
likelihoods = []
expName = []
accuracies = []
Y_dev_cum = []
X_train_raw, T_train, X_dev_raw, T_dev, nClasses = Synthetic()



iters = 1000
layer = (128,512)
all_X = []

# Without normalization
all_X.append([X_train_raw,X_dev_raw])
expName.append('Raw Data')

# Normalization
X_train,X_dev = normalise(X_train_raw, X_dev_raw)
all_X.append( [X_train,X_dev] )
expName.append('Normalized Data')

# PCA
pca_dim1 = 1
directions = PCA(X_train, pca_dim1)
X_train_PCA = project_data(X_train, directions)
X_dev_PCA = project_data(X_dev, directions)
all_X.append([X_train_PCA,X_dev_PCA])
expName.append('PCA (d = ' + str(pca_dim1) + ')')

# LDA
lda_dim1 = nClasses - 1
directions = LDA(X_train,T_train, nClasses, lda_dim1)
X_train_LDA = project_data(X_train, directions)
X_dev_LDA = project_data(X_dev,  directions)
all_X.append([X_train_LDA,X_dev_LDA])
expName.append('LDA (d = ' + str(lda_dim1) + ')')


# PCA + LDA
pca_dim2 = 1
directions = PCA(X_train, pca_dim2)
X_train_PCA_LDA = project_data(X_train, directions)
X_dev_PCA_LDA = project_data(X_dev, directions)

lda_dim2 = nClasses - 1
directions = LDA(X_train_PCA_LDA,T_train, nClasses, lda_dim2)
X_train_PCA_LDA = project_data(X_train_PCA_LDA, directions)
X_dev_PCA_LDA = project_data(X_dev_PCA_LDA,  directions)
all_X.append([X_train_PCA_LDA,X_dev_PCA_LDA])
expName.append('PCA (d = ' + str(pca_dim2) + ') + LDA (d = ' + str(lda_dim2) + ')')

for i in range(len(all_X)):
    X_train,X_dev = all_X[i]
    np.random.seed(123456789)
    clf = MLPClassifier(activation = 'logistic', hidden_layer_sizes=layer, max_iter = iters, random_state=1, tol = 10**-6)
    clf.fit(X_train, T_train)

    likelihoods.append(clf.predict_proba(X_dev))
    Y_dev = []
    for j in range(len(X_dev)):
        Y_dev.append(np.argmax(likelihoods[-1][j]))
    currAcc = accuracy_score(T_dev, Y_dev)
    accuracies.append(currAcc)

    Y_dev_cum.append(Y_dev)

    print(expName[i] + ': ' + str(currAcc))




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

