exec(open("./libraries.py").read())
exec(open("./read_input.py").read())
exec(open("./roc_det.py").read())
exec(open("./lda_pca.py").read())

# Logistic Regression - Function

# X - list of input vectors, each input vector is a 1D numpy array.
# T - numpy array of true values
# phi - Basis Function
# iters - No of iterations of gradient descent algorithm
def LogisticRegression(X, t, phi, wLen, nClasses, iters, eta):
    F = X[0].shape[0] # F - number of features
    N = len(X)

    phi_X = []
    for n in range(N):
        phi_X.append(phi(X[n]))

    wnew = [np.zeros(wLen, dtype=np.float64) for _ in range(nClasses)]
    wold = [np.zeros(wLen, dtype=np.float64) for _ in range(nClasses)]
    for ic in range(iters):
        if(ic % 50 == 0): print(ic)
        for j in range(nClasses):
            Del = np.zeros(wLen)
            for n in range(N):
                a = np.array( [wold[c] @ phi_X[n] for c in range(nClasses)] )
                ynj = np.exp(a[j]) / sum(np.exp(a))
                if(j == t[n]): Del += (ynj - 1) * phi_X[n]
                else: Del += (ynj - 0) * phi_X[n]
            wnew[j] = wold[j] - eta * Del
        wold, wnew = wnew, wold
    return wold

# Logistic Regression for Synthetic Data

# Basis Functions
def basis1(X):
    return np.insert(X, 0, 1)

def basis2(X):
    X_new = [i for i in basis1(X)]
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            X_new.append(X[i] * X[j])
    return np.array(X_new)

def calc_basis_vector(x, ord = 3):
    dim = len(x)
    basis_vec = [i for i in basis2(x)]
    for M in range(ord + 1):
        if(dim == 1):
            basis_vec.append(x[0]**M)
        else:
            # Exploiting all possible distribution of powers 
            for p in range(M+1):
                basis_vec.append( (x[0]**(M-p)) )
                basis_vec.append( (x[1]**p) * (x[0]**(M-p)) )
    return np.array(basis_vec)

prefix = 'LR_Synthetic_'
likelihoods = []
expName = []
accuracies = []
Y_dev_cum = []
X_train_raw, T_train, X_dev_raw, T_dev, nClasses = Synthetic()

# Experiment 1 - Normalised Data
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)

phi = calc_basis_vector # Basis Vector is a degree 3 polynomial
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 100, 10**-4.5)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('Normalised Data')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 2 - PCA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

directions = PCA(X_train, 1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

np.random.seed(123456789)

phi = calc_basis_vector # Basis Vector is a degree 3 polynomial
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 100, 10**-4.5)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA (d = 1)')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 3 - LDA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

directions = LDA(X_train,T_train, nClasses, nClasses - 1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)
X_train, X_dev = normalise(X_train, X_dev) # Normalise again to prevent exponential blowup

np.random.seed(123456789)

phi = calc_basis_vector # Basis Vector is a degree 3 polynomial
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 100, 10**-4.5)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('LDA (d = 1)')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
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
    confusionMatrix(title, prefix + 'CM_' + str(i) + '.svg', likelihoods[i], len(X_dev), nClasses, T_dev, ['0', '1'])

# LR for Real Images

# Basis Functions
def basis1(X):
    return np.insert(X, 0, 1)

def basis2(X):
    X_new = [i for i in basis1(X)]
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            X_new.append(X[i] * X[j])
    return np.array(X_new)

def calc_basis_vector(x, ord = 3):
    dim = len(x)
    basis_vec = [i for i in basis2(x)]
    for M in range(ord + 1):
        if(dim == 1):
            basis_vec.append(x[0]**M)
        else:
            # Exploiting all possible distribution of powers 
            for p in range(M+1):
                basis_vec.append( (x[0]**(M-p)) )
                basis_vec.append( (x[1]**p) * (x[0]**(M-p)) )
    return np.array(basis_vec)

prefix = 'LR_RealData_'
likelihoods = []
expName = []
accuracies = []
Y_dev_cum = []
X_train_raw, T_train, X_dev_raw, T_dev, nClasses = RealData()

# Experiment 1 - Normalised Data
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)

phi = basis1
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 200, 10**-4.5)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('Normalised Data (Linear Basis)')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 2 - PCA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

directions = PCA(X_train,60)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

np.random.seed(123456789)

phi = basis1
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 250, 10**-4.5)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA (d = 60, Linear Basis)')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 3 - LDA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

directions = LDA(X_train,T_train, nClasses, nClasses - 1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)

phi = calc_basis_vector
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 250, 10**-4.5)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('LDA (d = 4, Cubic Basis)')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 4 - PCA + LDA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

directions = PCA(X_train,60)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

directions = LDA(X_train,T_train, nClasses, nClasses - 1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)

phi = calc_basis_vector
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 250, 10**-4.5)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA(d=60) + LDA(d = 4) Cubic Basis')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
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

# LR - Speech Data

# Basis Functions
def basis1(X):
    return np.insert(X, 0, 1)

def basis2(X):
    X_new = [i for i in basis1(X)]
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            X_new.append(X[i] * X[j])
    return np.array(X_new)

def calc_basis_vector(x, ord = 3):
    dim = len(x)
    basis_vec = [i for i in basis2(x)]
    for M in range(ord + 1):
        if(dim == 1):
            basis_vec.append(x[0]**M)
        else:
            # Exploiting all possible distribution of powers 
            for p in range(M+1):
                basis_vec.append( (x[0]**(M-p)) )
                basis_vec.append( (x[1]**p) * (x[0]**(M-p)) )
    return np.array(basis_vec)

prefix = 'LR_SpeechData_'
likelihoods = []
expName = []
accuracies = []
Y_dev_cum = []
X_train_raw, T_train, X_dev_raw, T_dev, nClasses = speech_data()

# Experiment 1 - Normalised Data
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)

phi = basis1
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 400, 10**-4.5)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('Normalised Data (Linear Basis)')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 2 - PCA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

directions = PCA(X_train, 60)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

np.random.seed(123456789)

phi = basis1
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 400, 10**-4.5)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA (d = 60, Linear Basis)')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 3 - LDA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

directions = LDA(X_train,T_train, nClasses, nClasses - 1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)

phi = calc_basis_vector
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 400, 10**-4.5)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('LDA (d = 4, Cubic Basis)')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 4 - PCA + LDA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
X_train, X_dev = normalise(X_train, X_dev)

directions = PCA(X_train,60)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

directions = LDA(X_train,T_train, nClasses, nClasses - 1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)

phi = calc_basis_vector
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 450, 10**-4.5)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA(d=60) + LDA(d = 4) Cubic Basis')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
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

# LR - Character Data

# Basis Functions
def basis1(X):
    return np.insert(X, 0, 1)

def basis2(X):
    X_new = [i for i in basis1(X)]
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            X_new.append(X[i] * X[j])
    return np.array(X_new)

def calc_basis_vector(x, ord = 3):
    dim = len(x)
    basis_vec = [i for i in basis2(x)]
    for M in range(ord + 1):
        if(dim == 1):
            basis_vec.append(x[0]**M)
        else:
            # Exploiting all possible distribution of powers 
            for p in range(M+1):
                basis_vec.append( (x[0]**(M-p)) )
                basis_vec.append( (x[1]**p) * (x[0]**(M-p)) )
    return np.array(basis_vec)

prefix = 'LR_CharData_'
likelihoods = []
expName = []
accuracies = []
Y_dev_cum = []
X_train_raw, T_train, X_dev_raw, T_dev, nClasses = character_data()

# Experiment 1 - RAW Data
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)

np.random.seed(123456789)

phi = basis1
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 700, 10**-3)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('RAW Data (Linear Basis)')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 2 - PCA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
# X_train, X_dev = normalise(X_train, X_dev)

directions = PCA(X_train, 30)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

np.random.seed(123456789)

phi = basis1
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 700, 10**-3)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA (d = 30, Linear Basis)')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 3 - LDA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
# X_train, X_dev = normalise(X_train, X_dev)

directions = LDA(X_train,T_train, nClasses, nClasses - 1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)

phi = calc_basis_vector
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 700, 10**-3)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('LDA (d = 4, Cubic Basis)')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
print(expName[-1] + ': ' + str(currAcc))

# Experiment 4 - PCA + LDA
X_train, X_dev = np.copy(X_train_raw), np.copy(X_dev_raw)
# X_train, X_dev = normalise(X_train, X_dev)

directions = PCA(X_train, 30)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev, directions)

directions = LDA(X_train,T_train, nClasses, nClasses - 1)
X_train = project_data(X_train, directions)
X_dev = project_data(X_dev,  directions)
X_train, X_dev = normalise(X_train, X_dev)

np.random.seed(123456789)

phi = calc_basis_vector
w = LogisticRegression(X_train, T_train, phi, phi(X_train[0]).shape[0], nClasses, 700, 10**-3)

Y_dev = []
likelihood = []
for n in range(len(X_dev)):
    yn = []
    currTestLikelihood = []
    for j in range(nClasses):
        a = np.array( [w[c] @ phi(X_dev[n]) for c in range(nClasses)])
        softmax = np.exp(a[j]) / sum(np.exp(a))
        currTestLikelihood.append(softmax)
    likelihood.append(currTestLikelihood)
    pred = np.argmax(currTestLikelihood)
    Y_dev.append(pred)
currAcc = accuracy_score(T_dev, Y_dev)
accuracies.append(currAcc)
expName.append('PCA(d=30) + LDA(d = 4) Cubic Basis')
Y_dev_cum.append(Y_dev)
likelihoods.append(likelihood)
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