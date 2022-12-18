""" PCA and LDA """

def PCA(X,l):
    d = len(X[0])
    N = len(X)
    
    X_mean = np.mean(X,axis = 0)
    X_cov = np.cov(X,rowvar = False)
    
    eig_values,eig_vecs = np.linalg.eigh(X_cov)
    
    sorted_indices = np.argsort(abs(eig_values))[::-1]
    eig_values = eig_values[sorted_indices]
    eig_vecs = eig_vecs[:,sorted_indices]
    
    ans = []
    for i in range(l):ans.append(eig_vecs[:,i].T)
    return np.array(ans)

def project_data(X,W):
    W = np.array(W)
    X_new = []
    for x in X:
        # X_new.append(np.array(np.matrix(x) @ np.matrix(W).T))
        X_new.append(x @ W.T)
    return X_new




######################################################################################################

import scipy
def LDA(X,T,nClasses,l):
    X = np.array(X,dtype = np.float64)

    d = len(X[0])
    # overall mean
    m = np.mean(X,axis = 0)
    
    # categorizing data
    class_X = [[] for i in range(nClasses)]
    for i in range(len(X)):
        class_X[T[i]].append(X[i])
    
    class_mean = [np.mean(class_X[i],axis = 0) for i in range(nClasses)]
    N = [len(class_X[i]) for i in range(nClasses)]
    
    # SW = np.zeros((d,d))
    
    # for i in range(len(X)):
    #     SW += np.matrix(X[i] - class_mean[T[i]]).T @ np.matrix(X[i] - class_mean[T[i]])
        # ST += np.matrix(X[i] - m).T @ np.matrix(X[i] - m)


    _, y_t = np.unique(T, return_inverse=True)  # non-negative ints
    priors = np.bincount(y_t) / float(len(T))

    classes = np.unique(T)
    SW = np.zeros((d,d))
    for idx, group in enumerate(classes):
        Xg = X[T == group, :]
        SW += priors[idx] * np.atleast_2d(empirical_covariance(Xg))
        # print(priors[idx])
    
    
    ST = empirical_covariance(X)
    SB = ST - SW

    
    if(is_pos_def(SW)):
        eig_values,eig_vecs = scipy.linalg.eigh(SB,SW)
    else:
        eig_values,eig_vecs = np.linalg.eigh(np.linalg.pinv(SW) @ (SB))
    
    
    
    sorted_indices = np.argsort((eig_values))[::-1]
    # sorted_indices = get_sorted_indices(eig_values)

    eig_values = eig_values[sorted_indices]
    eig_vecs = eig_vecs[:,sorted_indices]
    
    ans = []
    for i in range(l):ans.append(eig_vecs[:,i].T)
    return ans
    


def get_sorted_indices(eig_values):
    real_eig_values = np.array([abs(value.real) for value in eig_values])
    ans = np.argsort(real_eig_values)[::-1]
    return ans

########################################################################################################



def is_pos_def(x):
    # d = x.shape[0]
    # for i in range(d):
    #     for j in range(i,d):
    #         if(x[i,j] != x[j,i]):
    #             print(i,j,x[i,j],x[j,i])
    # for v in np.linalg.eigvals(x):
    #     if(abs(v) == 0):
    #         print(v)
    return np.all(np.linalg.eigvals(x) > 0)
















