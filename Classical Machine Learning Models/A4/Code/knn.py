# Imports 

exec(open("./libraries.py").read())
exec(open("./read_input.py").read())
exec(open("./roc_det.py").read())
exec(open("./lda_pca.py").read())

path_pref = ''
types = ['nodr','pca','lda']
types_names = ['nodr','PCA','LDA']

# KNN



def KNN(X_train,T_train,nClasses,test_point,K):
    distances = np.array([np.linalg.norm(x - test_point) for x in X_train])
    indices = np.argsort(distances)
    distances = distances[indices]

    sort_T_train = np.array(T_train)[indices]
    
    cnt = [0 for i in range(nClasses)]
    for c in sort_T_train[:K]:
        cnt[c] += 1
    return np.argmax(cnt),cnt
        
def calc_accuracy(X_train,T_train,X_dev,T_dev,nClasses,K):
    ans = 0
    N_dev = len(X_dev)
    likelihood = []
    for i in range(N_dev):
        test_point = X_dev[i]

        pred,cnt= KNN(X_train,T_train,nClasses,test_point,K)
        if(pred == T_dev[i]):
            ans += 1
            
        likelihood.append([cnt[i]/K for i in range(nClasses)])
            
    return (ans / N_dev ),likelihood
    

# Synthetic Data

class_names = ['1','2']
dir = 'knn_plots/'
X_train,T_train,X_dev,T_dev,nClasses = Synthetic()
X_train,X_dev = normalise(X_train, X_dev)



# execfile('lda_pca.py')
l = 1

PCA_dir = PCA(X_train,l)
X_train_PCA = project_data(X_train, PCA_dir)
X_dev_PCA = project_data(X_dev, PCA_dir)

LDA_dir = LDA(X_train,T_train,nClasses,l)
X_train_LDA = project_data(X_train, LDA_dir)
X_dev_LDA = project_data(X_dev, LDA_dir)

Ks = [2,3,5,8,15,30]

train_data = [X_train,X_train_PCA,X_train_LDA]
dev_data = [X_dev,X_dev_PCA,X_dev_LDA]
x_axis = [[] for i in range(3)]
y_axis = [[] for i in range(3)]
all_types_likelihoods = [[] for i in range(3)]
accuracies = [[] for i in range(3)]

for i in range(3):
    train = train_data[i]
    dev = dev_data[i]
    for k in Ks:    
        acc,likelihood = calc_accuracy(train,T_train,dev,T_dev,nClasses,k)
        
        x_axis[i].append(k)
        y_axis[i].append(acc)
        all_types_likelihoods[i].append(likelihood)
        accuracies[i].append(acc)
        
        print(f'Acc : {acc} for K = {k} , data type : {i}')
    print()

""" ROC - DET - Confusion Matrix """
# execfile('roc_det.py')
nTests = len(X_dev)

plt.clf()

# confusion Matrix
filter_Ks = [8]    
for i in range(len(types)):
    path = dir 
    for j in range(len(Ks)):
        if(Ks[j] not in filter_Ks) : continue
        acc = round(accuracies[i][j] * 100,2)
        title = f'CM with {types_names[i]} & K = {Ks[j]}::Acc={acc}%'
        path_suff = f'syn_CM_K{Ks[j]}_{types_names[i]}.svg'
        confusionMatrix(title,path_pref + path + path_suff,all_types_likelihoods[i][j],
        nTests,nClasses,T_dev,class_names)
    
    
######## Roc
path = dir
colors = ['red','blue','green']

leg = []
for i in range(len(types)):
    for j in range(len(Ks)):
        K = Ks[j]
        if(K != 5 and K != 8):continue
        acc = round(accuracies[i][j] * 100, 2)
        plt.plot(color = colors[i])
        TPR,FPR = ROC(all_types_likelihoods[i][j],nTests,nClasses,T_dev)
        plt.plot(FPR, TPR, color = colors[i])
        leg.append(f'{types_names[i]}::K={K} Acc:{acc} Mis:{100-acc}')
    

plt.legend(leg)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'No Dim-Red Vs PCA vs LDA for some Ks')
path_suff = f'syn_all_ROC.svg'
plt.savefig(path_pref + path + path_suff)
plt.clf()



####### DET
# execfile('roc_det.py')
ax = plt.gca()
# filter_Ks=[[5,8],[15,30],[15,30]]
for i in range(len(types)):
#     if(i == 2):continue
    for j in range(len(Ks)):
        K = Ks[j]
        if(K != 15 and K != 30):continue
        DET(all_types_likelihoods[i][j],nTests,nClasses,T_dev,f'{types_names[i]}::K={K}').plot(ax, color = colors[i])
        
plt.xlabel('FPR')
plt.ylabel('FNR')
plt.title(f'DET : no dim-red vs PCA vs LDA')
path_suff = f'syn_all_DET.svg'
plt.savefig(path_pref + path + path_suff)
# plt.show()
plt.clf() 








# Real Image Data 

class_names = ['coast','forest','highway','mountain','opencountry']
dir = 'knn_plots/'
X_train,T_train,X_dev,T_dev,nClasses = RealData()

X_train,X_dev = normalise(X_train, X_dev)



# execfile('lda_pca.py')
l = 4

PCA_dir = PCA(X_train , 100)
X_train_PCA = project_data(X_train, PCA_dir)
X_dev_PCA = project_data(X_dev, PCA_dir)
LDA_dir = LDA(X_train_PCA, T_train, nClasses,l)
X_train_LDA = project_data(X_train_PCA, LDA_dir)
X_dev_LDA = project_data(X_dev_PCA, LDA_dir)


PCA_dir = PCA(X_train , l)
X_train_PCA = project_data(X_train, PCA_dir)
X_dev_PCA = project_data(X_dev, PCA_dir)

Ks = [2,3,5,8,15,30,50]

train_data = [X_train,X_train_PCA,X_train_LDA]
dev_data = [X_dev,X_dev_PCA,X_dev_LDA]
x_axis = [[] for i in range(3)]
y_axis = [[] for i in range(3)]
all_types_likelihoods = [[] for i in range(3)]
accuracies = [[] for i in range(3)]


for i in range(3):
    train = train_data[i]
    dev = dev_data[i]
    for k in Ks:    
        acc,likelihood = calc_accuracy(train,T_train,dev,T_dev,nClasses,k)
        
        x_axis[i].append(k)
        y_axis[i].append(acc)
        all_types_likelihoods[i].append(likelihood)
        accuracies[i].append(acc)
        
        print(f'Acc : {acc} for K = {k} , data type : {i}')
    print()

""" ROC - DET - Confusion Matrix """
# execfile('roc_det.py')
nTests = len(X_dev)

plt.clf()

# confusion Matrix
filter_Ks = [15]    
for i in range(len(types)):
    path = dir 
    for j in range(len(Ks)):
        if(Ks[j] not in filter_Ks) : continue
        acc = round(accuracies[i][j] * 100,2)
        title = f'CM with {types_names[i]} & K = {Ks[j]}::Acc={acc}%'
        path_suff = f'real_CM_K{Ks[j]}_{types_names[i]}.svg'
        confusionMatrix(title,path_pref + path + path_suff,all_types_likelihoods[i][j],
        nTests,nClasses,T_dev,class_names)
    
    
######## Roc
path = dir
colors = ['red','blue','green']

leg = []
for i in range(len(types)):
    for j in range(len(Ks)):
        K = Ks[j]
        if(K != 15 and K != 8):continue
        acc = round(accuracies[i][j] * 100, 2)
        plt.plot(color = colors[i])
        TPR,FPR = ROC(all_types_likelihoods[i][j],nTests,nClasses,T_dev)
        plt.plot(FPR, TPR, color = colors[i])
        leg.append(f'{types_names[i]}::K={K} Acc:{acc} Mis:{100-acc}')
    

plt.legend(leg)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'No Dim-Red Vs PCA vs LDA for some Ks')
path_suff = f'real_all_ROC.svg'
plt.savefig(path_pref + path + path_suff)
plt.clf()



####### DET
# execfile('roc_det.py')
ax = plt.gca()
# filter_Ks=[[5,8],[15,30],[15,30]]
for i in range(len(types)):
#     if(i == 2):continue
    for j in range(len(Ks)):
        K = Ks[j]
        if(K != 15 and K != 30):continue
        DET(all_types_likelihoods[i][j],nTests,nClasses,T_dev,f'{types_names[i]}::K={K}').plot(ax, color = colors[i])
        
plt.xlabel('FPR')
plt.ylabel('FNR')
plt.title(f'DET : no dim-red vs PCA vs LDA')
path_suff = f'real_all_DET.svg'
plt.savefig(path_pref + path + path_suff)
# plt.show()
plt.clf() 


# speech_data



class_names = ['1','4','5','7','o']
dir = 'knn_plots/'
X_train,T_train,X_dev,T_dev,nClasses = speech_data()
X_train,X_dev = normalise(X_train, X_dev)

# execfile('lda_pca.py')
l = 4

PCA_dir = PCA(X_train,l)
X_train_PCA = project_data(X_train, PCA_dir)
X_dev_PCA = project_data(X_dev, PCA_dir)

LDA_dir = LDA(X_train,T_train,nClasses,l)
X_train_LDA = project_data(X_train, LDA_dir)
X_dev_LDA = project_data(X_dev, LDA_dir)




Ks = [2,3,5,8,15,30]

train_data = [X_train,X_train_PCA,X_train_LDA]
dev_data = [X_dev,X_dev_PCA,X_dev_LDA]
x_axis = [[] for i in range(3)]
y_axis = [[] for i in range(3)]
all_types_likelihoods = [[] for i in range(3)]
accuracies = [[] for i in range(3)]


for i in range(3):
    train = train_data[i]
    dev = dev_data[i]
    for k in Ks:    
        acc,likelihood = calc_accuracy(train,T_train,dev,T_dev,nClasses,k)
        
        x_axis[i].append(k)
        y_axis[i].append(acc)
        all_types_likelihoods[i].append(likelihood)
        accuracies[i].append(acc)
        
        print(f'Acc : {acc} for K = {k} , data type : {i}')
    print()

""" ROC - DET - Confusion Matrix """
# execfile('roc_det.py')
nTests = len(X_dev)

plt.clf()

# confusion Matrix
filter_Ks = [8]    
for i in range(len(types)):
    path = dir 
    for j in range(len(Ks)):
        if(Ks[j] not in filter_Ks) : continue
        acc = round(accuracies[i][j] * 100,2)
        title = f'CM with {types_names[i]} & K = {Ks[j]}::Acc={acc}%'
        path_suff = f'speech_CM_K{Ks[j]}_{types_names[i]}.svg'
        confusionMatrix(title,path_pref + path + path_suff,all_types_likelihoods[i][j],
        nTests,nClasses,T_dev,class_names)
    
    
######## Roc
path = dir
colors = ['red','blue','green']

leg = []
for i in range(len(types)):
    for j in range(len(Ks)):
        K = Ks[j]
        if(K != 15 and K != 8):continue
        acc = round(accuracies[i][j] * 100, 2)
        plt.plot(color = colors[i])
        TPR,FPR = ROC(all_types_likelihoods[i][j],nTests,nClasses,T_dev)
        plt.plot(FPR, TPR, color = colors[i])
        leg.append(f'{types_names[i]}::K={K} Acc:{acc} Mis:{100-acc}')
    

plt.legend(leg)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'No Dim-Red Vs PCA vs LDA for some Ks')
path_suff = f'speech_all_ROC.svg'
plt.savefig(path_pref + path + path_suff)
plt.clf()



####### DET
# execfile('roc_det.py')
ax = plt.gca()
# filter_Ks=[[5,8],[15,30],[15,30]]
for i in range(len(types)):
#     if(i == 2):continue
    for j in range(len(Ks)):
        K = Ks[j]
        if(K != 15 and K != 30):continue
        DET(all_types_likelihoods[i][j],nTests,nClasses,T_dev,f'{types_names[i]}::K={K}').plot(ax, color = colors[i])
        
plt.xlabel('FPR')
plt.ylabel('FNR')
plt.title(f'DET : no dim-red vs PCA vs LDA')
path_suff = f'speech_all_DET.svg'
plt.savefig(path_pref + path + path_suff)
# plt.show()
plt.clf() 


# character_data

class_names = ['ai','bA','dA','lA','tA']
dir = 'knn_plots/'
X_train,T_train,X_dev,T_dev,nClasses = character_data()
X_train,X_dev = normalise(X_train, X_dev)

# execfile('lda_pca.py')
l = 4


PCA_dir = PCA(X_train,l)
X_train_PCA = project_data(X_train, PCA_dir)
X_dev_PCA = project_data(X_dev, PCA_dir)

LDA_dir = LDA(X_train,T_train,nClasses,l)
X_train_LDA = project_data(X_train, LDA_dir)
X_dev_LDA = project_data(X_dev, LDA_dir)





Ks = [2,3,5,8,15,30]

train_data = [X_train,X_train_PCA,X_train_LDA]
dev_data = [X_dev,X_dev_PCA,X_dev_LDA]
x_axis = [[] for i in range(3)]
y_axis = [[] for i in range(3)]
all_types_likelihoods = [[] for i in range(3)]
accuracies = [[] for i in range(3)]


for i in range(3):
    train = train_data[i]
    dev = dev_data[i]
    for k in Ks:    
        acc,likelihood = calc_accuracy(train,T_train,dev,T_dev,nClasses,k)
        
        x_axis[i].append(k)
        y_axis[i].append(acc)
        all_types_likelihoods[i].append(likelihood)
        accuracies[i].append(acc)
        
        print(f'Acc : {acc} for K = {k} , data type : {i}')
    print()

""" ROC - DET - Confusion Matrix """
# execfile('roc_det.py')
nTests = len(X_dev)

plt.clf()

# confusion Matrix
filter_Ks = [15]    
for i in range(len(types)):
    path = dir  
    for j in range(len(Ks)):
        if(Ks[j] not in filter_Ks) : continue
        acc = round(accuracies[i][j] * 100,2)
        title = f'CM with {types_names[i]} & K = {Ks[j]}::Acc={acc}%'
        path_suff = f'char_CM_K{Ks[j]}_{types_names[i]}.svg'
        confusionMatrix(title,path_pref + path + path_suff,all_types_likelihoods[i][j],
        nTests,nClasses,T_dev,class_names)
    
    
######## Roc
path = dir
colors = ['red','blue','green']

leg = []
for i in range(len(types)):
    for j in range(len(Ks)):
        K = Ks[j]
        if(K != 15 and K != 8):continue
        acc = round(accuracies[i][j] * 100, 2)
        plt.plot(color = colors[i])
        TPR,FPR = ROC(all_types_likelihoods[i][j],nTests,nClasses,T_dev)
        plt.plot(FPR, TPR, color = colors[i])
        leg.append(f'{types_names[i]}::K={K} Acc:{acc} Mis:{100-acc}')
    

plt.legend(leg)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'No Dim-Red Vs PCA vs LDA for some Ks')
path_suff = f'char_all_ROC.svg'
plt.savefig(path_pref + path + path_suff)
plt.clf()



####### DET
# execfile('roc_det.py')
ax = plt.gca()
# filter_Ks=[[5,8],[15,30],[15,30]]
for i in range(len(types)):
#     if(i == 2):continue
    for j in range(len(Ks)):
        K = Ks[j]
        if(K != 15 and K != 30):continue
        DET(all_types_likelihoods[i][j],nTests,nClasses,T_dev,f'{types_names[i]}::K={K}').plot(ax, color = colors[i])
        
plt.xlabel('FPR')
plt.ylabel('FNR')
plt.title(f'DET : no dim-red vs PCA vs LDA')
path_suff = f'char_all_DET.svg'
plt.savefig(path_pref + path + path_suff)
print(path_pref + path + path_suff)
# plt.show()
plt.clf() 
