import matplotlib.pyplot as plt
import numpy as np
import csv
import math


'''
Reading data from required txt file and returning in form of float-typed list of lists
'''
# cat belongs to [ 'train' , 'dev']
# dim belongs to [ 1 , 2 ] represents dimensions of feature vector
# return is in format of [ ..,[x1, target],.. ] or [ ..,[x1 , x2 , target],..] (depending on dim value)
def read_data(cat , dim):
    path = str(dim) + 'd_team_2_' + cat + '.txt'
    data = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines :
            data.append(list(map(float,line.split())))
        f.close()
    return data




# Hakesh

# x = feature vector ([x] form or [x1,x2] form) 
# ord = order of model
# Return : basis vector as normal python list
def calc_basis_vector(x,ord):
    dim = len(x)
    basis_vec = []
    for M in range(ord + 1):
        if(dim == 1):
            basis_vec.append(x[0]**M)
        else:
            # Exploiting all possible distribution of powers 
            for p in range(M+1):
                basis_vec.append( (x[1]**p) * (x[0]**(M-p)) )
    return basis_vec
                
            
# X = list of features = [ ..,[x0] ,..] or [ ..,[x[0],x[1]],..] (depending on feature dimensions) 
# ord = integer = representing order of linear regression
# Return : numpy matrix representing design matrix corresponding list of features X.
def calc_design_matrix(X,ord):
    design_matrix = []
    dim = len(X[0])
    # For each feature vector
    for x in X:
        design_matrix.append(calc_basis_vector(x,ord))
    return np.matrix(design_matrix)


# Prediction vs true target value scatter plot
def scatter_plot_1d(X,Y,title,path):
    plt.clf();     fig = plt.figure(figsize = (14, 8))
    plt.scatter(X,Y)
    plt.title(title)
    plt.savefig(path)

# X = [..,[x1,x2],..] form feature list
# T = [.., t ,..] form target value list
def scatter_plot_2d(X,T,title,path):
    # Scatter Plot for data visualization
    X1 = [x for x,_ in X]
    X2 = [x for _,x in X]
    plt.clf();     fig = plt.figure(figsize = (14, 8))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(X1,X2,T, color = "green")
    plt.title(title)
#     plt.show()
    plt.savefig(path)
    
    
# X = [..,[x1,x2],..] form feature list
# T = [.., t ,..] form target value list
# Y = [.., y ,..] form predicted value list
def approx_plot_1d(X , T , Y , title,path):
    plt.clf();     fig = plt.figure(figsize = (14, 8))
    plt.scatter(X , T, c = 'white' , linewidth = 2 , edgecolor = "#03AC13" , s = 20)
    plt.plot(np.matrix(X).A1 , Y)
    plt.title( title )
#     plt.show()
    plt.savefig(path)

# X = [..,[x1,x2],..] form feature list
# W = numpy matrix of dimension  (M x 1) = representing parameter/coefficients of model
# ord = integer = model degree in 2d
def approx_plot_2d(X,W,ord,path):
    X1 = [x for x,_ in X]
    X2 = [x for _,x in X]
    X1_mesh , X2_mesh = np.meshgrid(X1,X2)
    
    Z_mesh = np.zeros(X1_mesh.shape)
    for i in range(X1_mesh.shape[0]):
        for j in range(X2_mesh.shape[1]):
            basis_vec = calc_basis_vector([X1_mesh[i,j],X2_mesh[i,j]] , ord)
            Z_mesh[i,j] = basis_vec @ W 
    
    plt.clf();     fig = plt.figure(figsize = (14, 8))
    ax = plt.axes(projection="3d")
    plt.title("Appro. Surface with ord : " + str(ord))
    ax.plot_surface(X1_mesh , X2_mesh , Z_mesh)
#     plt.show()
    plt.savefig(path, dpi = 100)
    

# The following function calculate mean square error for vectors A and B (1 x n format each)
# A and B : simple numpy array's with dimension of form 1 x n.
def mean_squ_err(A , B):
    err = 0
    for i in range(len(A)):
        err += (A[i] - B[i])**2
    err = math.sqrt(err/len(A))

    return err

# Hakesh
'''
Least Square Regression for 1D data
'''
# Reading train and test data of 1D

data = read_data('train',1)
test_data = read_data('dev',1)

X = []
T = []
for x,t in data:
    X.append([x])
    T.append(t)

X_test = []
T_test = []
for x,t in test_data:
    X_test.append([x])
    T_test.append(t)

orders = [0] + [i+1 for i in range(50)]
err_list_train = [0]
err_list_test = [0]
plot_orders = [1,3,7,19]
for ord in orders[1:]:
    # Calculating Design Matrix for both train and test data
    DM = calc_design_matrix(X,ord)
    DM_test = calc_design_matrix(X_test, ord)

    # Estimating the parameters by MLE technique
    W = (np.linalg.pinv(DM)) @ (np.matrix(T)).T
    
    Y_predicted_train = DM @ W
    Y_predicted_test = DM_test @ W

    # Plotting Approximated Function 
    if(ord in plot_orders):
        path = 'LSR_1d_approx'+str(ord)+'.png'
        title = 'Approx. Function with #train samples : ' + str(len(X)) + ' , Deg. of Model : ' + str(ord)
        approx_plot_1d(np.matrix(X).A1 , T , Y_predicted_train, title,path)


    # predicted vs target graph for training data
    if(ord in plot_orders):
        path = 'LSR_1d_predVsTrue_trainData'+str(ord)+'.svg'
        title = 'Predicted Vs True Target for train data with order ' + str(ord)
        scatter_plot_1d(T , Y_predicted_train.T.A1,title, path)

    # predicted vs target graph for testing data
    if(ord in plot_orders):
        path = 'LSR_1d_predVsTrue_testData'+str(ord)+'.svg'
        title = 'Predicted Vs True Target for test data with order ' + str(ord)
        scatter_plot_1d(T_test , Y_predicted_test.T.A1 , title, path)
        
    # Mean Error for training data
    err = mean_squ_err(T , Y_predicted_train.T.A1)
    err_list_train.append(err)

    # Mean Error for test data
    err = mean_squ_err(T_test , Y_predicted_test.T.A1)
    err_list_test.append(err)

plt.clf();     fig = plt.figure(figsize = (14, 8))
plt.title('order vs. error for train & test - Erms for test are annotated')
plt.xlabel('Order of Model')
plt.ylabel('RMS Error')
plt.plot(orders[1:],err_list_train[1:])
plt.plot(orders[1:],err_list_test[1:])    
plt.legend(["train", "test"], loc = 'upper left')
special_orders = [1,4,10,19,24,30,36]
for ord in special_orders:
    plt.scatter(ord,err_list_test[ord])
    plt.annotate(str(round(err_list_test[ord],3)),(ord,err_list_test[ord]))
plt.savefig('LSR_1d_ord_vs_err.svg')


# Hakesh
'''
Least Square Regression for 2D data
'''
# Reading train and test data of 1D
data = read_data('train',2)
test_data = read_data('dev',2)

# X = [ ..,[x1, x2] ,.. ] format feature
# T = [..,t,..] format (given target value list)
X = []
T = []
for x1,x2,t in data:
    X.append([x1,x2])
    T.append(t)
    
scatter_plot_2d(X,T,"3D scatter plot of train data",'LSR_2d_trainData_distribution.svg')

X_test = []
T_test = []
for x1,x2,t in test_data:
    X_test.append([x1,x2])
    T_test.append(t)
    
orders = [0] + [i+1 for i in range(9)]
err_list_train = [0]
err_list_test = [0]
plot_orders = [1,3,5,7]
# LSR for 2d data over various model complexities
for ord in orders[1:]:
    DM = calc_design_matrix(X,ord)
    W = (np.linalg.pinv(DM)) @ (np.matrix(T)).T
    
    Z_predicted_train = []
    for x1,x2 in X:
        Z_predicted_train.append( (calc_basis_vector([x1,x2],ord) @ W).A1[0] )
        
    Z_predicted_test = []
    for x1,x2 in X_test:
        Z_predicted_test.append( (calc_basis_vector([x1,x2],ord) @ W).A1[0] )

    if(ord in plot_orders):
        path = 'LSR_2d_approx_'+str(ord)+'.png'
        approx_plot_2d(X,W,ord,path)
    if(ord in plot_orders):
        title = 'Predicted Vs True Target for 2d train data with order ' + str(ord)
        path = 'LSR_2d_PredVsTrue_trainData'+str(ord)+'.svg'
        scatter_plot_1d(T,Z_predicted_train,title,path)
    if(ord in plot_orders):
        title = 'Predicted Vs True Target for 2d test data with order ' + str(ord)
        path = 'LSR_2d_PredVsTrue_testData'+str(ord)+'.svg'
        scatter_plot_1d(T_test,Z_predicted_test,title,path)
    
    title = 'Mean Square Error for 2d train data by order'+str(ord)+' is :'
    err = mean_squ_err(np.array(T),np.array(Z_predicted_train))
    err_list_train.append(err)
    
    title = 'Mean Square Error for 2d test data by order'+str(ord)+' is :'
    err = mean_squ_err(np.array(T_test),np.array(Z_predicted_test))
    err_list_test.append(err)
    
plt.clf(); fig = plt.figure(figsize = (14, 8))
plt.title('order vs. error for train & test - Erms for test are annotated')
plt.xlabel('Order of Model')
plt.ylabel('RMS Error')
plt.plot(orders[1:],err_list_train[1:])
plt.plot(orders[1:],err_list_test[1:])    
special_orders = [1,2,3,4,5,6,7,8,9]
plt.legend(["train", "test"], loc = 'upper left')
for ord in special_orders:
    plt.scatter(ord,err_list_test[ord])
    plt.annotate(str(round(err_list_test[ord],3)),(ord,err_list_test[ord]))
plt.savefig('LSR_2d_ord_vs_err.svg')


