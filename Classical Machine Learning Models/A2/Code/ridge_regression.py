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


# Vineeth

def mean_squ_err_val(A , B):
    err = ((A - B) @ (A - B).T) / len(A)
    return err

'''
For all the orders - Lamda = 10^-5, because the line is straight for all the values less than that.
'''

'''
Ridge Regression for 1D data
'''
# Reading train and test data of 1D
data = read_data('train',1)
dev_data = read_data('dev',1)

X = []
T = []
for x,t in data:
    X.append([x])
    T.append(t)

X_dev = []
T_dev = []
for x,t in dev_data:
    X_dev.append([x])
    T_dev.append(t)

space = np.logspace(-10, 5, 20)
linspace = np.linspace(-10, 5, 20)
x_ticks = np.linspace(-10, 5, 4)

plt.clf()
plt.figure(figsize=(27, 13))
plt.rcParams.update({'font.size': 16})
for ord in range(1,8,1):
    plt.subplot(2, 4, ord)
    rms_train = []; rms_dev = []

    for lamda in space:
        # Calculating Design Matrix for both train and test data
        DM = calc_design_matrix(X,ord)
        DM_dev = calc_design_matrix(X_dev, ord)

        W = (np.linalg.inv((DM.T @ DM) + lamda * np.identity(DM.shape[1])) @ DM.T) @ np.matrix(T).T

        Y_predicted_train = DM @ W
        Y_predicted_dev = DM_dev @ W

        rms_train.append(math.sqrt(2 * mean_squ_err_val(T , Y_predicted_train.T.A1)))
        rms_dev.append(math.sqrt(2 * mean_squ_err_val(T_dev , Y_predicted_dev.T.A1)))

    plt.plot(linspace, rms_train, 'r')
    plt.plot(linspace, rms_dev, 'g')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.xticks(x_ticks); plt.xlabel('log(lamda)'); plt.ylabel('Erms')
    if(ord >= 5): plt.text(-5, 2.25, 'ORDER: ' + str(ord), fontweight='bold', color = 'blue')
    elif(ord >= 3): plt.text(-5, 2.6, 'ORDER: ' + str(ord), fontweight='bold', color = 'blue')
    else: plt.text(-5, 3.15, 'ORDER: ' + str(ord), fontweight='bold', color = 'blue')
plt.savefig('find_lamda_1d.svg')


# Vineeth

'''
Ridge Regression for 2D data
'''

'''
For all the orders - Lamda = 10^-3, because the line is straight for all the values less than that.
'''
def mean_squ_err_val(A , B):
    err = ((A - B) @ (A - B).T) / len(A)
    return err
    
dim = 2
# Reading train and test data of 1D
data = read_data('train',dim)
dev_data = read_data('dev',dim)

# X = [ ..,[x1, x2] ,.. ] format feature
# T = [..,t,..] format (given target value list)
X = []
T = []
for x1,x2,t in data:
    X.append([x1,x2])
    T.append(t)

X_dev = []
T_dev = []
for x1,x2,t in dev_data:
    X_dev.append([x1,x2])
    T_dev.append(t)

space = np.logspace(-10, 2, 20)
linspace = np.linspace(-10, 2, 20)
x_ticks = [-10, -6, -3, 0, 2]

# Erms for various orders & lamda
plt.clf()
plt.figure(figsize=(27, 13))
plt.rcParams.update({'font.size': 18})
for ord in range(1,8,1):
    plt.subplot(2, 4, ord)
    rms_train = []; rms_dev = []
    
    for lamda in space:
        DM = calc_design_matrix(X,ord)
        W = (np.linalg.inv((DM.T @ DM) + lamda * np.identity(DM.shape[1])) @ DM.T) @ np.matrix(T).T

        Z_predicted_train = []
        for x1,x2 in X:
            Z_predicted_train.append( (calc_basis_vector([x1,x2],ord) @ W).A1[0] )

        Z_predicted_dev = []
        for x1,x2 in X_dev:
            Z_predicted_dev.append( (calc_basis_vector([x1,x2],ord) @ W).A1[0] )

        rms_train.append(math.sqrt(2 * mean_squ_err_val(np.array(T),np.array(Z_predicted_train))))
        rms_dev.append(math.sqrt(2 * mean_squ_err_val(np.array(T_dev),np.array(Z_predicted_dev))))
        
    plt.plot(linspace, rms_train, 'r')
    plt.plot(linspace, rms_dev, 'g')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.xticks(x_ticks)
    plt.xlabel('log(lamda)')
    plt.ylabel('Erms')
    if(ord == 1): plt.text(-4.8, 1, 'ORDER: ' + str(ord), fontweight='bold', color = 'blue')
    else : plt.text(-4.8, 0.8, 'ORDER: ' + str(ord), fontweight='bold', color = 'blue')
plt.savefig('find_lamda_2d.svg')


'''
Ridge Regression for 1D data - plots
'''
# Reading train and test data of 1D

data = read_data('train',1)
test_data = read_data('dev',1)

lamda = 10**-5 # Found out using above code

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

orders = [0] + [i+1 for i in range(12)]
err_list_train = [0]
err_list_test = [0]
plot_orders = [1,3,7,19]
for ord in orders[1:]:
    # Calculating Design Matrix for both train and test data
    DM = calc_design_matrix(X,ord)
    DM_test = calc_design_matrix(X_test, ord)

    # Estimating the parameters by MLE technique
    W = (np.linalg.inv((DM.T @ DM) + lamda * np.identity(DM.shape[1])) @ DM.T) @ np.matrix(T).T
    
    Y_predicted_train = DM @ W
    Y_predicted_test = DM_test @ W

    # Plotting Approximated Function 
    if(ord in plot_orders):
        path = 'RR_1d_approx'+str(ord)+'.png'
        title = 'Approx. Function with #train samples : ' + str(len(X)) + ' , Deg. of Model : ' + str(ord)
        approx_plot_1d(np.matrix(X).A1 , T , Y_predicted_train, title,path)


    # predicted vs target graph for training data
    if(ord in plot_orders):
        path = 'RR_1d_predVsTrue_trainData'+str(ord)+'.svg'
        title = 'Predicted Vs True Target for train data with order ' + str(ord)
        scatter_plot_1d(T , Y_predicted_train.T.A1,title, path)

    # predicted vs target graph for testing data
    if(ord in plot_orders):
        path = 'RR_1d_predVsTrue_testData'+str(ord)+'.svg'
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
special_orders = [1,4,10]
plt.legend(["train", "test"], loc = 'upper left')
for ord in special_orders:
    plt.scatter(ord,err_list_test[ord])
    plt.annotate(str(round(err_list_test[ord],3)),(ord,err_list_test[ord]))
plt.savefig('RR_1d_ord_vs_err.svg')
# plt.show()    

'''
Ridge Regression for 2D data
'''
# Reading train and test data of 1D
data = read_data('train',2)
test_data = read_data('dev',2)

lamda = 10**-3

# X = [ ..,[x1, x2] ,.. ] format feature
# T = [..,t,..] format (given target value list)
X = []
T = []
for x1,x2,t in data:
    X.append([x1,x2])
    T.append(t)
    
scatter_plot_2d(X,T,"3D scatter plot of train data",'RR_2d_trainData_distribution.svg')

X_test = []
T_test = []
for x1,x2,t in test_data:
    X_test.append([x1,x2])
    T_test.append(t)
    
orders = [0] + [i+1 for i in range(9)]
err_list_train = [0]
err_list_test = [0]
plot_orders = [1,3,5,7]
# RR for 2d data over various model complexities
for ord in orders[1:]:
    DM = calc_design_matrix(X,ord)
    W = (np.linalg.inv((DM.T @ DM) + lamda * np.identity(DM.shape[1])) @ DM.T) @ np.matrix(T).T
    
    Z_predicted_train = []
    for x1,x2 in X:
        Z_predicted_train.append( (calc_basis_vector([x1,x2],ord) @ W).A1[0] )
        
    Z_predicted_test = []
    for x1,x2 in X_test:
        Z_predicted_test.append( (calc_basis_vector([x1,x2],ord) @ W).A1[0] )

    if(ord in plot_orders):
        path = 'RR_2d_approx_'+str(ord)+'.png'
        approx_plot_2d(X,W,ord,path)
    if(ord in plot_orders):
        title = 'Predicted Vs True Target for 2d train data with order ' + str(ord)
        path = 'RR_2d_PredVsTrue_trainData'+str(ord)+'.svg'
        scatter_plot_1d(T,Z_predicted_train,title,path)
    if(ord in plot_orders):
        title = 'Predicted Vs True Target for 2d test data with order ' + str(ord)
        path = 'RR_2d_PredVsTrue_testData'+str(ord)+'.svg'
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
plt.legend(["train", "test"], loc = 'upper left')
special_orders = [1,2,3,4,5,6,7]
for ord in special_orders:
    plt.scatter(ord,err_list_test[ord])
    plt.annotate(str(round(err_list_test[ord],3)),(ord,err_list_test[ord]))
plt.savefig('RR_2d_ord_vs_err.svg')