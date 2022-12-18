import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

org_img = plt.imread('74.jpg') # Reading the gray scale image as 2D numpy array
org_img = np.array(org_img, dtype=np.float64) # Convert to high precision data type.

# Parameters: Any square matrix, with linearly independent eigen vectors
# Returns: Eigen values in descending order based on magnitude and corresponding normalised eigen vectors.
def evd(A):
    EigVal, EigVec = LA.eig(A)
    
    # Sorting in descending order based on magnitude
    idx = abs(EigVal).argsort()[::-1]
    EigVal = EigVal[idx]
    EigVec = EigVec[:, idx]
    
    return EigVec, np.diag(EigVal), LA.inv(EigVec)


# Parameters: Any real matrix
# Returns: Performs SVD and returns U, S, Vt where A = U @ S @ Vt, S has the singular values in descending order.
def svd(A):
    SigmaSqr, U = LA.eigh(A @ A.transpose())    # Returns Eigen Values of a hermitian matrix in Ascending order.
    
    # Convert ascending order to descending order, by using flip
    SigmaSqr = np.flipud(SigmaSqr)
    U = np.fliplr(U)

    Sigma = np.sqrt(SigmaSqr)
    
    # Finding sigma inverse, if we have a zero eigen value we will just leave it as zero in inverse, else do 1 / val
    SigmaInv = np.copy(Sigma)
    idx = np.where(Sigma != 0)
    SigmaInv[idx] = 1/SigmaInv[idx]
    
    # Convert 1D array to 2D diagonal array
    SigmaInv = np.diag(SigmaInv)
    Sigma = np.diag(Sigma)
    
    Vt =  SigmaInv @ (U.transpose() @ A)  # Vt = inv(S) @ ( inv(U) @ A ), and inv(U) = Ut
    return U, Sigma, Vt

# Computes the Forbenius Norm of the given matrix
def frobeniusNorm(A):
    return math.sqrt(np.trace(A @ np.transpose(np.conj(A))).real)




# Plotting SVD reconstructed images and error counter parts
U, S, Vt = svd(org_img)

plt.figure(figsize=(19, 6)) # Overall Grid Size

ctr = 1
for k in [1, 10, 20, 40, 100, 256]:
    Img = U[:,:k] @ S[:k, :k] @ Vt[:k, :] # SVD
    plt.subplot(2, 6, ctr); ctr += 1
    plt.imshow(Img, cmap='gray', vmin = 0, vmax = 255)
    plt.gca().set_title("k = " + str(k), fontweight='bold', color='green', fontsize=16)
    plt.gca().axis('off')
        
    # Corresponding Error Counter Part
    plt.subplot(2, 6, ctr); ctr += 1
    plt.imshow(org_img - Img, cmap='gray', vmin = 0, vmax = 255)
    plt.gca().set_title("(Error) k = " + str(k), fontweight='bold', color='red', fontsize=16)
    plt.gca().axis('off')

plt.savefig('svd_sample.svg')



# SVD Forbenius Norm vs. k
U, S, Vt = svd(org_img)

fig = plt.figure(figsize = (14, 8)) # Overall plot size

MAX = 256
x = [i for i in range(1, MAX+1)]
y = [frobeniusNorm(org_img - (U[:,:x] @ S[:x, :x] @ Vt[:x, :])) for x in range(1, MAX+1)]
plt.plot(x, y)

plt.grid(True, linestyle =':')
plt.xlabel('# of Singular Values', fontsize=16)
plt.ylabel('Forbenius Norm', fontsize=16)
plt.xticks(np.arange(0, MAX+1, 20))

# plt.show()
plt.savefig('svd_norm.svg')



# Plotting EVD reconstructed images and error counter parts
E, S, Einv = evd(org_img)

plt.figure(figsize=(19, 6)) # Overall Grid Size

ctr = 1
for k in [2, 20, 81, 100, 161, 256]:
    # Reconstructed image with first k eigen values
    Img = (E[:,:k] @ S[:k, :k] @ Einv[:k, :]).real
    plt.subplot(2, 6, ctr); ctr += 1
    plt.imshow(Img, cmap='gray', vmin = 0, vmax = 255)
    plt.gca().set_title("k = " + str(k), fontweight='bold', color='green', fontsize=16)
    plt.gca().axis('off')
    
    # Corresponding Error Counter Part
    plt.subplot(2, 6, ctr); ctr += 1
    plt.imshow(org_img - Img, cmap='gray', vmin = 0, vmax = 255)    # Error = Orignal Image - Reconstructed Image
    plt.gca().set_title("(Error) k = " + str(k), fontweight='bold', color='red', fontsize=16)
    plt.gca().axis('off')

plt.savefig('evd_sample.svg')



# EVD Forbenius Norm vs. k
E, S, Einv = evd(org_img)

fig = plt.figure(figsize = (14, 8)) # Overall size of the plot

MAX = 256 # Maximun x value
x = [i for i in range(1, MAX+1)]
# Finding y values corresponding to every x
y = []; k = 1
while(k <= MAX):
    if(abs(S[k-1, k-1].imag) > 1e-6):
        k = k+1 # In case of conjugate Eig. Vals we have to consider both of them, so do k+1
        Norm = frobeniusNorm(org_img - (E[:,:k] @ S[:k, :k] @ Einv[:k, :]))
        y += [Norm, Norm]
    else:
        Norm = frobeniusNorm(org_img - (E[:,:k] @ S[:k, :k] @ Einv[:k, :]))
        y += [Norm]
    k = k+1

plt.plot(x, y)
plt.grid(True, linestyle =':')  # Show Grid Lines
plt.xlabel('k', fontsize=16)
plt.ylabel('Forbenius Norm', fontsize=16)
plt.xticks(np.arange(0, MAX+1, 20))

plt.savefig('evd_norm.svg')


# Experiment3: SVD Ascending Order

# Parameters: Any real matrix
# Returns: Performs SVD and returns U, S, Vt where A = U @ S @ Vt, S has the singular values in ascending order.
def svdAscending(A):
    SigmaSqr, U = LA.eigh(A @ A.transpose())    # Returns Eigen Values of a hermitian matrix in Ascending order.

    Sigma = np.sqrt(SigmaSqr)
    
    SigmaInv = np.copy(Sigma)
    idx = np.where(Sigma != 0)
    SigmaInv[idx] = 1/SigmaInv[idx]
    
    SigmaInv = np.diag(SigmaInv)
    Sigma = np.diag(Sigma)

    # SigmaInv = LA.inv(Sigma)
    
    Vt =  SigmaInv @ (U.transpose() @ A)  # Vt = inv(S) @ ( inv(U) @ A ), and inv(U) = Ut
    return U, Sigma, Vt

# SVD Forbenius Norm vs. k
U, S, Vt = svdAscending(org_img)

fig = plt.figure(figsize = (14, 8)) # Overall plot size

MAX = 256
x = [i for i in range(1, MAX+1)]
y = [frobeniusNorm(org_img - (U[:,:x] @ S[:x, :x] @ Vt[:x, :])) for x in range(1, MAX+1)]
plt.plot(x, y)

plt.grid(True, linestyle =':')
plt.xlabel('# of Singular Values', fontsize=16)
plt.ylabel('Forbenius Norm', fontsize=16)
plt.xticks(np.arange(0, MAX+1, 20))

plt.savefig('svd_norm_ascending.svg')
