import numpy as np
from numpy import linalg as LA
import statsmodels
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Load prepared coefficients
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def load_coefficients_pod(num_modes):
    #phi = np.load('./Coefficients/POD_Modes.npy')
    #cf = np.load('./Coefficients/Coeffs_train.npy')
    #smean = np.load('./Coefficients/Snapshot_Mean.npy')
    phi = np.load('./Coefficients/POD_Modes_svd.npy')
    cf = np.load('./Coefficients/Coeffs_data.npy')

    # Do truncation
    phi = phi[:,0:num_modes] # Columns are modes
    cf = cf[0:num_modes,:] #Columns are time, rows are modal coefficients

    return phi, cf 


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Generate POD basis from svd
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def generate_pod_bases_from_svd(snapshot_matrix,num_modes): #Mean removed
    '''
    Takes input of a snapshot matrix and computes POD bases
    Outputs truncated POD bases and coefficients
    '''
    u,s,v = LA.svd(snapshot_matrix) # (3340, 3340) (53,) (53, 53)
    phi = u[:,:np.shape(snapshot_matrix)[1]]
    coeff_matrix = np.matmul(np.transpose(phi), snapshot_matrix)
    
    print(f"snapshot train : {snapshot_matrix.shape}")
    print(f"phi : {phi.shape}")
    print(f"u,s,v: {u.shape}, {s.shape}, {v.shape}")
    print(f"coeff : {coeff_matrix.shape}")

    # Output amount of energy retained
    print('Amount of energy retained:',np.sum(s[:num_modes])/np.sum(s))
    input('Press any key to continue')

    np.save('./Coefficients/POD_Modes_svd.npy',phi)
    np.save('./Coefficients/Coeffs_data.npy',coeff_matrix)

    return phi, coeff_matrix


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Generate POD basis
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def generate_pod_bases(snapshot_matrix_train,num_modes): #Mean removed
    '''
    Takes input of a snapshot matrix and computes POD bases
    Outputs truncated POD bases and coefficients
    '''
    new_mat = np.matmul(np.transpose(snapshot_matrix_train),snapshot_matrix_train)

    w,v = LA.eig(new_mat)

    print(new_mat.shape)

    # Bases
    phi = np.real(np.matmul(snapshot_matrix_train,v))
    trange = np.arange(np.shape(snapshot_matrix_train)[1])
    phi[:,trange] = phi[:,trange]/np.sqrt(w[:])

    print(f"snapshot train : {snapshot_matrix_train.shape}")
    print(f"phi : {phi.shape}")
    print(f"w: {w[:num_modes]}")
    print(f"sum(w): {np.sum(w)}")
    print(f"w/sum(w): {w[:num_modes] / np.sum(w[:num_modes])}")
    print(f"v: {v.shape}")
    coefficient_matrix = np.matmul(np.transpose(phi),snapshot_matrix_train)
    print(f"coeff train : {coefficient_matrix.shape}")

    # Output amount of energy retained
    print('Amount of energy retained:',np.sum(w[:num_modes])/np.sum(w))
    input('Press any key to continue')

    np.save('./Coefficients/POD_Modes.npy',phi)
    np.save('./Coefficients/Coeffs_train.npy',coefficient_matrix)

    return phi, coefficient_matrix



#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Plot POD modes
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def plot_pod_modes(phi,mode_num):
    plt.figure()
    plt.plot(phi[:,mode_num])
    plt.show()

def plot_pod_w(w):
    plt.figure()
    plt.plot(np.log(w))
    plt.show()
if __name__ == '__main__':
    pass
