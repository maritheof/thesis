import numpy as np
import math
import random
from thesis import SW_algorithm

#function that produces the data sets for the training of the network, given the range of temperatures (T_min and T_max), 
#the temperature step (T_step), the size of the lattice (N) and the number of configurations per temperature (conf_per_T)
def produce_data(T_min, T_max, T_step, N, conf_per_T):
    #lists to hold the data sets
    training_data = []
    validation_data = []
    test_data = []
    
    #initializing the spin distribution
    spin_distribution = np.full([N,N],1)
    
    for T in np.arange(T_min, T_max, T_step):
        #printing temperature to track the process
        print(T)
        #calling culc_spin_distribution 20 times for the spin distribution to become independent of its starting condition
        spin_distribution = SW_algorithm.culc_spin_distribution(T, spin_distribution, 20)
        
        for k in range(conf_per_T):
            
            spin_distribution = SW_algorithm.culc_spin_distribution(T, spin_distribution, 0)
            training_data.append(shape_data(spin_distribution, N**2))
            
            spin_distribution = SW_algorithm.culc_spin_distribution(T, spin_distribution, 0)
            test_data.append(shape_data(spin_distribution, N**2))
            
            if k % 3 == 0:   
                spin_distribution = SW_algorithm.culc_spin_distribution(T, spin_distribution, 0)
                validation_data.append(shape_data(spin_distribution, N**2))
    
    #saving the data sets in the appropriate files
    np.save('training_data.npy', training_data)
    np.save('validation_data.npy', validation_data)
    np.save('test_data', test_data)

#function used to transform the spin distribution arrays in the appropriate form (lists of tuples) to be used to train the network.
#since the data are used for an autoencoder the input and the desired output are the same and are symbolized by x
def shape_data(spin_distribution, N):
    x = np.eye(N,1)
    n = 0 
    for j in spin_distribution:
        for i in j:
            x[n] = i
            n += 1
    data = (x,x)
    return data    

#function that produces the data set , given the range of temperatures (T_min and T_max), 
#the temperature T_step (T_step), the size of the lattice (N) and the number of configurations per temperature (conf_per_T)
def produce_dat(T_min, T_max, T_step, N, conf_per_T):
    #lists to hold the data sets
    result_data = []
    
    #initial spin distribution
    spin_distribution = np.full([N,N],1)
    
    for T in np.arange(T_min, T_max, T_step):
        #printing temperature to track the process
        print(T)
        #calling culc_spin_distribution 50 times to become independent of its starting condition
        spin_distribution = SW_algorithm.culc_spin_distribution(T, spin_distribution, 50)
        
        for k in range(conf_per_T):
            spin_distribution = SW_algorithm.culc_spin_distribution(T, spin_distribution, 0)
            result_data.append(res_data(spin_distribution, T, N**2))
    
    #saving the data in the appropriate files
    result_data = np.asanyarray(result_data, dtype=object)
    np.save('result_data', result_data)

#function that returns a tuples of the spin distribution with the respective temperature
def res_data(spin_distribution, T, N):
    x = np.eye(N,1)
    n = 0 
    for j in spin_distribution:
        for i in j:
            x[n] = i
            n += 1
    data = (x,T)
    return data 



   
