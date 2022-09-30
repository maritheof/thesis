import numpy as np
import math
import random

#function that given the temperature, the starting spin distribution and the iterations returns the new spin_distribution
def culc_spin_distribution(T, spin_distribution, iterations):
    N = len(spin_distribution)
    clusters = np.full((N,N),-1)  #array that defines the clusters, a cluster is defined as the elements with the same cluster ID in the array
    cluster_ID = 0                #the spins that belong to the same cluster have a same cluster ID 
    
    #loop that defines the clusters by only taking into account the bonds that are made at the rows
    for j in range(N):            
        cluster_ID += 1
        clusters[j,0] = cluster_ID
        for i in range(N-1):
            if spin_distribution[j,i] == spin_distribution[j,i+1] and bond_probability(T) == 1:
                    clusters[j,i+1] = clusters[j,i]
            else:
                cluster_ID += 1
                clusters[j,i+1] = cluster_ID 
    
    #loop that defines clusters by taking into account the bonds that are made at the columns (the bonds made in the rows before are beeing considered)
    for i in range(N):          
        for j in range(N-1):
            if spin_distribution[j+1,i] == spin_distribution[j,i] and bond_probability(T) == 1 and clusters[j+1,i] != clusters[j,i]:
                clusters = np.where(clusters == clusters[j+1,i], clusters[j,i], clusters)
    
    #create dictionary with the cluster ID as key and a random spin value for every cluster ID
    clusters_dict = {k: set_random_spin() for k in range(1,cluster_ID+1)}         
    
    #loop that gives new spin values to the spin_disribution
    for i in range(N):                                                  
        for j in range(N):
            spin_distribution[i,j] = clusters_dict[clusters[i,j]]
    
    #loop that calls the function culc_spin_distribution again as many times as iterations define
    if iterations > 1:      
        iterations -= 1
        culc_spin_distribution(T, spin_distribution, iterations)
    
    return(spin_distribution)


#function that take as input the temperature and calculates the probability of two spins formating a bond
def bond_probability(T, J=1):  
    p = 1 - math.exp(-(2*J)/T)    #probability for two spins to form a bond in Ising model
    r = random.random()
    if r <= p:
        return 1                  #bond
    else:
        return 0                  #no bond
    
    
#function that returns the spin value -1 or +1 with equal probability
def set_random_spin():  
    r = random.random()
    if r>0.5:
        return -1
    else:
        return 1
