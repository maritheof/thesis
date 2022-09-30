import numpy as np
import math
import random

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

#function that given the temperature, the starting spin_distribution and the iterations returns the new spin_distribution
def culc_spin_distribution(T, spin_distribution, iterations):
    n=0
    N = len(spin_distribution)
    #array that defines the clusters, a cluster is defined as the elements with the same value in the array
    cluster = np.full((N,N),-1)   
    
    #loop that defines the clusters by only taking into account the bonds made by neighboring spins in the rows of the array cluster 
    for j in range(N):            
        n +=1
        cluster[j,0] = n
        for i in range(N-1):
            if spin_distribution[j,i] == spin_distribution[j,i+1] and bond_probability(T) == 1:
                    cluster[j,i+1] = cluster[j,i]
            else:
                n += 1
                cluster[j,i+1] = n 
    
    #loop that defines clusters by taking into account the bonds made by neighboring spins at the columns of the array cluster
    #the bonds made in the rows before are considered
    for i in range(N):          
        for j in range(N-1):
            if spin_distribution[j+1,i] == spin_distribution[j,i] and bond_probability(T) == 1 and cluster[j+1,i] != cluster[j,i]:
                cluster = np.where(cluster == cluster[j+1,i], cluster[j,i], cluster)
    
    #create dictionary with the cluster values as keys and a random spin for every cluster value
    cluster_dict = {k: set_random_spin() for k in range(1,n+1)}         
    
    #loop that gives new spin values to the spin_disribution
    for i in range(N):                                                  
        for j in range(N):
            spin_distribution[i,j] = cluster_dict[cluster[i,j]]
    
    #loop that calls the function culc_spin_distribution again as many times as iterations define
    if iterations > 1:      
        iterations -= 1
        culc_spin_distribution(T, spin_distribution, iterations)
    return(spin_distribution)

