import numpy as np
import math
import random

#function that take as input the temperature and calculates the probability of two spins formating a bond
def bond_probability(T, J=1):  
    p = 1 - math.exp(-(2*J)/T)    #probability of formating a bond in Swendsen-Wang cluster algorithm 
    r = random.random()
    if r <= p:
        return 1                  #bond
    else:
        return 0                  #no bond
#function that take as input the parameter q and returns an integer number that belong to [0,q]
def set_random_spin(q):  
    n = random.randint(0, q-1)
    return n

#function that given the temperature, the starting spin_distribution and the iterations returns the new spin_distribution
def culc_spin_distribution(T, spin_distribution, iterations, q):
    n=0
    N = len(spin_distribution)
    cluster = np.full((N,N),-1)   #array that defines the clusters, a cluster is defined as the all the elements that have the same value in the array
    
    for j in range(N):            #loop that defines the clusters by only taking into account the bonds made by neighboring spins in the rows of the array cluster 
        n +=1
        cluster[j,0] = n
        for i in range(N-1):
            if spin_distribution[j,i] == spin_distribution[j,i+1] and bond_probability(T) == 1:
                    cluster[j,i+1] = cluster[j,i]
            else:
                n += 1
                cluster[j,i+1] = n 
    
    for i in range(N):           #loop that defines clusters by taking into account the bonds made by neighboring spins at the columns of the array cluster (considering the bonds made in the rows before)
        for j in range(N-1):
            if spin_distribution[j+1,i] == spin_distribution[j,i] and bond_probability(T) == 1 and cluster[j+1,i] != cluster[j,i]:
                cluster = np.where(cluster == cluster[j+1,i], cluster[j,i], cluster)
    
    cluster_dict = {k: set_random_spin(q) for k in range(1,n+1)}         #create dictionary with the cluster values as keys and a random spin for every cluster value
    for i in range(N):                                                  #loop that runs through clusters and gives new spin values
        for j in range(N):
            cluster[i,j] = cluster_dict[cluster[i,j]]
    spin_distribution = cluster
    
    if iterations > 1:      #loop that calls the function again as many times as the numder_of_loops defines
        iterations -= 1
        culc_spin_distribution(T, spin_distribution, iterations, q)
    return(spin_distribution)
