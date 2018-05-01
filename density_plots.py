#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:07:55 2018

@author: ibarlow
"""

#density plot function - adapted from DCplot package

def density_calc (dist, XY, ND, rho, delta,ordrho,dc,nneigh, rhomin,deltamin, directory):
    """ This function is adapted from Dcluster DCplot to assign clusters based on rho and delta values
    Input:
        dist - distance matrix
        
        XY - scaled x,y values calculated from the distance matrix
        
        ND - the number of dimensions
        
        rho - list of rho values
        
        delta - list of delta values
        
        ordrho - index for sorting the rho values in ascending order
        
        dc - density cut off
        
        nneigh - nearest neighbour for each item in matrix
        
        rhomin - rho threshold as decided using kneed function
        
        deltamin - delta threshold as decided using kneed function
        
        directory - to save the cluster value .txt file to
        
    Output:
        clusters - array containing index, cluster without halo, and cluster with halo values
        """

    import numpy as np
    import os
    import seaborn as sns
   
    #f, axarr = plot1(rho, delta)
    print('Cutoff: (min_rho, min_delta): (%.2f, %.2f)' %(rhomin,deltamin))
    NCLUST = 0
    cl = np.zeros(ND)-1
    # 1000 is the max number of clusters
    icl = np.zeros(1000)
    for i in range(ND):
        if rho[i]>rhomin and delta[i]>deltamin:
            cl[i] = int(NCLUST)
            icl[NCLUST] = int(i)
            NCLUST = NCLUST+1

    print('NUMBER OF CLUSTERS: %i'%(NCLUST))
    print('Performing assignation')
    # assignation
    for i in range(ND):
        if cl[ordrho[i]]==-1:
            cl[ordrho[i]] = cl[int(nneigh[ordrho[i]])]

    #halo
    # cluster id start from 1, not 0
    ## deep copy, not just reference
    halo = np.zeros(ND)
    halo[:] = cl

    if NCLUST>1:
        bord_rho = np.zeros(NCLUST)
        for i in range(ND-1):
            for j in range((i+1),ND):
                if cl[i]!=cl[j] and dist[i,j]<=dc:
                    rho_aver = (rho[i]+rho[j])/2
                    if rho_aver>bord_rho[int(cl[i])]:
                        bord_rho[int(cl[i])] = rho_aver
                    if rho_aver>bord_rho[int(cl[j])]:
                       bord_rho[int(cl[j])] = rho_aver
            for i in range(ND):
                if rho[i]<bord_rho[int(cl[i])]:
                    halo[i] = -1

    for i in range(NCLUST):
        nc = 0
        nh = 0
        for j in range(ND):
            if cl[j]==i:
                nc = nc+1
                if halo[j]==i:
                    nh = nh+1
            print('CLUSTER: %i CENTER: %i ELEMENTS: %i CORE: %i HALO: %i'%( i+1,icl[i]+1,nc,nh,nc-nh))
        # print , start from 1
        
        ## save CLUSTER_ASSIGNATION
        print('Generated file:CLUSTER_ASSIGNATION')
        print('column 1:element id')
        print('column 2:cluster assignation without halo control')
        print('column 3:cluster assignation with halo control')
        clusters = np.array([np.arange(ND)+1,cl+1,halo+1]).T
        np.savetxt(os.path.join(directory, 'CLUSTER_ASSIGNATION_%.2f_%.2f_.txt'%(rhomin,deltamin)),clusters,fmt='%d\t%d\t%d')
        print('Result are saved in file CLUSTER_ASSIGNATION_%.2f_%.2f_.txt'%(rhomin,deltamin))
        
        cmap = sns.color_palette('Set1', NCLUST)
        
        ax = plot2(rho,delta,cmap,cl,icl, XY, NCLUST)
        
        return (ax, clusters)

#%%    
def plot1(rho, delta):
    """ this function calculates and then plots the knee points for deciding 
    rhomin and deltamin
    Input:
        rho - list of rho (density) values
        
        delta - list of delta (distance) values
        
    Output:
        """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from kneed import KneeLocator
    
    #first calculate knee
    y = np.sort(-rho*delta)
    x = np.arange(len(rho))
    
    kneedle = KneeLocator(x,y,S=1.0, invert = False)
    
    rho_delta = -y[kneedle.knee_x]
    
    f, axarr = plt.subplots(1,2)
    axarr[0].set_title('DECISION GRAPH')
    axarr[0].scatter(rho, delta, alpha=0.6,c='black')
    axarr[0].set_xlabel(r'$\rho$')
    axarr[0].set_ylabel(r'$\delta$')
    
    axarr[1].set_title('DECISION GRAPH 2')
    axarr[1].scatter(np.arange(len(rho))+1, -np.sort(-rho*delta), alpha=0.6,c='black')
    axarr[1].plot([kneedle.knee, kneedle.knee], [0,-min(y)+10], 'r--')
    axarr[1].set_xlabel('Sorted Sample')
    axarr[1].set_ylabel(r'$\rho*\delta$')
    
    return (f, axarr, rho_delta)

#%% the next plots

def plot2(rho, delta,cmap,cl,icl,XY,NCLUST):
    import numpy as np
    import matplotlib.pyplot as plt
    
    y = -np.sort(-rho*delta)
    
    f,axarr = plt.subplots(1,2)
    axarr[0].scatter(rho, delta, alpha=0.1,c='black')
    axarr[1].scatter(np.arange(len(rho))+1, y, alpha=0.1,c='black')
    for i in range(NCLUST):
        axarr[0].scatter(rho[int(icl[i])], delta[int(icl[i])], alpha=0.6, c=cmap[i])
        axarr[1].scatter(i+1, rho[int(icl[i])]*delta[int(icl[i])], alpha=0.6,c=cmap[i])
    axarr[1].set_xlabel('Sorted Sample')
    axarr[1].set_ylabel(r'$\rho*\delta$')
    axarr[0].set_title('DECISION GRAPH')
    axarr[1].set_title('DECISION GRAPH 2')
