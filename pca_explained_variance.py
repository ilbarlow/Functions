#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:14:33 2019

@author: ibarlow
"""

def explainVariance (pca_object, X, cut_off, feature_names, save_dir, PCs_to_plot):
    """ Function to produce plots and return data frames
    Input:
        pca_object = pca object generated from applying pca.fit_transform
        X = transformed data
        cut_off = number of PCs that explain 95% variance and so want to keep
        feature_names = ordered list of the names of the features used in the PCA
        save_dir = 
        PCs_to_plot = tuple of PCs to plot. Nb python indexing starts at 0,
        so if want PC1 and two input (0,1)
    Ouput:
        Biplot
        PC_df"""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    
    #components that explain the variance
    #make a dataframe ranking the features for each PC and also include the explained variance
        #in separate dataframe called PC_sum
    PC_feat = [] #features
    PC_sum =[] #explained variance
    for PC in range(0, cut_off):
        _sortPCs = np.flip(np.argsort(pca_object.components_[PC]**2), axis=0)
        PC_feat.append(np.array(feature_names)[_sortPCs])
        _weights = (pca_object.components_[PC]**2)/np.sum(pca_object.components_[PC]**2)
        PC_sum.append(list(_weights))
    
    #dataframe containing standards scores of contibution of each feature
        #rows correspond to PCs
    PC_vals = pd.DataFrame(data= PC_sum, columns = feature_names)
    
    #plot as biplot
    plt.figure()
    plt.arrow(0,
              0,
              PC_vals[PC_feat[PCs_to_plot[0]][0]][PCs_to_plot[0]]*100, 
              PC_vals[PC_feat[PCs_to_plot[0]][0]][PCs_to_plot[1]]*100,
              color= 'b')
    plt.arrow(0,
              0, 
              PC_vals[PC_feat[PCs_to_plot[1]][0]][PCs_to_plot[0]]*100,
              PC_vals[PC_feat[PCs_to_plot[1]][0]][PCs_to_plot[1]]*100, 
              color='r')
    plt.text(PC_vals[PC_feat[PCs_to_plot[0]][0]][PCs_to_plot[0]] + 0.7,
             PC_vals[PC_feat[PCs_to_plot[0]][0]][PCs_to_plot[1]] - 0.3,
             PC_feat[PCs_to_plot[0]][0],
             ha='center', 
             va='center')
    plt.text(PC_vals[PC_feat[PCs_to_plot[1]][0]][PCs_to_plot[0]]+0.5,
             PC_vals[PC_feat[PCs_to_plot[1]][0]][PCs_to_plot[1]]+1,
             PC_feat[PCs_to_plot[1]][0],
             ha='center',
             va='center')

    plt.xlim (-1, 1)
    plt.ylim (-1, 1)
    plt.xlabel('%' + 'PC_1 (%.2f)' % (pca_object.explained_variance_ratio_[0]*100), fontsize = 16)
    plt.ylabel('%' + 'PC_2 (%.2f)' % (pca_object.explained_variance_ratio_[1]*100), fontsize = 16)
    plt.show()
    plt.savefig(os.path.join(save_dir, 'PC{}_PC{}_Biplot.png'.format(PCs_to_plot[0], PCs_to_plot[1])))
    #add on the metadata
    PC_df = pd.DataFrame(X[:,:cut_off], columns = ['PC_{}'.format(i) for i in range (1,cut_off+1)])
    
    return PC_df, PC_sum, PC_feat

#if __name__ == '__main__':
#    sys.argv
#    