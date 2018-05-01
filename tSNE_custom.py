#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:47:18 2018

@author: ibarlow
"""

#custom tSNE function for using with TierPsy tools


def tSNE_custom(features_df, excl_margin, to_test):
    """ Custome tSNE function for computing and iterating over tSNE perplexities 
    use sckit-learn toolbox
    Input:
        features_df - dataframe containing z-score standardised data, with NaNs
        already imputed in. Last three columns are excluded in this code as they contain the descriptors of the data
	
	excl_margin - last columns of data to be excluded as desciptors        

        to_test - the perplexities to iterate over and test
        
    Output:
        X_tsne_df - dataframe containing the t-sne values for each track, and the descriptors for each track
        
        t0 - the time to do each tSNE computation
        """
    from time import time #for timing how long code takes to rum
    from sklearn import (manifold)
    import pandas as pd
    import  numpy as np
    
    X = np.array(features_df.iloc[:,:excl_margin]) #all values except the descriptors of the data
    #y = np.array(features_df.index) #indices for the data
    n_samples, n_features = X.shape
    
    #now to compute t-SNE
    X_tsne = {}
    X_tsne_df = {}
    t0 = {}
    if isinstance(to_test, int):
        t0[to_test] = time()
        tsne = manifold.TSNE(n_components =2, perplexity = to_test, \
              init = 'pca', random_state = 0)
        X_tsne [to_test] = tsne.fit_transform(X)
        t0[to_test] = time() - t0[to_test]
        
        X_tsne_df[to_test] = pd.DataFrame (X_tsne[to_test], columns = ['SNE_1', 'SNE_2'])
        X_tsne_df[to_test] = pd.concat([X_tsne_df[to_test], features_df.iloc[:,excl_margin:]], axis = 1)
    else:
        for i in to_test:
            t0 [i] = time()
            print ('Computing t-SNE')
            tsne = manifold.TSNE(n_components = 2, perplexity = i, \
                                 init = 'pca', random_state = 0)
            X_tsne [i] = tsne.fit_transform(X)
            t0 [i] = time() - t0[i]
        
            print (str(i) + ': ' + str(t0[i]) + 'secs')
        
        #convert to dataframe for easy plotting
            X_tsne_df[i] = pd.DataFrame (X_tsne[i])
            X_tsne_df[i].columns = ['SNE_1', 'SNE_2']
            X_tsne_df[i] = pd.concat([X_tsne_df[i], features_df.iloc[:,excl_margin:]], axis = 1)
    
    return X_tsne_df, t0

#%%
    
def pre_plot(plotting):
    """ this plot actually juse makes tSNE scatter plots
    Input:
        plotting - dataframe containing the SNE values to plot and the drugs
    Output:
        tSNE scatter plot
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.lmplot(x = 'SNE_1', y= 'SNE_2', data= plotting, hue = 'drug', fit_reg = False, legend = False)
    plt.axis('equal')
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1) ,ncol = 1, frameon= True)
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.show()


def sne_plot(tSNE_df, to_plot, conc, uniqueConcs1):
    """Function to plot tSNE
    Input:
        t_SNE_df - dataframe containing the two SNE dimensions and conditions - doses, drugs and dates
        
        to_plot - the perplexities to plot - number or array
        
        conc - concentration to plot
        
        uniqueConcs - the unique concentrations in the experiment
    Output:
        tSNE_plot
        """
    import seaborn as sns
    
    sns.set_style('whitegrid')
    if isinstance(to_plot, list):
        if conc == []:
            for i in to_plot:
                plotting = tSNE_df[i][tSNE_df[i]['concentration'].isin(uniqueConcs1)]
                pre_plot(plotting)
        else:
            for i in to_plot:
                plotting = tSNE_df[i][tSNE_df[i]['concentration']==float(conc)]
                plotting = plotting.append(tSNE_df[i][tSNE_df[i]['drug']=='DMSO'])
                plotting = plotting.append(tSNE_df[i][tSNE_df[i]['drug'] == 'No_compound'])
                pre_plot(plotting)
                del plotting
    else:
        i = to_plot
        if conc == []:
            plotting = tSNE_df[i][tSNE_df[i]['concentration'].isin(uniqueConcs1)]
            pre_plot(plotting)
        else:
            plotting = tSNE_df[i][tSNE_df[i]['concentration']==float(conc)]
            plotting = plotting.append(tSNE_df[i][tSNE_df[i]['drug']=='DMSO'])
            plotting = plotting.append(tSNE_df[i][tSNE_df[i]['drug'] == 'No_compound'])
            pre_plot(plotting)
            
