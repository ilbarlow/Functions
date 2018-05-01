#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:53:28 2018

@author: ibarlow

PCA functions

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os


#function defines the pca - can actually put this earlier in the script
def pca(X_std, rep, directory, file_type):
    """pca function that returns PCA scree plots and ...
    Input:
        X_std - standardly scaled raw features data
        
        rep - the name of the experiment (as in the replicate)
        
        directory - the directory for saving files
        
        file_type - type of file to save the screen plots
        
    Output:
        eig_vecs - eigen vectors (ie planes) for each of the principle components (type = ?)
        
        eig_vals - eigen values are the scaling factors for each eigenvector (type = ). Used to calculate the amount of variance explained
        
        eig_pairs - tuple containg the PC eigenvalue and array of eigenvectors for that PC - the contribution of each features tot that plane
        
        PC_pairs - tuple containing PC number, variance explained, and cumulative variance explained
        
        PC_df - dataframe of PC_pairs
        
        cut_off - integer of the number of PCs that explain 95% of the cumulative variance
        
        PCA scree plots as images
        """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    eig_vecs, eig_vals, v, = np.linalg.svd(X_std.T)
    #test the eig_vecs
    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print ('Everything OK!')
    
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    #then sort - high to low
    eig_pairs.sort(key =lambda tup:tup[0])
    eig_pairs.reverse()

    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])
    
    #make plots
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp= np.cumsum(var_exp)
    #add in cut off for where 95% variance explained
    cut_off = cum_var_exp <95
    cut_off = np.argwhere(cut_off)
    cut_off = int(cut_off[-1])

    #first make dataframe with all the PCs in
    x=['PC %s' %i for i in range(1,len(eig_vals))]
    y= var_exp[0:len(eig_vals)]
    z=cum_var_exp[0:len(eig_vals)]
    PC_pairs= [(x[i], y[i], z[i]) for i in range(0,len(eig_vals)-1)]
    PC_df = pd.DataFrame(data=PC_pairs, columns = ['PC', 'variance_explained', \
                                                'cum_variance_explained'])
    
    #make a a figure
    sns.set_style ('whitegrid')
    f, (ax1,ax2) = plt.subplots(ncols=2, sharey=True)
    plt.title(rep)
    trace1 = sns.barplot(y= 'variance_explained', x= 'PC', data=PC_df, ax=ax1)
    sns.despine()
    ax1.xaxis.set_ticks(np.arange(0,70,10))
    ax1.xaxis.set_ticklabels(PC_df['PC'][0:71:10])
    ax1.axes.tick_params(axis = 'x', rotation = 45, direction = 'in', labelbottom = True)
    ax1.xaxis.axes.set_xlim(left = 0, right= 70)
    trace2 = sns.stripplot(y='cum_variance_explained', x='PC', data=PC_df, ax=ax2)
    ax2.xaxis.set_ticks(np.arange(0,70,10))
    ax2.xaxis.set_ticklabels(PC_df['PC'][0:71:10])
    ax2.axes.tick_params(axis = 'x', rotation = 45, direction = 'in', labelbottom = True)
    ax2.xaxis.axes.set_xlim(left = 0, right= 70)
    trace2 = plt.plot([cut_off, cut_off], [0,95], linewidth =2)
    plt.text(cut_off, 100, str(cut_off))
    
    sns.despine()
    f.savefig(os.path.join(directory[0:-7], 'Figures', rep + '_PC_variance_explained.' + file_type), dpi=400)
    plt.show()
    del x,y, z, tot, var_exp, cum_var_exp, f, ax1, ax2, trace1, trace2
    
    return eig_vecs, eig_vals, eig_pairs, PC_pairs, PC_df, cut_off

#%%
#now to find the top features that contribute to PC1 and PC2
def PC_feats(eig_pairs, cut_offs, features):
    """ finds the top features and returns dataframes with contributions and 
    features
    
    Input:
        eig_pairs - eigenvalue-vector tuple
        
        cut_offs - the number of PCs that contribute 95% of the variance
        
        features - features dataframe containing all the feature names
        
    Output:
        PC_contribs - list of arrays of contribution of each feature for each PC in range of cut_offs
        
        PC_features - Dataframe of PC_contribs with feature names added
        
        PC_tops - Rank list of top features contributing to each PC
        
        x - list of names of PCs
        
        """
    x = ['PC_%s' %i for i in range(1,cut_offs+1)]
    PC_contribs = [(eig_pairs[i][1]) for i in range (0,cut_offs)]
    features_1 = list(features.columns)
    PC_features = pd.DataFrame(PC_contribs)
    PC_features = PC_features.T
    PC_features.columns = x
    PC_features['features'] = features_1
    
    #rank the features
    PC_tops = {}
    for PC in PC_features.columns:
        PC_tops[PC] = list(PC_features[PC].sort_values().index[:])
        PC_tops[PC].reverse()
        PC_tops[PC] = PC_features['features'].iloc[PC_tops[PC]]
    return PC_contribs, PC_features, PC_tops, x

#%%
#biplot function
def biplot(ranks, coeff, pc1, pc2, n_feats, directory, rep, file_type, uniqueDrugs):
    """ biplot function  - specify output file type"""
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    cmap = sns.color_palette("husl", len(uniqueDrugs))
    sns.set_style('whitegrid')
    pcs = ('PC_%d' %(pc1), 'PC_%d' %(pc2))
    plt.figure()
    for pc in range(len(pcs)):
        if pc == 1:
            for i in range (n_feats):
                plt.arrow(0,0,\
                          coeff[np.flip(pcs,axis=0)[pc]].iloc[ranks[pcs[pc]].index[i]],  \
                          coeff[pcs[pc]].iloc[ranks[pcs[pc]].index[i]],\
                          color = cmap[pc], alpha = 1, label = pcs[pc])    
                if coeff is None:
                    continue
                else:
                    plt.text (coeff[np.flip(pcs,axis =0)[pc]].iloc[ranks[pcs[pc]].index[i]]*3, \
                                    coeff[pcs[pc]].iloc[ranks[pcs[pc]].index[i]]*1.5, \
                              coeff['features'].iloc[ranks[pcs[pc]].index[i]], color = cmap[pc],\
                              ha = 'center', va='center')
        else:
            for i in range (n_feats):
                plt.arrow(0,0, coeff[pcs[pc]].iloc[ranks[pcs[pc]].index[i]],\
                          coeff[np.flip(pcs,axis=0)[pc]].iloc[ranks[pcs[pc]].index[i]],\
                          color = cmap[pc], alpha = 1, label = pcs[pc])    
                if coeff is None:
                    continue
                else:
                    plt.text (coeff[pcs[pc]].iloc[ranks[pcs[pc]].index[i]]*4, \
                              coeff[np.flip(pcs,axis =0)[pc]].iloc[ranks[pcs[pc]].index[i]]*3,\
                              coeff['features'].iloc[ranks[pcs[pc]].index[i]], color = cmap[pc],\
                              ha = 'center', va='center')
    plt.xlim (-1, 1)
    plt.ylim (-1,1)
    #plt.axis('equal')
    plt.xlabel ('PC_1')
    plt.ylabel('PC_2')
    plt.legend()
    plt.savefig(os.path.join(directory[0:-7], 'Figures', rep + '_biplot.' + file_type), dpi =200)
    plt.show()

#%%
def feature_space(features, eig_pairs, X_std, cut_offs, x, drug, conc, date):
    """ transforms features data into the PC space
    Input:
        features - features dataframe after filtering
        
        eig_pairs - eig value - vector tuples
        
        X_std - standard scaled data
        
        cut_offs - number of PCs that explain 95% variance
        
        x - array of PCnames
        
        drug - list containing corresponding drugs for each row of features dataframe
        
        cocn - list containing corresponding concentrtaion for each row of features dataframe
        
        date - list of dates for corresponding row of dataframe
        
    Output:
        matrix_w - matrix of features transformed into PC space
        
        Y - 
        
        PC_df - dataframe containing all the PCs for each condition
        
        """
    import numpy as np
    import pandas as pd
    
    matrix_w = eig_pairs[0][1].reshape(eig_pairs[0][1].size,1)
    for i in range(1,cut_offs):
        temp_matrix = eig_pairs[i][1].reshape(eig_pairs[i][1].size,1)
        matrix_w = np.hstack((matrix_w, temp_matrix))
        del temp_matrix
    print ('Matrix W: \n', matrix_w)
    
    Y = X_std.dot(matrix_w)
    PC_df = pd.DataFrame(Y)
    PC_df.columns = x
    PC_df['drug'] = drug
    PC_df['concentration'] = conc
    PC_df['experiment'] = date
    return matrix_w, Y, PC_df

#%%
#to make plots    
def PC12_plots (df, dose, rep, directory, file_type):
    """this makes plots that are scaled PCs
    Input:
        df - dataframe containing PCs for each condition
        
        dose - dose to be plotted
        
        rep - experiment name
        
        directory - directory into which the plot will be saved
        
        file_type - tif or svg
    
    Output:
        plots of each of the conditions along PCs 1 and 2
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_style('whitegrid')
    if dose == []:
        temp = df
    else:
        to_plot = (df['concentration'] == float(dose))# or (df['concentration'] == float(14))
        temp = df[to_plot]
        temp = temp.append(df[df['drug']=='DMSO']) #add on DMSO controls
        temp = temp.append (df[df['drug'] == 'No_compound'])
    xs = temp['PC_1']
    ys = temp['PC_2']
    scalex = 1/(xs.max() - xs.min())
    scaley = 1/(ys.max() - ys.min())
    temp ['PC_1'] = temp['PC_1'].replace(temp['PC_1'].values, xs*scalex)
    temp['PC_2'] = temp['PC_2'].replace(temp['PC_2'].values, ys*scaley)
    f = plt.figure
    f= sns.lmplot(x= 'PC_1', y='PC_2', data= temp, hue = 'drug',fit_reg = False)
   
    plt.xlim (-1, 1)
    plt.ylim (-1,1)
    plt.title ('concentration = ' + str(dose))
    plt.savefig (os.path.join(directory[0:-7], 'Figures', rep + '_'\
                              + str(dose) + '_PC12_norm.' + file_type), dpi = 200)

#%%   
#now can make dataframe containing means and column names to plot trajectories through PC space
def PC_av(PC_dataframe, x):
    """function to convert to average PC for replicates. Requires PC dataframe
    and x containing all the column name
    Input:
        PC_dataframe - average value for each condition
        
        x - name of PCs
        
    Output:
        PC_means - average PC dataframe
    
    """
    import numpy as np
    import pandas as pd
    
    PC_means= pd.DataFrame(data = None, columns = x)
    uniqueDrugs1 = np.unique(PC_dataframe['drug'])
    for drug in uniqueDrugs1:
        finders = PC_dataframe['drug'] == drug
        keepers = PC_dataframe[finders]
        concs = np.unique(keepers['concentration'])
        for dose in concs:
            refine = keepers['concentration'] == dose
            final = keepers[refine]
            temp = final.iloc[:,0:-2].mean(axis=0)
            temp = temp.to_frame().transpose()
            temp['drug'] = drug
            temp['concentration'] = dose
            PC_means= PC_means.append(temp)
            del refine, final, temp
        del finders, keepers, concs

    PC_means = PC_means.reset_index(drop=True)
    return PC_means

#%%
def PC_traj(df,rep, directory, file_type, cmap):
    """this function groups by drug an plots the trajectories through PC space
    Input
        df - dataframe containing the PC values for each of the drugs
        rep - the name of the experiments
        directory - the directory to save the files into
        file_type - type of image ('tif' or 'svg' ...)
        cmap - colormap to use
        
    Output
        Plot showing trajectory through PC space
    """ 
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    #scale the PCs
    xscale = 1/(np.max(df['PC_1']) - np.min(df['PC_1']))
    yscale = 1/(np.max(df['PC_2']) - np.min(df['PC_2']))
    
    #okay so now have a summary of each drug for each PC.
        #scale and plot the drugs across the PC1 and 2 space
    uniqueDrugs1 = np.unique(df['drug'])
    #cmap = sns.choose_colorbrewer_palette(data_type = 'q')
    
    #make a figure of the data
    plt.figure()
    #cmap = sns.color_palette("husl", len(uniqueDrugs1)) #set colormap
    for drug in range(len(uniqueDrugs1)):
        to_plot = df['drug'] == uniqueDrugs1[drug]
        plotting1 = df[to_plot]
        ax = plt.plot(plotting1['PC_1']*xscale, plotting1['PC_2']*yscale, \
                      linewidth =2, color = cmap[drug], marker = 'o', \
                      label = uniqueDrugs1[drug])
    plt.axis('scaled')
    plt.xlim (-1,1)
    plt.ylim(-1,1)
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1) ,ncol = 1, frameon= True)
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.xlabel ('PC_1')
    plt.ylabel('PC_2')
    plt.savefig(os.path.join(directory[0:-7], 'Figures', rep + '_PCtraj.' + file_type),\
                bbox_inches="tight", dpi = 200)
    plt.show()