
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def PC_trajGraded(PCmean, PCsem, rep, saveDir, file_type, scaling, start_end, cum_var, legend):

    """ New figure style to plot the PC trajectories with a gradated colorscheme according to scaling
     Input:
        PCmean - dataframe of the PCmeans calculated after doing PCA and transforming into PC space
        
        PCsem - sem of PCs
                
        rep - if replicated specify
        
        saveDir - directory into which the figure should be saved
        
        file_type - filetype eg .png, .svg, .eps

        scaling - how to scale and grade the data

        start_end - whether to write on start and end of trajectory

        cum_var - array of cumulative variance explained by each PC

        legend - 'on' or 'off'

    Output:
    Figure

    """
    
    xscale = 1/(PCmean.max()['PC_1'] - PCmean.min()['PC_1'])
    yscale = 1/(PCmean.max()['PC_1'] - PCmean.min()['PC_2'])
    
    allDrugs= np.unique(PCmean['drug'])

    plt.figure()
    for drug in allDrugs:
        cscale = np.arange(1, np.unique(PCmean[PCmean['drug']==drug][scaling]).shape[0]+1,1)

        plt.errorbar(x= PCmean[PCmean['drug']==drug]['PC_1']*xscale,\
                     y = PCmean[PCmean['drug']==drug]['PC_2']*yscale,\
                     xerr = PCsem[PCsem['drug']==drug]['PC_1']*xscale,\
                     yerr = PCsem[PCsem['drug']==drug]['PC_2']*yscale, \
                     color = [0.8, 0.8, 0.8], zorder = -0.5, elinewidth = 2, label = None)
        plt.pause(0.1)
        plt.scatter(x = PCmean[PCmean['drug']==drug]['PC_1']*xscale,\
                    y=PCmean[PCmean['drug']==drug]['PC_2']*yscale,\
                    cmap = plt.get_cmap(drug),c=cscale , vmin = -1, label = drug, alpha =0.9)
    plt.axis('scaled')
    plt.xlim (-0.75,0.75)
    plt.ylim (-0.75,0.75)
    plt.title(scaling)
    plt.xlabel('PC_1 ' + str(cum_var[0]*100) + '%')
    plt.ylabel('PC_2 ' + str((cum_var[1] - cum_var[0])*100) + '%')
    if legend == 'on':
        plt.legend(loc='upper left', bbox_to_anchor=(1.1,1.05) ,ncol = 1, frameon= True)

    plt.tight_layout()
    plt.savefig(os.path.join(saveDir, str(rep) + 'PC12_errorbar' + file_type), dpi = 200)

    plt.show()

