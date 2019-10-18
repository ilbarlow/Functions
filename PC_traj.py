
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import re

def PC_trajGraded(PCmean, PCsem, PCsToPlot, rep, saveDir, file_type, scaling, start_end, cum_var, legend):

    """ New figure style to plot the PC trajectories with a gradated colorscheme according to scaling
     Input:
        PCmean - dataframe of the PCmeans calculated after doing PCA and transforming into PC space
        
        PCsem - sem of PCs

        PCsToPlot = principal components to plot
                
        rep - if replicated specify
        
        saveDir - directory into which the figure should be saved
        
        file_type - filetype eg .png, .svg, .eps

        scaling - how to scale and grade the data

        start_end - whether to write on start and end of trajectory

        cum_var - array of cumulative variance explained by each PC

        legend - 'on' or 'off'

        animation - whether to animate the plot (Boolean)

    Output:
    Figure

    """
    PC1 = PCsToPlot[0]
    PC2 = PCsToPlot[1]

    xscale = 1/(PCmean.max()[PC1] - PCmean.min()[PC1])
    yscale = 1/(PCmean.max()[PC2] - PCmean.min()[PC2])
    
    allDrugs= np.unique(PCmean['drug'])

    plt.figure()
    for drug in allDrugs:
        cscale = np.arange(1, np.unique(PCmean[PCmean['drug']==drug][scaling]).shape[0]+1,1)

        plt.errorbar(x= PCmean[PCmean['drug']==drug][PC1]*xscale,\
                     y = PCmean[PCmean['drug']==drug][PC2]*yscale,\
                     xerr = PCsem[PCsem['drug']==drug][PC1]*xscale,\
                     yerr = PCsem[PCsem['drug']==drug][PC2]*yscale, \
                     color = [0.8, 0.8, 0.8], zorder = -0.5, elinewidth = 2, label = None)
        plt.pause(0.1)
        plt.scatter(x = PCmean[PCmean['drug']==drug][PC1]*xscale,\
                    y=PCmean[PCmean['drug']==drug][PC2]*yscale,\
                    cmap = plt.get_cmap(drug),c=cscale , vmin = -1,\
                     vmax = cscale[-1]+3, label = drug, alpha =0.9)
    plt.axis('scaled')
    plt.xlim (-1,1)
    plt.ylim (-1,1)
    plt.title(scaling)
    try:
        plt.xlabel(PC1 + '_' + str(cum_var[0]*100) + '%')
        plt.ylabel(PC2 + '_'+ str((cum_var[1] - cum_var[0])*100) + '%')
    except TypeError:
        print ('no cumulate variance input')
    if legend == 'on':
        plt.legend(loc='upper left', bbox_to_anchor=(1.1,1.05) ,ncol = 1, frameon= True)

    plt.tight_layout()
    plt.savefig(os.path.join(saveDir, '{}_{}_PC_errorbar{}'.format(rep, PCsToPlot, file_type)), dpi = 200)

    plt.show()

  #   if animation:
  #   	Writer = animation.writers['ffmpeg']
		# writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)


def PC_trajGradedNoScale(PCmean, PCsem, rep, saveDir, file_type, scaling, start_end, cum_var, legend):

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
    
    
    allDrugs= np.unique(PCmean['drug'])

    plt.figure()
    for drug in allDrugs:
        cscale = np.arange(1, np.unique(PCmean[PCmean['drug']==drug][scaling]).shape[0]+1,1)

        plt.errorbar(x= PCmean[PCmean['drug']==drug]['PC_1'],\
                     y = PCmean[PCmean['drug']==drug]['PC_2'],\
                     xerr = PCsem[PCsem['drug']==drug]['PC_1'],\
                     yerr = PCsem[PCsem['drug']==drug]['PC_2'], \
                     color = [0.8, 0.8, 0.8], zorder = -0.5, elinewidth = 2, label = None)
        plt.pause(0.1)
        plt.scatter(x = PCmean[PCmean['drug']==drug]['PC_1'],\
                    y=PCmean[PCmean['drug']==drug]['PC_2'],\
                    cmap = plt.get_cmap(drug),c=cscale , vmin = -1,\
                     vmax = cscale[-1]+3, label = drug, alpha =0.9)
    plt.axis('scaled')
    plt.title(scaling)
    plt.xlabel('PC_1 ' + str(cum_var[0]*100) + '%')
    plt.ylabel('PC_2 ' + str((cum_var[1] - cum_var[0])*100) + '%')
    if legend == 'on':
        plt.legend(loc='upper left', bbox_to_anchor=(1.1,1.05) ,ncol = 1, frameon= True)

    plt.tight_layout()
    plt.savefig(os.path.join(saveDir, str(rep) + 'PC12_errorbarNS' + file_type), dpi = 200)

    plt.show()




