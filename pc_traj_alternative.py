#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:08:07 2018

@author: ibarlow
"""

from matplotlib.colors import makeMappingArray

def PC_traj(dfMEAN, dfSEM,PCs_toplot, rep, directory, file_type, cmap,  drugsToPlot, start_end):
    """this function groups by drug an plots the trajectories through PC space
    Input
        dfMEAN - dataframe containing the PC values for each of the drugs
        dfSEM - dataframe containing the PC SEM for each drug at each dose
        PCs_toplot 
        rep - the name of the experiments
        directory - the directory to save the files into
        file_type - type of image ('tif' or 'svg' ...)
        cmap - colormap to use
        drugstoPlot
        start_end
    Output
        Plot showing trajectory through PC space with errorbars
        
    """ 
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    
    #scale the PCs
    xscale = 1/(np.max(dfMEAN[PCs_toplot[0]]) - np.min(dfMEAN[PCs_toplot[0]]))
    yscale = 1/(np.max(dfMEAN[PCs_toplot[1]]) - np.min(dfMEAN[PCs_toplot[1]]))
    
    #okay so now have a summary of each drug for each PC.
        #scale and plot the drugs across the PC1 and 2 space
    
    #make note of drugs to plot
    if drugsToPlot ==[]:
        uniqueDrugs1 = np.unique(dfMEAN['drug'])
    else:
        uniqueDrugs1 = drugsToPlot

    #to make this work need to set the colormap and then use the c to set a scale and change with cscale
    
    cscale = np.arange(0,plottingMEAN[PCs_toplot[0]].shape[0],1)    
    fig= plt.figure()
    #for each drug plot the mean and SEM in both PC1 and PC2
    for drug in range(len(uniqueDrugs1)):
        MeanPlot = dfMEAN['drug'] == uniqueDrugs1[drug]
        SemPlot = dfSEM['drug'] == uniqueDrugs1[drug]
        plottingMEAN = dfMEAN[MeanPlot]
        plottingSEM = dfSEM[SemPlot]
        plt.errorbar(x=plottingMEAN[PCs_toplot[0]]*xscale, y=plottingMEAN[PCs_toplot[1]]*yscale, \
                     xerr = plottingSEM[PCs_toplot[0]]*xscale, yerr=plottingSEM[PCs_toplot[1]]*yscale, \
                      zorder = -1,linewidth =0.5, linestyle = '--', marker = 'o',color = [0.9, 0.9, 0.9], label = None)
    for drug in range(len(uniqueDrugs1)):
        MeanPlot = dfMEAN['drug'] == uniqueDrugs1[drug]
        plottingMEAN = dfMEAN[MeanPlot]
        #plt.set_cmap(uniqueDrugs1[drug])
        plt.scatter(x=plottingMEAN[PCs_toplot[0]]*xscale, y=plottingMEAN[PCs_toplot[1]]*yscale, \
                      zorder =1,cmap = plt.get_cmap(uniqueDrugs1[drug]), c = cscale, marker = 'o', label = uniqueDrugs1[drug])

        if start_end == True:
            plt.text(x=plottingMEAN[PCs_toplot[0]].iloc[0]*xscale, y=plottingMEAN[PCs_toplot[1]].iloc[0]*yscale, s='start')
            plt.text(x=plottingMEAN[PCs_toplot[0]].iloc[-1]*xscale, y= plottingMEAN[PCs_toplot[1]].iloc[-1]*yscale, s='end')
        else:
            continue
    plt.axis('scaled')
    plt.xlim (-1,1)
    plt.ylim(-1,1)
    plt.legend(loc='upper left', bbox_to_anchor=(1.1,1.05) ,ncol = 1, frameon= True)
    plt.tight_layout(rect=[0,0,1,1])
    plt.xlabel (PCs_toplot[0])
    plt.ylabel(PCs_toplot[1])
    plt.show()
    try:
        plt.savefig(os.path.join(os.path.dirname(directory), 'Figures', rep + '_PCtraj.' + file_type),\
                bbox_inches="tight")
    except TypeError:
        plt.savefig(os.path.join(os.path.dirname(directory), 'Figures', 'PC_Traj.' + file_type), bbox_inches='tight')
    plt.show()


