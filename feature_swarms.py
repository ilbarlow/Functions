#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:59:17 2018

@author: ibarlow
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def swarms (rep1, feature, features_df, directory, file_type, cmap):
    """Makes swarm plot of features
    Input:
        rep1 - name of experiment
        
        feature - feature to be plotted
        
        features_df - dataframe of features
        
        directory - folder into which figure wil be saved
        
        file_type - image type to save (eg .tif, .svg)
    
    Output:
        swarm plot - 
    """
    plt.ioff()
    
    plt.figure()
   
    sns.swarmplot(x='drug', y=feature, data=features_df, \
                      hue = 'concentration', palette = cmap)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5) ,ncol = 1, frameon= True)    
    plt.xticks(rotation = 45)
    #plt.show()
    plt.savefig(os.path.join (os.path.dirname(directory), 'Figures', rep1 + feature + file_type),\
                 bbox_inches="tight", dpi = 200) 
