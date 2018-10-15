#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 18:10:10 2018

@author: em812
"""

import numpy as np

def get_colormaps(RGBsteps=7,thirdColorSteps=3, n_bins=20):
    """
    Function that creates colormaps with specific colors that go gradually from very light to very dark
    param: 
        RGBsteps = number of R, G, B gradients that are combined by pairs (RG,RB,GB) to form the colors
        thirdColorSteps = number of gradients of the third color added (to increase the variety of colors produced) 
    return: 
        cm = a list of colormap objects with single colors going from light to dark
            the size of cm is (RGBsteps**2)*thirdColorSteps
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    n = RGBsteps
    nad = thirdColorSteps
    #n_bins = 20 # steps in the colormap from light to dark
    ncolor=0
    
    colors=[]
    for ad in np.linspace(0,1,num=nad):
        for i,j in zip(np.linspace(1,0,num=n),np.linspace(0,1,num=n)):
            colors.append([(1, 1, 1), (i, j, ad), (0, 0, 0)])
            ncolor += 1
    
    for ad in np.linspace(0,1,num=nad):
        for i,j in zip(np.linspace(1,0,num=n),np.linspace(0,1,num=n)):
            colors.append([(1, 1, 1), (i, ad, j), (0, 0, 0)])
            ncolor += 1
            
    for ad in np.linspace(0,1,num=nad):
        for i,j in zip(np.linspace(1,0,num=n),np.linspace(0,1,num=n)):
            colors.append([(1, 1, 1), (ad, i, j), (0, 0, 0)])
            ncolor += 1
            
    cm=[]
    for ic,color in enumerate(colors):
        cmap_name = 'mymap_{}'.format(ic)
        # Create the colormap
        cm.append(LinearSegmentedColormap.from_list(
            cmap_name, color, N=n_bins))
            
    return cm
    
def plot_color_gradients(cmap_list, drug_names):
    """
    Function that plots the colormaps as colorbar legends, with the drug_name label on the side.
    You can plot it as a supplement of the main figure (because you cannot plot multiple colorbars automatically)
    cmap_list = list of colormap objects that you want to plot
    drug_names = corresponding labels (list of strings)
    nrows = number of 
    """
    import matplotlib.pyplot as plt

    nrows = len(drug_names)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    fig, axes = plt.subplots(nrows=nrows,figsize=(10, 10))
    #fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    
    for ax, name, drName in zip(axes, cmap_list, drug_names):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, drName, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()
