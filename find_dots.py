#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:36:05 2018

@author: ibarlow
"""

""" Code to view the dots on the plates during for MultiWorm Tracker Experiments -
based on Avelino's scripts but cut down to work smoothly with metadata files

Also the idea is to use OpenCV to open the first full frame and 15000 frame of the 
raw files, then keep back ground and determine if there is a dot on the plate
"""

import os
import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import tables

from scipy.misc import imresize

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap'] = 'gray'

directory = %pwd
directory =os.path.join(directory, 'metadata.csv')

metadata = pd.read_csv(directory)

test = metadata.loc[0][' filename']

t = cv2.hdf.open(test)
test2 = t.dsread('/mask')[10]
