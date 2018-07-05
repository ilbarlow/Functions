#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 16:59:07 2018

@author: ibarlow
"""

"""
Created on Mon Jun 25 14:22:45 2018

@author: ibarlow
"""

""" Analysis of long-term drug exposure"""

import TierPsyInput as TP
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import the features and trajectories
directoryA, fileDirA, featuresA, trajectoriesA = TP.TierPsyInput('new', 'none')

#list the files in
reps = list(trajectoriesA.keys())
exps = list(trajectoriesA[reps[0]].keys())
conds=[]
drugs =[]
for line in exps:
    conds.append(line.split('_')[0] +'_'+ line.split('_')[-2])
    drugs.append(line.split('_')[0])

allDrugs = np.unique(drugs)

del featuresA

#now cut the trajectories data up into 30 min chunks, fps = 25
chunk = 45000
nChunks = 10
chunkSize = {}
for i in range(0,nChunks+1):
    chunkSize[i] = (chunk*(i+1)) - (chunk-1)
    #chunkSize[i]+=(chunk*i)
del i
    
#make new dataframe containing the data for the half-hour windows
delT = 5*25 #5 second sliding window
trajectories2 = {}
for rep in trajectoriesA:
    for chunkT in range(1,len(chunkSize)-1):
        trajectories2[chunkT]={}
        for i in range(0,len(conds)):
            foo = trajectoriesA[rep][exps[i]]['timestamp']<chunkSize[chunkT]
            bar = trajectoriesA[rep][exps[i]]['timestamp']>=chunkSize[chunkT-1]
            trajectories2[chunkT][conds[i]]=trajectoriesA[rep][exps[i]][foo & bar]
            trajectories2[chunkT][conds[i]]['ttBin']=np.ceil(trajectoriesA[rep][exps[i]]['timestamp']/delT) #create binning window
            del foo,bar

del trajectoriesA

#first need to average the tracks, z-score and then standardise
    #for each time chunk
        #take mean for every track on each plate, and then median, std, 10th and 90th quartile for plate
features = {}
featMatMean = pd.DataFrame()
for chunk in trajectories2:
    features[chunk]=pd.DataFrame()
    for rep in trajectories2[chunk]:
        worms = np.unique(trajectories2[chunk][rep]['worm_index'])
        for worm in worms:
            temp = trajectories2[chunk][rep][trajectories2[chunk][rep]['worm_index']==worm].mean().to_frame().transpose()
            temp['rep'] = rep
            temp['chunk'] = chunk
            temp['drug'] = list(temp['rep'])[0].split('_')[0]
            features[chunk] = features[chunk].append(temp)
            del temp
        #fill in the nans for each drug
    for drug in allDrugs:
        temp2 = features[chunk][features[chunk]['drug'] == drug]
        temp2 = temp2.fillna(temp2.mean(axis=0))
        featMatMean = featMatMean.append(temp2) #put into one big dataframe
        del temp2
    
    features[chunk] = features[chunk].reset_index(drop=True)
featMatMean = featMatMean.reset_index(drop=True)

#Save the chunked feature dataframe to a .csv file so that can be combined with the other experiments
writer = pd.ExcelWriter(os.path.join(os.path.dirname(directoryA), 'LongExposureFeatures.xlsx'))
for chunk in features.keys():
    features[chunk].to_excel(writer, sheet_name = str(chunk))
writer.save()   

writer.close()

#Save the chunked feature dataframe to a .csv file so that can be combined with the other experiments
writer = pd.ExcelWriter(os.path.join(os.path.dirname(directoryA), 'LongExposureFeatMatMean.xlsx'))
featMatMean.to_excel(writer, sheet_name = 'FeatMatMean')
writer.save()  