#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:42:52 2018

@author: ibarlow
"""

def TierPsyInput(version, exclude):
    """ A filedialog will open to select the results folder. The script will 
    then search through this folder for the features files
    
    Input:
        version - 'old' or 'new'
                This determines which features are imported
        
        exclude - the name of any folders to exclude
        
    Output:
        directory - pathname of the selected results folder
        
        fileDirA - list of files within each directory
        
        features - a dictionary containing the features timeseries (old) or
            summaries (new) for each results folder
            
        trajectories - filtered so that only worms with trajectories > 3000 frames are kept"""
    
    from tkinter import Tk, filedialog
    import pandas as pd
    import os
    import numpy as np
    import time
    
    #test version
    if version.lower() == 'new':
        feat_file = '_featuresN.hdf5'
    elif version.lower() == 'old':
        feat_file = '_features.hdf5'
    else:
        print ('Version not specified!')

    #popup window
    print ('Select Data Folder')
    root = Tk()
    root.withdraw()
    root.lift()
    root.update()
    root.directory = filedialog.askdirectory(title = "Select Results folder", \
                                             parent = root)
    
    if root.directory == []:
        print ('No folder selected')
    else:
        directory = root.directory
        #find the folders within
        reps = os.listdir(directory)
        print(reps)
           
    #now find within each subfolder all the feature files
    fileDir ={}
    try:
        for repeat in reps:
            if repeat != '.DS_Store': #ignore these hidden files  
                if exclude in repeat: #filter out data to exclude
                    continue
                else:
                    temp = os.listdir(os.path.join(directory, repeat))
                    fileDir[repeat] = []
                    for line in temp:
                        if line.endswith(feat_file) == True:
                          fileDir[repeat].append(line)
                        else:
                            continue
        
    except NotADirectoryError:
        print('No subfolders')
        rep = os.path.basename(directory)
        fileDir[rep] = []
        temp = os.listdir(os.path.join(directory))
        for line in temp:
            if line.endswith(feat_file) == True:
                fileDir[rep].append(line)
            else:
                continue

    #now have a dictionary of all the filenames to load
        #import features and trajectories        
    features ={}
    trajectories = {}
    for rep in fileDir:
        features[rep] = pd.DataFrame()
        trajectories[rep] = {}
        for line in fileDir[rep]:
            if len(fileDir)>1:
                loadDir = os.path.join(directory, rep, line)
            else:
                loadDir = os.path.join(directory,line)
            
            print (loadDir)
            start = time.time()
            trajectories[rep][line] = pd.DataFrame()
            with pd.HDFStore(loadDir, 'r') as fid:
                if version.lower() == 'old':
                    temp = fid['/features_summary/means']
                elif version.lower() == 'new':
                    if len(fid.groups()) <4:
                        continue
                    else:
                        temp = fid['/features_stats'].pivot_table(columns = 'name', values = 'value')
                        temp2 = fid['/timeseries_data']
                        worms = np.unique(temp2['worm_index'])
                        for worm in worms: #filter out the short tracks
                            if temp2[temp2['worm_index']==worm].shape[0] >= 3000:
                                trajectories[rep][line] = \
                                trajectories[rep][line].append(temp2[temp2['worm_index']==worm])
                            else:
                                continue
            
                stop = time.time()
                print (loadDir + ': ' + str(start-stop))
            
                temp['exp'] = line
                temp = temp.reset_index  (drop = True)                    
                features[rep] = features[rep].append(temp)
                del temp, temp2, worms
            features[rep] = features[rep].reset_index(drop=True)
    
    return directory, fileDir, features, trajectories

#%%

def extractVars(exp_names):
    """ Extracts out the date, drugs, concentrations and uniqueIDs from the experiments
    Input - exp_name - experiment names
    
    Output:
        date - date of recording
        
        drugs- list of drugs tested for each experiments
        
        concs - list of concentrations tested
        
        uniqueID - unique ID for plate
        """
    import pandas as pd
    
    drug = []
    conc = []
    date = []
    uniqueID =[]
    for line in exp_names: #split the line and find the drug, concentration and date
        if line.split('_')[2] == 'No':
            drug.append('No_compound')
            conc.append(float(line.split('_')[4]))
            date.append(str(line.split('_') [8]))
            uniqueID.append(int(line.split('_')[-2]))
        else:
            drug.append(line.split('_')[2])
            conc.append(float(line.split('_')[3]))
            date.append(str(line.split('_') [7]))
            uniqueID.append(int(line.split('_')[-2]))
    
    drug = pd.DataFrame(drug)
    drug.columns = ['drug']
    conc = pd.DataFrame(conc)
    conc.columns = ['concentration']
    date = pd.DataFrame(date)
    date.columns = ['date']
    uniqueID = pd.DataFrame(uniqueID)
    uniqueID.columns = ['uniqueID']
    return drug, conc, date, uniqueID

#%%
def z_score(features):
    """ function for z_score normalisation
    Input:
        features - features dataframe with experiment column removed
        
    Output:
        featZ - Z-normalised features
    """
    import pandas as pd
    import numpy as np
    
    featZ = pd.DataFrame()
    for column in range(0, features.shape[1]): #every feature
        temp = (features.iloc[:,column] - \
                (np.nanmean(features.iloc[:,column])))/ \
                (np.nanstd(features.iloc[:,column]))
        featZ[temp.name] = temp.values
        del temp
    featZ = featZ.reset_index(drop=True)
    return featZ    

#%%

def FeatFilter(features):
    """ this function removes features with too many NaNs
    Input:
        features - dataframe of features
    
    Output:
        to_exclude - list of features with >50% NaNs"""
    import numpy as np
    import pandas as pd
    
    to_exclude  = []
    n_worms = features.shape[0]
    for feat in features.columns:
        if np.sum(np.isnan(pd.to_numeric(features[feat])))> 0.5*n_worms:
            to_exclude.append(feat)
        else:
           continue
    return to_exclude

#remove these features from the features dataframes
def FeatRemove(features, exlList):
    features.drop(exlList, axis=1, inplace=True)
    return features
