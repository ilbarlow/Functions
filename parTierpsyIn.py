#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:53:38 2018

@author: ibarlow
"""

""" Testing run times with parallel processing"""


#%%try running with multiple threads

#first just define a function to extract the files to process
def filesIn (version, toExclude):
    from tkinter import Tk, filedialog
    import os
    
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
                if toExclude in repeat: #filter out data to exclude
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
            
    return directory, fileDir
    
directoryB, fileDirB = filesIn('new', 'none')

def TierpsyParIn (directory, file, version):
    import pandas as pd
    import time
    import os
    import numpy as np
    
    #now have a dictionary of all the filenames to load
        #import features and trajectories        

    loadDir = os.path.join(directory,file)
    
    print ('loading ' + loadDir + '\r\n')
    start = time.time()
    
    trajectories = pd.DataFrame()
    features = pd.DataFrame()
    with pd.HDFStore(loadDir, 'r') as fid:
        if version.lower() == 'old':
            temp = fid['/features_summary/means']
        elif version.lower() == 'new':
            if len(fid.groups()) >=4:
                temp = fid['/features_stats'].pivot_table(columns = 'name', values = 'value')
                temp2 = fid['/timeseries_data']
                worms = np.unique(temp2['worm_index'])
                for worm in worms: #filter out the short tracks
                    if temp2[temp2['worm_index']==worm].shape[0] >= 3000:
                        trajectories= \
                        trajectories.append(temp2[temp2['worm_index']==worm])
                    else:
                        continue
            else:
                print(loadDir + 'not processed')
    
        stop = time.time()
        #print (loadDir + ': ' + str(stop-start) + ' seconds')
    
        temp['exp'] = file
        temp = temp.reset_index  (drop = True)                    
        features = features.append(temp)
        del temp, temp2, worms
    timings = (start,stop)
    
    return features, trajectories, timings

#TierpsyParIn (directoryB, file, 'new')
#so now havea list of files to run the parallel processing over

import concurrent.futures
import os
import time
import pandas as pd
import numpy as np

rep = os.path.basename(directoryB)
#threading to generate 
b =  (time.time())
with concurrent.futures.ThreadPoolExecutor(max_workers = 3) as ex:
    fut = []
    for file in fileDirB[rep]:
        fut.append(ex.submit(TierpsyParIn, directoryB, file, 'new'))
        
e = time.time()
print ('time taken : ' + str(e-b) + 'seconds')

#now loop through the future results to generate the final features and timeseries dataframes
features = pd.DataFrame()
trajectories ={}
timings = []
for f in fut:
    features = features.append(f.result()[0])
    trajectories[f.result()[0]['exp'][0]] = f.result()[1]
    timings.append(f.result()[2])
del f, fut

#and extract out the different conditions and drugs
exps = list(trajectories.keys())
conds=[]
drugs =[]
for line in exps:
    conds.append(line.split('_')[0] +'_'+ line.split('_')[-2])
    drugs.append(line.split('_')[0])

allDrugs = np.unique(drugs)

del features

#Now to do the trajectory chunking using parallel processing 
#now cut the trajectories data up into 30 min chunks, fps = 25
chunk = 90000
nChunks = int(np.ceil(np.max(trajectories[file]['timestamp'])/chunk))
chunkSize = {}
for i in range(0,nChunks+1):
    chunkSize[i] = (chunk*(i+1)) - (chunk-1)
    #chunkSize[i]+=(chunk*i)
del i


def ChunkTraj (timeseriesDF, chunkSize, conditions, name):
    """ This function can be used in parallel processing to chunk up the 
    timeseries data to create summaries for each timechunk
    
    Input: 
        timeseriesDF - dataframe containing the timeseries features extracted
        from featuresN; contains worm_index and timestamp
                
        chunkSize - dictionary containing the start and stop for each chunk
        
        conditions - list containing the unique identifier of each experiment
        
    Output:
        trajectories - a dictionary containing the timeseries data chunked 
        according to chunkSize dictionary
        """
     
    import time
    
    #make new dataframe containing the data for the half-hour windows
    trajectories = {}
    start = time.time()
    for chunkT in range(1,len(chunkSize)-1):
        trajectories[chunkT]={}
        foo = timeseriesDF['timestamp']<chunkSize[chunkT]
        bar = timeseriesDF['timestamp']>=chunkSize[chunkT-1]
        trajectories[chunkT]=timeseriesDF[foo & bar]
        del foo,bar
    stop = time.time()
    timing = (start, stop)
    conditions = '_'.join(name.split('_')[:-1])    
    
    return trajectories, timing, conditions

#threading to generate 
b =  (time.time())
with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as ex:
    fut = []
    for TJ in trajectories:
        print ('loading' + TJ + '\n')
        fut.append(ex.submit(ChunkTraj, trajectories[TJ], chunkSize, conds ,TJ))       
e = time.time()
print ('time taken : ' + str(e-b) + 'seconds')

#now extract out the chunked timeseries trajectories and write to csv in separate folders
TrajDir = os.path.join(os.path.dirname(directoryB), 'ChunkedTrajectories')
try:
    os.mkdir(TrajDir)
except FileExistsError:
    print (TrajDir, ' already exists')    
# for f in fut:
#     savedir = os.path.join(f.result()[2])
#     try:
#         os.mkdir(savedir)
#         print('Making Directory: ', savedir)
#     except FileExistsError:
#         continue
#     for chunk in f.result()[0]:
#         f.result()[0][chunk].to_csv(os.path.join(savedir, str(chunk)+ '_trajectories.csv'))
        
#alternative is to write to a new hdf5 file
for f in fut:
    savedir = os.path.join(TrajDir, f.result()[2] + '_Chunkedtrajectories.h5')
    try:
        os.mkdir(savedir)
        print('Making Directory: ', savedir)
    except FileExistsError:
        print('Folder already exits')
    
    for chunk in f.result()[0]:
        f.result()[0][chunk].to_hdf(savedir, 'Chunk_' + str(chunk),\
                 mode = 'a', format = 't')

del fut
        
   

#pyexcelerate seems like a better option for writing this notebook, or us to_csv.
