#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:32:51 2018

@author: ibarlow
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:24:02 2018

@author: ibarlow
"""
import os
import pandas as pd
import numpy as np


#use os.walk to get all the files in the folder for the experiment
def getFiles(FolderInput, validExt):
    """ This Function gets all the relevant files from the selected folder
    Input- 
    FolderInput: a directory to the Folder containig the raw files
    
    OutPut -
    existingFiles: the files to be processed """
    
    existingFiles = []
    for (root, dirs, files) in os.walk(FolderInput):
        if files == []: #pass empty folders
            continue
        elif files[0].endswith('.ini'): #don't add the worm encoder file
            existingFiles.append(os.path.join(root, files[0]))
        else:
            for file1 in files: #loop through to add the files
                if file1.endswith(validExt):
                    existingFiles.append(os.path.join(root, file1))
                else:
                    continue

    return existingFiles
 
#now loop through the filenames to replace with the correct channel and find all the attributes
def FilenameDF (existingFiles, ScreenPrintOut, deltatime):
    """ this function creates a dataframe of the data with the channel corrected
    according to the acquisition PC, and then corrected for by time of acquisition 
    to account for when PCs are not started in the correct order.
    The metadata from the ScreeningExcelPrintout is then added
    
    Input-
        existingFiles: list of the file directories to rename and compile with metadata
    
        ScreenPrintOut: directory of .xls printout used for setting up the experiment
        that contains the metadata
    
        deltatime: max time to allow between each channel (eg 5 (mins))
    
    Output-
        SortedRename - dataframe containing files names sorted by PC and channel
        
        unprocessed - list of files that have not had any metadata attached
    """
    
    geckoTimes =[] #timestamp from gecko
    channels = [] #channels
    PCs = [] #PCnumber
    dates = [] #date of experimemt
    setN = [] #set number from gecko file
    unprocessed = []
    for exp in existingFiles: #this loop goes through all the input files and extracts the metadata
        try:
            root, file = os.path.split(exp) #split the pathname and root
            metaData = file.split('_')
            if len(metaData) >6:
                ss,s1,s2,s3,s4,channel,dd,gecko = metaData
            else:
                ss, s1, s2, channel, dd, gecko = metaData
                
                
            #ss, s1, s2, channel, dd, gecko = file.split('_') #split up file to constituent\
            #timestamp  = gecko, which can be used to sort the files
            
            geckoTimes.append(pd.to_datetime(gecko.split('.')[0], format = '%H%M%S'))
            
            dates.append(pd.to_datetime(dd, format = '%d%m%Y'))
            
            setN.append(int(ss[-1]))
                
            try:    
                comp = int(root.split(os.sep)[-2][-1])
                PCs.append(comp)
                #now correct the channel
                if channel == 'Ch1':
                    channels.append((2*comp)-1)
                else:
                    channels.append(2*comp)   
        
            except TypeError:
                print (exp + ' NoPC')
            
            del root, file, ss, channel, dd, gecko, comp
        
        except ValueError:
            print ('Did not get processes' + exp)
            unprocessed.append(exp)
            geckoTimes.append('nan')
            channels.append('nan')
            dates.append('nan')
            setN.append('nan')
            PCs.append('nan')

    #put in to a dataframe that will then can add in the drug and dose details
    data_tuples = list(zip(existingFiles, geckoTimes, dates, channels, PCs, \
                                    setN))
    renameDF = pd.DataFrame(data = data_tuples, columns= ['Old', 'time', 'date', 'channel',\
                                        'PC', 'Set'])
    #sort by time and then channel
    renameDF = renameDF.sort_values(['date', 'time'], ascending = [True, True])
    renameDF = renameDF.reset_index(drop=True)

    #number of days the experiment occurred over
    nDays = np.unique(renameDF['date'])
    
    dt = pd.to_timedelta(deltatime, unit = 'm') #tolerated delta time
    
    SortedRename = pd.DataFrame(index=[1]) #initialise dataframe with one row - required for loop
    
    #now sort through to rearrange the rig positions - sort by metadate record time
    for day in nDays:
        try:
            if np.isnat(day):
                DayAll = renameDF[renameDF['date'].isnull()]
                SortedRename = SortedRename.append(DayAll, ignore_index = True) #append in and use index
            else:    
                DayAll = renameDF[renameDF['date']==day]
                for row in DayAll.iterrows():
                    sortSet = DayAll[abs(DayAll['time'] - row[1]['time'])<dt]
                    sortSet = sortSet.sort_values(['channel'])
                    if np.max(sortSet.index.tolist()) != np.max(SortedRename.index.tolist()):
                        SortedRename = SortedRename.append(sortSet, ignore_index = False)
                    else:
                        continue
        
            try:
                del sortSet
            except NameError:
                continue
        except ValueError:
            print('not DateTime')
        
        try:
            del DayAll
        except UnboundLocalError:
            continue
    
    SortedRename = SortedRename.dropna(how='all')
    SortedRename = SortedRename.reset_index(drop=True)
    
    #make an extra dataframe for adding in the conditions
    SortedRename2 = SortedRename
    SortedRename2 = SortedRename2.dropna(how='any')
    SortedRename2 = SortedRename2.reset_index(drop=True)
    #now read in condition data from excel spreadsheet

    exceldf = pd.read_excel(ScreenPrintOut)
    try:
        SortedRename2['Drug'] = exceldf['Drug']
    except KeyError:
        print ('no Drugs')
    try:
        SortedRename2['Concentration'] = exceldf['Concentration (uM)']    
    except KeyError:
        print ('no Concentration')
    try:
        SortedRename2['N_worms'] = exceldf['N_Worms']
    except KeyError:
        print ('No n_worms')
    try:
        SortedRename2['Pos'] = exceldf['Rig_Pos']
    except KeyError:
        print ('No position information')
    try:
        SortedRename2['Strain'] = exceldf['Strain']
    except KeyError:
        print ('No Strain info')
    try:
        SortedRename2['Combination'] = exceldf['Combination']
    except KeyError:
        print ('no Combinations')

    #now add the wormencoder back in
    SortedRename2 = SortedRename2.append(SortedRename[SortedRename['date'].isna()])
    
    return SortedRename2, unprocessed


def get_new_names(SortedRename, OutputFolder, raw_ext):
    """ Assigning the new filename according to the SortedRename dataframe, and
    creating a directory to move the files into
    Input -
        SortedRename: dataframe containing the old filename, and the metadata to attach to new
    
        OutputFolder: -folder to create the RawVideos and ExtraFiles folders into
        
        raw_ext: raw extension that is added on to the videos before processing
    
    Output - 
        SortedRenameNew: DataFrame of the new FileNames
    
        renameFiles.csv - spreadsheet of the metadata
    
        renameLog.tsv - initiation of empty file ready for logging the files to move
    """
    SortedRenameNew = SortedRename
    newNames = []
    
    #create output directory where the files are going to be moved
    output_dir = os.path.join(OutputFolder, 'RawVideos')
    
    #create directory of log
    renameLogdir = os.path.join(OutputFolder, 'ExtraFiles', 'renameLog.tsv')

    #create output if doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.dirname(renameLogdir)):
        os.makedirs(os.path.dirname(renameLogdir))
    #now loop through to find files
    for row in SortedRename.iterrows():
        root, file = os.path.split(row[1]['Old'])
        try:
            if 'Strain' in SortedRename.columns:
                new_base = '{}_Set{}_Pos{}_Ch{}_{}_{}{}'.format(row[1]['Strain'],
                                                                int(row[1]['Set']),
                                                                int(row[1]['Pos']),
                                                                int(row[1]['channel']),
                                                                row[1]['date'].strftime('%y%m%d'),
                                                                row[1]['time'].strftime('%H%M%S'),
                                                                raw_ext)
            
            if 'Drug' in SortedRename.columns:   
                new_base = '{}_{}_Set{}_Pos{}_Ch{}_{}_{}{}'.format(row[1]['Drug'],
                                                        row[1]['Concentration'],
                                                        int(row[1]['Set']), 
                                                        int(row[1]['Pos']), 
                                                        int(row[1]['channel']),
                                                        row[1]['date'].strftime('%y%m%d'),
                                                        row[1]['time'].strftime('%H%M%S'),
                                                        raw_ext)
            
            if 'Combination' in SortedRename.columns:
                new_base = '{}_Set{}_Pos{}_Ch{}_{}_{}{}'.format(row[1]['Combination'],
                                                        int(row[1]['Set']), 
                                                        int(row[1]['Pos']), 
                                                        int(row[1]['channel']),
                                                        row[1]['date'].strftime('%y%m%d'),
                                                        row[1]['time'].strftime('%H%M%S'),
                                                        raw_ext)
    
        except ValueError:
            new_base = file #to add the wormencoder.ini to the list
        
        newNames.append(os.path.join(OutputFolder, 'RawVideos', new_base))

    SortedRenameNew['New'] = newNames
      
    try:
        SortedRenameNew.to_csv(os.path.join(OutputFolder,'ExtraFiles', 'renamedFiles.csv'))
    except FileExistsError:
        print('File Error')

    return SortedRenameNew, renameLogdir



def print_files_to_rename(files_to_rename):
    """ This function lists all the files to be renamed so that can be checked
    before completing the name change
    Input:
        files_to_rename: tuple of lists containing old and new file names
        
    Output:
        prints out file name changes with => in between"""
    for fnames in files_to_rename:
        old_name, new_name = fnames #expand tuples
        new_name = os.path.basename(new_name)
        dnameo, fname_old = os.path.split(old_name)
        
        pc_n = [x for x in dnameo.split(os.sep) if x.startswith('PC')]
        if len(pc_n) > 0:
            pc_n = pc_n[0]
        else:
            pc_n = ''
        
        print('%s => %s' % (os.path.join(pc_n, fname_old), new_name))

def rename_files(files_to_rename, save_renamed_files):
    """ This function renames all the files in the files_to_rename list of tuples
    
    It calls print_files_to_rename first and then asks if okay to rename, so reply 
    y or n
    
    
    Input:
        files_to_rename: list of tuple of old and new name as dirs
        
        save_renamed_files: a record of all the filename changes that occurred
        as a .tsv file, which is created in get_new_names function above
        
    Output:
        Renames files in RawVideos folder
        """
    if not files_to_rename:
        print('No files to be renamed found. Nothing to do here.')
    else:
        print_files_to_rename(files_to_rename)
        reply = input('The files above are going to be renamed. Do you wish to continue (y/N)?')
        reply = reply.lower()
        if reply in ['yes', 'ye', 'y']:
            print('Renaming files...')
            
            #move files and save the changes into _renamed.tsv
            
            with open(save_renamed_files, 'a') as fid:
                for old_name, new_name in files_to_rename:
                    os.rename(old_name, new_name)
                    fid.write('{}\t{}\n'.format(old_name, new_name));
            
            print('Done.')
        else:
            print('Aborted.')

#%%
#run functions

#foldin = '/Volumes/behavgenom_archive$/Ida/MultiWormTracker/LongExposure1'
#excelIn = '/Volumes/behavgenom_archive$/ScreeningExcelPrintout/LongVideos180524.xlsx'
 
foldin = '/Volumes/behavgenom_archive$/Priyanka/Mating Assay/Mating_Assay_080618'
excelIn = '/Volumes/behavgenom_archive$/Priyanka/Mating Assay/Mating_Assay_080618.xlsx'           
existFiles = getFiles(foldin, '.hdf5')
Sorted, notProcessed = FilenameDF(existFiles, excelIn, deltatime = 20)

Renamed, LogDir = get_new_names(Sorted, foldin, '.raw_hdf5')

#make lists of old and new names to export
files1 = Renamed['Old'].tolist()
files2 = Renamed['New'].tolist()

filesfinal = list(zip(files1, files2))
    
print_files_to_rename(filesfinal)
rename_files(filesfinal, LogDir)


