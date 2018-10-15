#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:01:57 2018

@author: ibarlow
"""
import sys
import os
import re

def takeSet(elem):
    """ This function takes the folder  containing the filename returns set and channel"""
    setno = elem.split('/')[-2]
    s = re.search(r"\d+", setno)
    s = int(s.group(0))
    
    channel = elem.split('_')[2]
    c = re.search(r"\d+", channel)
    c = int(c.group(0))
    return s,c
    
def displayfiles(validExt, inputfolder, outputfile):
     """ This function finds all the raw file names and saves them to a tsv file
    Input : 
        validExt - the extension to look for for the ifles
        
        outputfile - the name to save the filenames into eg. RawFileNames.tsv
        
        inputfolder - string of the directory to scan through for the files, or none to start a file dialog
        """
    existingFiles = []
    for (root, dirs, files) in os.walk(inputfolder):
        if files == []: #pass empty folders
            continue
        else:
            for file1 in files: #loop through to add the files
                if file1.endswith(validExt):
                    existingFiles.append(os.path.join(root, file1))
                else:
                    continue
    
    #sort the files
    try:
        existingFiles.sort(key=takeSet)
    except AttributeError:
        print ('No Set Name and Channel name in filenames')
    
    #save the output file
    savefile = os.path.join (os.path.dirname(inputfolder), outputfile)
    with open(savefile, 'w', newline = '') as myfile:
        for item in existingFiles:
            myfile.write(item + '\n')
    
    print (existingFiles)
    return existingFiles
        
if __name__ == '__main__':
    #inputfolder = argv[0]
    validExt = sys.argv[1]
    inputfolder = sys.argv[2]
    outputfile = sys.argv[3]
    displayfiles(validExt, inputfolder, outputfile)
