#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:01:57 2018

@author: ibarlow
"""
import sys
import os
import re
 
def takeSetChannel(elem):
    """ This function takes the folder containing the filename and returns set and channel"""
    setno = r"(?<=run|set)\d{1,}"
    s = int(re.findall(setno, elem)[0])
    
    channel =r"(?<=Ch)\d{1,}"
    c = int(re.findall(channel, elem)[0])
    return s,c
    
def displayfiles(validExt, inputfolder, outputfile):
    """ This function finds all the raw file names and saves them to a tsv file
    Input : 
        validExt - the extension to look for for the ifles
        
        outputfile - the name to save the filenames into eg. RawFileNames.tsv
        
        inputfolder - string of the directory to scan through for the files, or none to start a file dialog
        """ 
    
    if inputfolder ==None:
        #popup window
        from tkinter import Tk, filedialog
        print ('Select Data Folder')
        root = Tk()
        root.withdraw()
        root.lift()
        root.update()
        root.directory = filedialog.askdirectory(title = "Select Results folder", \
                                                 parent = root)
        inputfolder = root.directory
    else:
        inputfolder = inputfolder

    #list the files
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
        existingFiles.sort(key=takeSetChannel)
    except AttributeError:
        print ('No Set Name and Channel name in filenames')
    
    #save the output file
    #try:
    savefile = os.path.join (os.path.dirname(inputfolder), outputfile)
    with open(savefile, 'w+') as myfile:
        for item in existingFiles:
            myfile.write(item + '\n')
    print (existingFiles)
    return inputfolder, existingFiles
        
if __name__ == '__main__':
    #inputfolder = argv[0]
    validExt = sys.argv[1]
    inputfolder = sys.argv[2]
    outputfile = sys.argv[3]
    displayfiles(validExt, inputfolder, outputfile)
