#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:01:57 2018

@author: ibarlow
"""
import sys
import os

    
def displayfiles(validExt, inputfolder):
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
    print (existingFiles)
    return existingFiles
        
if __name__ == '__main__':
    #inputfolder = argv[0]
    validExt = sys.argv[1]
    inputfolder = sys.argv[2]
    displayfiles(validExt, inputfolder)