#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:35:22 2019

@author: ibarlow
"""

""" Function to collate metadata from input .csv and filenames generated by 
Hydra rigs"""

from pathlib import Path
import pandas as pd
import glob
import re
#import sys
import json


#global variables and regular expressions
set_no = r"(?<=run|set)\d{1,}"
date = r"(?=20)\d{8}"
camera = r"(?<=20\d{6}\_\d{6}\.)\d{8}"
meta_index = ('date_yyyymmdd',
              'run_number',
              'well_number',
              'instrument_name')

HYDRA2CAM_DF = pd.DataFrame({'Hydra01':["22956818", "22956816", "22956813", "22956805", "22956807","22956832"],
                  'Hydra02':["22956839", "22956837", "22956836", "22956829","22956822","22956806"],
                  'Hydra03':["22956814", "22956827", "22956819", "22956833", "22956823","22956840"],
                  'Hydra04':["22956812", "22956834","22956817","22956811","22956831", "22956809"],
                  'Hydra05':["22594559", "22594547", "22594546","22436248","22594549","22594548"]})


def generate_metadata(project_directory, metadata_file):
    """ project_directory = directory containing the input metadata.csv file and
    all the featuresN.hdf5 files"""
    
    results_files = glob.glob(str(Path(project_directory) / '**/*metadata_featuresN.hdf5'),
                              recursive=True)
    try:
        input_metadata_fname = Path(metadata_file)
    except Exception:
        input_metadata_fname = Path(glob.glob(str(Path(project_directory) / '**/metadata.csv'),
                                         recursive=True)[0])
    
    metadata_in = pd.read_csv(input_metadata_fname, index_col=False)
    metadata_in.drop(columns = 'filename',inplace=True)
    metadata_in.index = pd.MultiIndex.from_arrays([metadata_in.date_yyyymmdd,
                                                  metadata_in.run_number,
                                                  metadata_in.well_number,
                                                  metadata_in.instrument_name],
                                                  names = meta_index)
            
    dates_to_analyse = list(metadata_in.date_yyyymmdd.unique())
    
    #extract from the results filename the setnumber, date, camera number and then from file extract well names
    metadata_extract = pd.DataFrame()
    for r in results_files:
        #and extract other metadata from the filename
        if 'bluelight' in r:
            continue
        else:           
            _date = int(re.findall(date, r)[0])
            if _date in dates_to_analyse:
                _set = re.findall(set_no, r, re.IGNORECASE)[0]
                _camera = re.findall(camera,r)[0]
                _rig = HYDRA2CAM_DF.columns[(HYDRA2CAM_DF == _camera).any(axis=0)][0]       
                #extra wells from featuresM
                with pd.HDFStore(r, 'r') as fid:
                    wells = list(fid['/fov_wells'].well_name.unique())
        
                metadata_extract = metadata_extract.append(pd.DataFrame(
                                    {'run_number':int(_set),
                                     'date_yyyymmdd':_date,
                                     'camera_no': _camera,
                                     'well_number': wells,
                                     'filename': r,
                                     'instrument_name':_rig}))
        
    metadata_extract.reset_index(drop=True, inplace=True)
    metadata_extract.index = pd.MultiIndex.from_arrays([metadata_extract.date_yyyymmdd,
                                                        metadata_extract.run_number,
                                                        metadata_extract.well_number,
                                                        metadata_extract.instrument_name],
                                                        names = meta_index)
    
    #concatenate together so that can merge
    metadata_concat = pd.concat([metadata_extract, metadata_in], axis=1, join='inner', sort=True)
    metadata_concat = metadata_concat.drop(columns = ['date_yyyymmdd',
                                                      'run_number',
                                                      'well_number',
                                                      'instrument_name'])
    metadata_concat.reset_index(drop=False, inplace=True)
       
    #save to csv
    metadata_concat.to_csv(input_metadata_fname.parent / 'updated_metadata.csv', index=False)

    # add in extracting out the temperature and humidity data
    extra_jsons = glob.glob(str(Path(project_directory) / '**/*extra_data.json'), recursive=True)
    
    if len(extra_jsons)>0:
        json_metadata = pd.DataFrame()
        for e in extra_jsons:
            _date = int(re.findall(date, e)[0])
            if _date in dates_to_analyse:
                _set = re.findall(set_no, e, re.IGNORECASE)[0]
                _camera = re.findall(camera,e)[0]
                _rig = HYDRA2CAM_DF.columns[(HYDRA2CAM_DF == _camera).any(axis=0)][0] 
                with open(e) as fid:
                    extras = json.load(fid)
                    for t in extras:
                        json_metadata = json_metadata.append(pd.concat([pd.DataFrame.from_records([
                                {'run_number' : int(_set),
                                 'date_yyyymmdd':_date,
                                 'camera_no': _camera,
                                 'filename': e,
                                 'filedir': Path(e).parent,
                                 'instrument_name':_rig}]), pd.DataFrame(pd.Series(t)).transpose()], axis=1), ignore_index=True, sort=True)
        # summarise json metadata by     
        json_metadata.to_csv(input_metadata_fname.parent / 'extra_data.csv')
    
    if 'json_metadata' in locals():
        return metadata_concat, json_metadata
    else:
        return metadata_concat

if __name__=='__main__':
#    project_directory = sys.argv[0]
#    metadata_file = sys.argv[1]
    
    project_directory = '/Volumes/behavgenom$/Luigi/Data/LoopBio_tests/'
    metadata_file = '/Volumes/behavgenom$/Luigi/Data/LoopBio_tests/metadata.csv'
    
    generate_metadata(project_directory, metadata_file)