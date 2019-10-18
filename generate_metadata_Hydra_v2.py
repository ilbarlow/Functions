#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:56:38 2019

@author: ibarlow
"""

""" Function to collate well numbers from features summaries, filenames and 
fuse with the metadata.csv. Make sure only one feat summary and filename per folder"""

from pathlib import Path
import pandas as pd
import re
import json


#global variables and regular expressions
set_no = r"(?<=run|set)\d{1,}"
date = r"(?=\d{8}_)\d{8}"
camera = r"(?<=20\d{6}\_\d{6}\.)\d{8}"
meta_index = ('date_yyyymmdd',
              'run_number',
              'well_name',
              'instrument_name')

HYDRA2CAM_DF = pd.DataFrame({'Hydra01':["22956818", "22956816", "22956813", "22956805", "22956807","22956832"],
                  'Hydra02':["22956839", "22956837", "22956836", "22956829","22956822","22956806"],
                  'Hydra03':["22956814", "22956827", "22956819", "22956833", "22956823","22956840"],
                  'Hydra04':["22956812", "22956834","22956817","22956811","22956831", "22956809"],
                  'Hydra05':["22594559", "22594547", "22594546","22436248","22594549","22594548"]})

def read_json_data(project_directory,dates_to_analyse):
    # add in extracting out the temperature and humidity data
    extra_jsons = list(Path(project_directory).rglob('*extra_data.json'))
    
    if len(extra_jsons)>0:
        json_metadata = pd.DataFrame()
        for e in extra_jsons:
            e= str(e)
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
    
    return json_metadata


def generate_metadata(project_directory, metadata_file, json_metadata):
    """ project_directory = directory containing the input metadata.csv file and
    all the featuresN.hdf5 files"""
    
    feature_summary_file = list(Path(project_directory).rglob('features_summary*'))[0]
    filename_summary_file = list(Path(project_directory).rglob('filenames_summary*'))[0]
    
    try:
        input_metadata_fname = Path(metadata_file)
    except Exception:
        input_metadata_fname = list(Path(project_directory).rglob('metadata.csv'))[0]
    
    metadata_in = pd.read_csv(input_metadata_fname, index_col=False)
    metadata_in.drop(columns = 'filename',inplace=True)
    dates_to_analyse = list(metadata_in.date_yyyymmdd.unique())
    
    #make multi-level index
    metadata_in.index = pd.MultiIndex.from_arrays([metadata_in.date_yyyymmdd,
                                                  metadata_in.run_number,
                                                  metadata_in.well_name,
                                                  metadata_in.instrument_name],
                                                  names = meta_index)
    metadata_in.drop(columns = ['date_yyyymmdd','run_number','well_name','instrument_name'],inplace=True)
             
    #extract from the filename the setnumber, date, camera number and then from file extract well names
    metadataMat = pd.concat([pd.read_csv(feature_summary_file, index_col='file_id')[['well_name']],
                         pd.read_csv(filename_summary_file,index_col='file_id')], axis=1)
    metadataMat.reset_index(drop=False,inplace=True)
    
    metadata_extract = pd.DataFrame()
    for i,r in metadataMat.iterrows():
        _date = int(re.findall(date, r.file_name)[0])
        if _date in dates_to_analyse:
            _set = re.findall(set_no, r.file_name, re.IGNORECASE)[0]
            _camera = re.findall(camera,r.file_name)[0]
            _rig = HYDRA2CAM_DF.columns[(HYDRA2CAM_DF == _camera).any(axis=0)][0]       
           
            metadata_extract = metadata_extract.append(r.append(pd.Series(
                                {'run_number':int(_set),
                                 'date_yyyymmdd':_date,
                                 'camera_no': _camera,
                                 'instrument_name':_rig})),ignore_index=True)
          
    metadata_extract.reset_index(drop=True, inplace=True)
    metadata_extract.index = pd.MultiIndex.from_arrays([metadata_extract.date_yyyymmdd,
                                                        metadata_extract.run_number,
                                                        metadata_extract.well_name,
                                                        metadata_extract.instrument_name],
                                                        names = meta_index)
    metadata_extract.drop(columns = ['date_yyyymmdd','run_number','well_name','instrument_name'],inplace=True)
    
    #concatenate together so that can merge
    metadata_concat = pd.concat([metadata_extract, metadata_in], axis=1, join='inner', sort=True)
    metadata_concat.reset_index(drop=False, inplace=True)
       
    #save to csv
    metadata_concat.to_csv(input_metadata_fname.parent / 'updated_metadata.csv', index=False)

    if json_metadata:
        json_metadata = read_json_data(project_directory, dates_to_analyse)
        # summarise json metadata by export metadata 
        json_metadata.to_csv(project_directory / 'extra_data.csv')
    
        return metadata_concat, json_metadata
    
    else:
        return metadata_concat

if __name__=='__main__':
#    project_directory = sys.argv[0]
#    metadata_file = sys.argv[1]
    
    project_directory = '/Volumes/behavgenom$/Luigi/Data/LoopBio_tests/'
    metadata_file = '/Volumes/behavgenom$/Luigi/Data/LoopBio_tests/metadata.csv'
    
    generate_metadata(project_directory, metadata_file)