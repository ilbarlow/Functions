#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:23:53 2019

@author: ibarlow
"""
from scipy import stats
import statsmodels.stats.multitest as smm
import sys
import pandas as pd
import itertools

def statsFeat(FeatureDF, distributionType, threshold, drugControl):
    """ function to do statistics on a dataframe and correcting for multiple 
    comparisons
    Input:
         FeatureDF - multilevelinde- grouped dataframe containing the features,
         and columns date, drug, and worm_number
         
         distributionType - 'normal' or 'other'
         
         threshold - critical values false discovery rate (Q). Default = 0.05
        
    Output:
        pValsDF - dataframe containing raw p values after the pairwise test
        
        bhP_values - multilevel index dataframe containing only the pValues 
        that are significant after controlling for false discovery using the 
        benjamini hochberg procedure, which ranks the raw p values,
        then calculates a threshold as (i/m)*Q,  where i=rank, m=total number 
        of samples, Q=false discovery rate; p values < (i/m)*Q are considered
        significant. Uncorrected p-values are returned where the null 
        hypothesis has been rejected 
        
        """
    if distributionType.lower() == 'normal':
        test = stats.ttest_ind
    else:
        test = stats.ranksums
        
    if threshold == None:
        threshold = 0.05
        
    if drugControl == None:
        cc = 'DMSO'
    else:
        cc= drugControl

    if not isinstance(FeatureDF, pd.core.groupby.DataFrameGroupBy):
        print ('not a multilevel index dataframe')
        return
        
    #match up keys with values
    metadata_dict = {}
    for i in FeatureDF.all().index.levels:
        metadata_dict[i.name] = list(i.values)
    
    #make a list generator to iterate through and great pValues dataframe
    pVals_iterator = list(itertools.product(*metadata_dict.values())) #upack the dictionary
    pVals = pd.DataFrame()
    for item in pVals_iterator:
        if cc not in item:
            control = tuple(s if type(s)!=str else cc for s in item)
            try:
                _vals = pd.Series()
                for i,r in FeatureDF.get_group(item).iteritems():
                   if i != 'drug' or i!='date' or i!='worm_number' or i!='window':
                       _vals[i] = test(r.values, FeatureDF.get_group(control)[i].values)[1]
                _vals['metadata'] = item
                pVals = pVals.append(_vals, ignore_index=True)
            except Exception as error:
                print ('error processing ' + str(error))
           
    bhP_values = pd.DataFrame(columns = pVals.columns)
    for i,r in pVals.iterrows():
        _corrArray = smm.multipletests(r.drop(index= ['metadata']).values, \
                          alpha = threshold, method = 'fdr_bh', is_sorted = False, \
                          returnsorted= False) #applied by row
        _corrArray = r.drop(index =['metadata'])[_corrArray[0]]
        _corrArray['metadata'] = r['metadata']
        bhP_values = bhP_values.append(_corrArray, ignore_index=True)
     
    #make new column for number of significantly different
    bhP_values['sumSig'] = bhP_values.notna().sum(axis=1)
    bhP_values = bhP_values.set_index('metadata')

    return pVals, bhP_values

if __name__ == '__main__':
    GroupedFeatureDF = sys.argv[1]
    distributionType = sys.argv[2]
    threshold = sys.argv[3]
    drugControl = sys.argv[4]
    
    statsFeat(GroupedFeatureDF, distributionType, threshold, drugControl)
