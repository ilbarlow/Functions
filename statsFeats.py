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
import numpy as np

def statsFeat(FeatureDF, distributionType, threshold, drugControl):
    """ function to do statistics on a dataframe and correcting for multiple comparisons
    Input:
         FeatureDF - dataframe containing the features and columns date, drug, and worm_number
         
         distributionType - 'normal' or 'other'
         
         threshold - critical values false discovery rate (Q). Default = 0.05
        
    Output:
        pValsDF - dataframe containing raw p values after the pairwise test
        
        bhP_values - dataframe containing only the pValues that are significant 
        after controlling for false discovery using the benjamini hochberg procedure,
        which ranks the raw p values, then calculates a threshold as (i/m)*Q, 
        where i=rank, m=total number of samples, Q=false
        discovery rate; p values < (i/m)*Q are considered significant.
        Uncorrected p-values are returned where the null hypothesis has been rejected 
        
        """
    if isinstance(FeatureDF.index, pd.core.index.MultiIndex):
        FeatureDF = FeatureDF.reset_index()

    datesAll =list( np.unique(FeatureDF['date']))
    drugsAll = list(np.unique(FeatureDF['drug']))
    nworms = list(np.unique(FeatureDF['worm_number']))
    
    RawFeatGrouped = FeatureDF.groupby(['date', 'drug', 'worm_number'])
    
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
    
    pVals = pd.DataFrame()
    for date in datesAll:
        for drug in drugsAll:
                if drug != cc:
                    for n in nworms:
                        valsTemp = pd.Series()
                        try:
                            for i,r in RawFeatGrouped.get_group((date,drug,n)).iteritems(): #i is feature name, r is drug data
                                if i == 'drug' or i=='date' or i=='worm_number':
                                    continue
                                else:
                                    valsTemp[i] = test(r.values, RawFeatGrouped.get_group((date,cc,n))[i].values)[1] #use rank sum test instead

                            valsTemp = valsTemp.append (pd.Series({'date': str(date),\
                                                                   'drug':drug,\
                                                                   'worm_number':int(n)}))
                            pVals = pVals.append(valsTemp, ignore_index=True)
                            del valsTemp
                            
                        except KeyError:
                            print ('error processing ' + ', '.join([str(date),drug,str(n) + ' worms']))
    
    #correct for multiple comparisons - set false discovery using benjamini-hochberg
    bhP_values = pd.DataFrame(columns = pVals.columns)
    for i,r in pVals.iterrows():
        corrArray = smm.multipletests(r.drop(index= ['date', 'drug', 'worm_number']).values, \
                          alpha = threshold, method = 'fdr_bh', is_sorted = False, \
                          returnsorted= False) #applied by row
        corrArray = r.drop(index =['date', 'drug', 'worm_number'])[corrArray[0]]
        corrArray = corrArray.append(pd.Series({'date': int(r.date), 'drug':r.drug, \
                                                'worm_number':int(r.worm_number)}))
        bhP_values = bhP_values.append(corrArray, ignore_index=True)
        del corrArray

    #make new column for number of significantly different
    bhP_values['sumSig'] = bhP_values.notna().sum(axis=1)

    return pVals, bhP_values

if __name__ == '__main__':
    GroupedFeatureDF = sys.argv[1]
    distributionType = sys.argv[2]
    threshold = sys.argv[3]
    drugControl = sys.argv[4]
    
    statsFeat(GroupedFeatureDF, distributionType, threshold, drugControl)
    
          