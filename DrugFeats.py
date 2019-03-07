#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""class to describe drug statistically significant functions"""

class DrugFeats:
	import itertools

	def __init__(self, bhP_dataframe):
		self.drug = bhP_dataframe['drug'].values[0]
		self.date = bhP_dataframe['date'].values[0]
		self.sigFeats = bhP_dataframe #dataframe
		self.nworms = int(bhP_dataframe['worm_number'].values[0])
		self.FeatureList = list(bhP_dataframe.select_dtypes(include='float').columns[(bhP_dataframe.select_dtypes(include='float')<0.05).any()])

	# def compareFeats(self):
	# 	self.consistentFeatures = 