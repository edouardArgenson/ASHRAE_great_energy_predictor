#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 22:26:06 2020

@author: edouard
"""

import pandas as pd
import sklearn.model_selection as ms



#%%


df_features = pd.read_csv('../../data/intermediate/building_1176/features_clean_1176.csv',
                        parse_dates=['timestamp'], index_col=['timestamp'])

# no Nans
df_features.info()

#%%

df_features.head()

#%%


ms._split.__all__



class