#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:32:59 2020

@author: edouard
"""


#%%
import pandas as pd
import joblib

from os import path

import make_dataset

from model_utils import ModelContainer, MeanByMultiCatEstimator

#%%


training_time_id = '20200405_153303'

data_folder = '../data/raw/csvs/'



test_df = pd.read_csv(path.join(data_folder, 'test.csv'), parse_dates=['timestamp'])
test_df.set_index('row_id', inplace=True) # (in two steps to avoid a warning)

min_tps = test_df['timestamp'].min()
max_tps = test_df['timestamp'].max()


# retrieve training directory path
base_directory_path = '../models/test/'
timed_base_folder_name = 'trained_models_' + training_time_id
training_folder_path = path.join(base_directory_path, timed_base_folder_name)


"""
- prepare each site weather data

- get each building site

- (for tests only) clean building list, keep only buildings for which we saved a model (using training_infos.csv)

- for all meters
    load and predict

"""



site_data = make_dataset.load_and_prepare_site_data(data_folder, min_tps, max_tps)



#%% Retrieve each building site




#%% Keep only (building, meter) for which we trained a model

# Load training info
training_info_path = path.join(training_folder_path, 'training_info.csv')

training_info = pd.read_csv(training_info_path)

trained_meter_index = pd.MultiIndex.from_frame(training_info[['building', 'meter_id']])
sub_test_df_grouped = test_df_grouped.loc[trained_meter_index]




#%% Load models and predict


"""
for (building, meter) in sub_test_df_grouped.index:
    - get site
    - get timestamps to predict
    - load model
    - predict

"""
def predict_all(prediction_rows):
    
    # TODO do this outside this function
    # Retrieve each building site
    meter_site_table = build_meter_site_table(prediction_rows)
    
    
    # TODO Keep only (building, meter) for which we trained a model
    
    
    
    
    
    prediction_dfs = []

    for (building, meter) in sub_test_df_grouped.index:
        
        print('({}, {})'.format(building, meter))
        
        site = sub_test_df_grouped.loc[(building, meter), 'site_id']
        
        # Load model
        model_container = load_model(building, meter)
        
        # Extract x_test
        # TODO regroup with get_site_data
        x_test = extract_meter_data(building, meter, site_data[site])
        
        meter_preds = model_container.predict(x_test)
        
        meter_preds_df = pd.DataFrame({
            'building_id' : building,
            'meter' : meter,
            'timestamp' : meter_preds.index,
            'meter_reading' : meter_preds.reset_index(drop=True, inplace=False)
        })
        
        prediction_dfs.append(meter_preds_df)
        
    # Prepare submission csv
    submission_df = shape_submission_df(prediction_dfs)
    
    # TODO save submissions
    
    return submission_df
    
    
def build_meter_site_table(data_folder, prediction_rows):
    
    building_site_data = pd.read_csv(
        path.join(data_folder, 'building_metadata.csv'), 
        index_col='building_id', 
        usecols=['building_id', 'site_id']
    )
    
    buildings_and_meters = prediction_rows.groupby(['building_id', 'meter']).count()
    buildings_and_meters.drop('timestamp', axis=1, inplace=True)
    
    return buildings_and_meters.join(building_site_data, on='building_id', how='left')
    
    
def load_model(building, meter):
    b_folder = 'building_' + str(building)
    m_folder = 'meter_' + str(meter)
    model_path = path.join(training_folder_path, b_folder, m_folder, 'model_container.joblib')
    return joblib.load(model_path)


def extract_meter_data(df, building, meter, meter_site_data):
    this_b_and_m = (df['building_id']==building) & (df['meter']==meter)
    timestamps_to_predict = df[this_b_and_m]['timestamp']
    return meter_site_data.loc[pd.Index(timestamps_to_predict)]


# Prepare submission csv
def shape_submission_df(prediction_dfs):
    predictions_df = pd.concat(prediction_dfs, axis=0)
    predictions_df.set_index(['building_id', 'meter', 'timestamp'], inplace=True)
    submission_df = test_df.join(predictions_df, on=['building_id', 'meter', 'timestamp'], how='left')
    submission_df.drop(['building_id', 'meter', 'timestamp'], axis=1, inplace=True)
    return submission_df


















