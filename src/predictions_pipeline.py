#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:32:59 2020

@author: edouard
"""


#%%
import pandas as pd
import joblib
import os

from os import path
from time import localtime, strftime

import make_dataset

from model_utils import ModelContainer, MeanByMultiCatEstimator

#%%


training_time_id = '20200405_153303'

data_folder = '../data/raw/csvs/'

# retrieve training directory path
base_directory_path = '../models/test/'


#%% Main fonction, executes all pipe.

"""
- prepare each site weather data

- get each building site

- (for tests only) clean building list, keep only buildings for which we saved a model (using training_infos.csv)

- for all meters
    load and predict

- save predictions as csv
"""
def make_predictions(data_folder, trained_model_base_folder, training_timestamp):
    
    # Path to saved models and training info.
    training_folder = build_training_folder_path(trained_model_base_folder, training_timestamp)
    
    # Load prediction timestamps for each meter of each building.
    (prediction_rows, min_timestamp, max_timestamp) = load_prediction_rows(data_folder)
    
    # Prepare each site weather data.
    site_weather_data = make_dataset.load_and_prepare_site_data(data_folder, min_timestamp, max_timestamp)

    # Retrieve each building site.
    meter_site_table = build_meter_site_table(data_folder, prediction_rows)

    # Keep only (building, meter) for which we trained a model.
    sub_meter_site_table = filter_buildings_without_model(training_folder, meter_site_table)

    prediction_dfs = []

    for (building, meter) in sub_meter_site_table.index:
        
        print('({}, {})'.format(building, meter))
        
        meter_predictions = load_model_and_predict(
                training_folder,
                site_weather_data,
                meter_site_table,
                prediction_rows,
                building,
                meter
                )
        
        prediction_dfs.append(meter_predictions)
    
    # Prepare submission csv
    submission_df = shape_submission_df(prediction_rows, prediction_dfs)
    
    # Save submissions
    save_predictions(submission_df, training_folder)
    
    return submission_df
    
    
#%% Implementation details

def load_prediction_rows(data_folder):
    
    test_ids = pd.read_csv(path.join(data_folder, 'test.csv'), parse_dates=['timestamp'])
    test_ids.set_index('row_id', inplace=True) # (in two steps to avoid a warning)
    
    min_tps = test_ids['timestamp'].min()
    max_tps = test_ids['timestamp'].max()
    
    return (test_ids, min_tps, max_tps)


def filter_buildings_without_model(training_folder_path, entry_rows):
    
    # Load training info
    training_info_path = path.join(training_folder_path, 'training_info.csv')
    
    training_info = pd.read_csv(training_info_path)
    
    trained_meter_index = pd.MultiIndex.from_frame(training_info[['building', 'meter_id']])
    
    return entry_rows.loc[trained_meter_index]


"""
for (building, meter):
    - get site
    - get timestamps to predict
    - load model
    - predict
"""
def load_model_and_predict(
        training_folder,
        weather_data, 
        meter_site_table, 
        prediction_rows, 
        building, meter
        ):
    
    # Load model
    model_container = load_model(training_folder, building, meter)
    
    # Extract x_test
    x_test = extract_meter_data(weather_data, meter_site_table, prediction_rows, building, meter)
    
    meter_predictions = model_container.predict(x_test)
    
    meter_predictions_df = pd.DataFrame({
        'building_id' : building,
        'meter' : meter,
        'timestamp' : meter_predictions.index,
        'meter_reading' : meter_predictions.reset_index(drop=True, inplace=False)
    })
    
    return meter_predictions_df


def build_meter_site_table(data_folder, prediction_rows):
    
    building_site_data = pd.read_csv(
        path.join(data_folder, 'building_metadata.csv'), 
        index_col='building_id', 
        usecols=['building_id', 'site_id']
    )
    
    buildings_and_meters = prediction_rows.groupby(['building_id', 'meter']).count()
    buildings_and_meters.drop('timestamp', axis=1, inplace=True)
    
    return buildings_and_meters.join(building_site_data, on='building_id', how='left')
    
    
def load_model(training_folder_path, building, meter):
    b_folder = 'building_' + str(building)
    m_folder = 'meter_' + str(meter)
    model_path = path.join(training_folder_path, b_folder, m_folder, 'model_container.joblib')
    return joblib.load(model_path)


def extract_meter_data(weather_data, meter_site_table, test_rows_ids, building, meter):
    
    site = meter_site_table.loc[(building, meter), 'site_id']
    
    this_b_and_m = (test_rows_ids['building_id']==building) & (test_rows_ids['meter']==meter)
    timestamps_to_predict = test_rows_ids[this_b_and_m]['timestamp']
    
    return weather_data[site].loc[pd.Index(timestamps_to_predict)]


def save_predictions(predictions, folder):
    
    print('saving as csv..')
    
    pred_folder_path = path.join(folder, 'predictions')

    if(not path.isdir(pred_folder_path)):
        os.mkdir(pred_folder_path)
    
    timestamped_filename = 'predictions_' + strftime('%Y%m%d_%H%M%S', localtime()) + '.csv'
    
    prediction_file_path = path.join(pred_folder_path, timestamped_filename)
    
    predictions.to_csv(prediction_file_path)
    
    print('saved to {}'.format(prediction_file_path))


def build_training_folder_path(trained_model_base_folder, training_timestamp):
    timed_base_folder_name = 'trained_models_' + training_timestamp
    return path.join(trained_model_base_folder, timed_base_folder_name)


# Prepare submission csv
def shape_submission_df(row_ids, prediction_dfs):
    predictions_df = pd.concat(prediction_dfs, axis=0)
    predictions_df.set_index(['building_id', 'meter', 'timestamp'], inplace=True)
    submission_df = row_ids.join(predictions_df, on=['building_id', 'meter', 'timestamp'], how='left')
    submission_df.drop(['building_id', 'meter', 'timestamp'], axis=1, inplace=True)
    return submission_df

