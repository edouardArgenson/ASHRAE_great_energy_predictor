#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:50:37 2020

@author: edouard
"""


#%%
import pandas as pd
import numpy as np
import joblib
import os
import itertools

from sys import stdout
from os import path
from time import time, localtime, strftime
from datetime import timedelta

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from tscv import GapKFold

import make_dataset

from model_utils import ModelContainer, MeanByMultiCatEstimator


#%%
    

data_folder = '../data/raw/csvs/'
base_directory_path = '../models/training_no_grid_search/'
n_models = None


#%%


"""
Does not perform grid-search.
Saves models.

for site in site_id:

    load_and_prepare_site_data()

    for building on this site:
        
        for meter in building_meter:
        
            load_meter_data()
            prepare_train_set()
            
            cross-validate()
            fit()
            
            save()
"""

def perform_training(data_folder, model_base_folder, n_meters_max):
    
    print('Loading training meter_reading data...')
    train_df = pd.read_csv(path.join(data_folder, 'train.csv'), parse_dates=['timestamp'])
    print('done.')
    
    # Prepare each site weather data.
    site_weather_data_dict = make_dataset.load_and_prepare_site_data(
            data_folder,
            'train',
            min_timestamp=None, 
            max_timestamp=None, 
            extrapolate_mas=False, 
            drop_nas=True
            )
    
    # Retrieve each building site.
    meter_site_table = make_dataset.build_meter_site_table(data_folder, train_df)
    
    # sub-sample train_df
    
    if n_meters_max != None:
        np.random.seed(102)
        subsample_indexes = np.random.choice(meter_site_table.shape[0], n_meters_max, replace=False)
        subsample_building_meters = meter_site_table.iloc[subsample_indexes]
    else:
        subsample_building_meters = meter_site_table
    
    # Compute n_meters (even if we already know it)
    n_meters = len(meter_site_table.index)
    
    print('n_meters={}'.format(n_meters))
    
    
    tot_start_time = time()
    
    
    results = []
    meter_count = 0
    
    # Create directory to save models
    (training_folder_path, model_folder_path) = build_timestamped_model_folder(model_base_folder)
    
    print('Begin training.')

    # GapKFold
    # gap ~ two weeks, train = 1 month (12 folds)
    gap = 24*7*2
    gap_kf = GapKFold(n_splits=12, gap_before=gap, gap_after=gap)

    for building, meter in subsample_building_meters.index:
        
        site = subsample_building_meters.loc[(building, meter)]['site_id']

        meter_start_time = time()
        meter_count += 1
        
        print('.k={:4} (b, m) = ({:4}, {}), training..'.format('{},'.format(meter_count-1), building, meter), end='')
        stdout.flush()

        x_train, y_train = prepare_meter_train_set(site_weather_data_dict[site], train_df, building, meter)
        
        # List of (trained model object, model_score, is_robust, model_name)
        model_tuple_list = []
        
        # Dummy estimator cross-validation score
        
        dummy_regressor = DummyRegressor(strategy="mean")
        dummy_score = cross_val_score(
            estimator=dummy_regressor,
            X=x_train,
            y=y_train,
            scoring='neg_mean_squared_log_error',
            cv=gap_kf
        ).mean()
        
        dummy_regressor.fit(x_train, y_train)
        
        model_tuple_list.append((dummy_regressor, dummy_score, True, type(dummy_regressor).__name__))
        
        # Time-only model cross-validation score
        
        day_hour_col_idx = x_train.columns.to_list().index('day_hour')
        day_of_week_col_idx = x_train.columns.to_list().index('day_of_week')
        time_col_indexes = [day_hour_col_idx, day_of_week_col_idx]
        
        time_avg_model = MeanByMultiCatEstimator(time_col_indexes)
        time_avg_score = cross_val_score(
            estimator=time_avg_model,
            X=x_train,
            y=y_train,
            scoring='neg_mean_squared_log_error',
            cv=gap_kf
        ).mean()
        
        time_avg_model.fit(x_train, y_train)
        
        model_tuple_list.append((time_avg_model, time_avg_score, True, type(time_avg_model).__name__))
        
        # Time + weather random forest model
        
        rfr_model = RandomForestRegressor(n_estimators=70, max_depth=10)
        
        rfr_no_gcv_score = cross_val_score(
            estimator=rfr_model,
            X=x_train,
            y=y_train,
            scoring='neg_mean_squared_log_error',
            cv=gap_kf,
            n_jobs=6,
        ).mean()
        
        # fit model
        rfr_model.fit(X=x_train, y=y_train)
        
        model_tuple_list.append((rfr_model, rfr_no_gcv_score, False, type(rfr_model).__name__))
        
        # Select models
        (best_model, helper_model), best_model_score = select_models(model_tuple_list)
        
        # save model(s) as ModelContainer
        model_container = ModelContainer(best_model, helper_model)
        meter_folder_name = 'b_' + str(building) + '_m_' + str(meter) + '.joblib'
        model_path = os.path.join(model_folder_path, meter_folder_name)
        joblib.dump(model_container, model_path)
        
        # training info savings in df 
        
        unzipped_model_tuple_list = list(zip(*model_tuple_list))
        scores = unzipped_model_tuple_list[1]
        model_types = unzipped_model_tuple_list[3]
        
        result_row = [
            building, 
            meter,
            site,
            *scores,
            best_model_score,
            type(best_model).__name__,
            type(helper_model).__name__
        ]
        
        results.append(result_row)
        
        training_time = time() - meter_start_time
        
        print(' done, training_time = {:<6} seconds'.format(round(training_time, 3)))
        
        if meter_count%50==0:
            print_progress(meter_count, n_meters, tot_start_time)
                
    tot_end_time = time()
            
            
    print('total time : {}'.format(timedelta(tot_end_time-tot_start_time)))
    
    col_names = ['building', 'meter', 'site', *model_types, 'best_score', 'best_model', 'helper_model']
    results_df = pd.DataFrame(results, columns=col_names)
    
    results_df.to_csv(os.path.join(training_folder_path, 'training_info.csv'), index=False)
    
    
    
"""
Selects only meter data from a specific building and meter id. 
Drops rows that are not in both dtaframe indexes.
Converts Y from pd.df to pd.Series
"""
def prepare_meter_train_set(weather_data, meter_readings, building, meter):
    
    to_keep = (meter_readings['building_id']==building) & (meter_readings['meter']==meter)
    this_meter_readings = meter_readings[to_keep].copy()

    this_meter_readings.drop('building_id', axis=1, inplace=True)
    this_meter_readings.drop('meter', axis=1, inplace=True)

    this_meter_readings.set_index('timestamp', inplace=True)
    this_meter_readings.sort_index(inplace=True)
    
    common_index = weather_data.index.intersection(other=this_meter_readings.index)
    
    # Reset indexes
    
    X = weather_data.loc[common_index].copy()
    Y = this_meter_readings.loc[common_index].copy()

    return (X, Y['meter_reading'])




    
    
#%%
    
def build_timestamped_model_folder(base_directory_path):
    
    timed_base_folder_name = 'training_' + strftime('%Y%m%d_%H%M%S', localtime())
    training_folder_path = path.join(base_directory_path, timed_base_folder_name)
    os.mkdir(training_folder_path)
    
    print('Created training folder: {}'.format(training_folder_path))
    
    model_folder_path = path.join(training_folder_path, 'trained_models')
    os.mkdir(model_folder_path)
    
    return (training_folder_path, model_folder_path)
    
    
    
#%%
    
    
"""
:param model_tuple_list: a list of tuples (model, score, is_robust),
the score is the model cross_validation score, 
is_robust a boolean equal to true if the model can predict rows containing nans values for weather features.
"""
def select_models(model_tuple_list):
    
    best_model = None
    helper_model = None
    
    best_model, best_score, is_robust = parse_for_best(model_tuple_list, True)
    
    if not is_robust:
        robust_model_list = list(itertools.filterfalse(lambda elem: not elem[2], model_tuple_list))
        helper_model = parse_for_best(robust_model_list)
    
    return ((best_model, helper_model), best_score)
    
    
def parse_for_best(model_list, get_robustness_and_score=False):
    unzipped = list(zip(*model_list))
    models = unzipped[0]
    scores = np.array(unzipped[1])
    
    best_model_arg = np.argmax(scores)
    best_model = models[best_model_arg]
    
    if not get_robustness_and_score:
        return best_model
    
    robustnesses = unzipped[2]
    is_robust = robustnesses[best_model_arg]
    
    best_score = scores[best_model_arg]
    
    return (best_model, best_score, is_robust)
    

def print_progress(current_meter, n_meters, start_time):
    
    curr_time_seconds = time()-start_time
    curr_time_r = timedelta(seconds=round(curr_time_seconds))
    
    mean_training_time_seconds = curr_time_seconds / current_meter
    mean_training_time = timedelta(seconds=mean_training_time_seconds)
    
    est_tot_time_r = timedelta(seconds=round(mean_training_time_seconds * n_meters))
    
    print('---')
    print('meters: {}/{}'.format(current_meter, n_meters))
    print('time: {}/{}'.format(curr_time_r, est_tot_time_r))
    print('mean training time: {}'.format(mean_training_time))
    print('---')

#%% Main

if __name__ == "__main__":
    perform_training(data_folder, base_directory_path, n_models)
    


    
    
