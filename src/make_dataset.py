#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:38:37 2020

@author: edouard
"""


import pandas as pd

from os import path




"""
:param filename_suffix: a string in {'train', 'test'} indicating which weather data file to load.
"""
def load_and_prepare_site_data(
        data_folder_path,
        filename_suffix,
        min_timestamp=None, 
        max_timestamp=None, 
        extrapolate_mas=False, 
        drop_nas=False
        ):
    
    if not filename_suffix in ['train', 'test'] :
        raise ValueError('filename_suffix must be eather \'train\' or \'test\'')
    
    print('Loading and preparing each site ' + filename_suffix + ' weather data..')
    
    # Loads weather data
    data_path = path.join(data_folder_path, 'weather_' + filename_suffix + '.csv')
    raw_df_weather = pd.read_csv(
            data_path,
            parse_dates=['timestamp'],
            index_col=['site_id', 'timestamp']
            )
    
    # Get site list
    site_list = raw_df_weather.index.get_level_values('site_id').unique().tolist()
    prepared_site_data = {}
    
    for site in site_list:
        prepared_site_data[site] = prepare_site_data(
                raw_df_weather, 
                site, 
                min_timestamp, 
                max_timestamp, 
                extrapolate_mas, 
                drop_nas
                )
        
    print('done.')
             
    return prepared_site_data





"""
For test set we also perform linear extrapolation on moving average, on a 24h period (contrary to train).
We extrapolate index between min_tps and max_tps. 
If min(max)_tps==None, we use the site weather data min (max) timestamp
Also we do not drop rows with nan(s) for test set.
"""
def prepare_site_data(weather_df, site_id, min_tps=None, max_tps=None, extrapolate_mas=False, drop_nas=False):
    
    b_df_weather = weather_df.loc[(site_id,)]

    # keep only air_temperature and dew_temperature
    b_df_weather.drop(
        ['precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed', 'cloud_coverage'],
        axis=1,
        inplace=True
    )

    # Clean timestamps index.
    b_df_weather = clean_index(b_df_weather, min_tps, max_tps)
    
    # Interpolate missing values.
    b_df_weather.interpolate(method='linear', limit=3, inplace=True)
    
    # Build time features
    b_df_weather['day_hour'] = b_df_weather.index.to_series().dt.hour
    b_df_weather['day_of_week'] = b_df_weather.index.to_series().dt.dayofweek

    # Builds averaged weather features.

    timeframes = [24]
    features_to_avg = ['air_temperature', 'dew_temperature']
    do_center = False

    for c in features_to_avg:
        ts = b_df_weather[c]
        for timeframe in timeframes:
            
            ma_ts = ts.rolling(timeframe, center=do_center).mean()
            new_col_name = '' + c + '_ma_' + str(timeframe) + 'H'
            
            # Extrapolate missing values (specific to test set preparation)
            if extrapolate_mas:
                ma_ts = ma_ts.interpolate(
                    method='linear',
                    limit=24,
                    limit_direction='backward', 
                    limit_area='outside', 
                    inplace=False
                )
            
            b_df_weather[new_col_name] = ma_ts
            
    # Drop rows with NaNs (for training).
    if drop_nas:
        b_df_weather.dropna(axis=0, how='any', inplace=True)
            
    return b_df_weather

#%%

def clean_index(b_df_weather, min_tps, max_tps):
    
    if min_tps==None:
        min_tps = b_df_weather.index.min()
    if max_tps==None:
        max_tps = b_df_weather.index.max()
    
    clean_index = pd.date_range(start=min_tps, end=max_tps, freq='H')
    
    b_df_weather_cleaned = b_df_weather.reindex(index=clean_index, copy=True)
    b_df_weather_cleaned.sort_index(inplace=True)

    return b_df_weather_cleaned

#%%

"""
Maps each building (and meter) to its site.
"""    
def build_meter_site_table(data_folder, prediction_rows):
    
    building_site_data = pd.read_csv(
        path.join(data_folder, 'building_metadata.csv'), 
        index_col='building_id', 
        usecols=['building_id', 'site_id']
    )
    
    buildings_and_meters = prediction_rows.groupby(['building_id', 'meter']).count()
    buildings_and_meters.drop(buildings_and_meters.columns, axis=1, inplace=True)
    
    return buildings_and_meters.join(building_site_data, on='building_id', how='left')

