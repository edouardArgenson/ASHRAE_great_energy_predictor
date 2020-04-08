#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:38:37 2020

@author: edouard
"""


import pandas as pd

from os import path




def load_and_prepare_site_data(data_folder_path, min_timestamp, max_timestamp):
    
    print('Loading and preparing each site weather data..')
    
    # Loads weather data
    raw_df_weather = pd.read_csv(path.join(data_folder_path, 'weather_test.csv'), 
                     parse_dates=['timestamp'], index_col=['site_id', 'timestamp'])
    
    # Get site list
    site_list = raw_df_weather.index.get_level_values('site_id').unique().tolist()
    prepared_site_data = {}
    
    for site in site_list:
        prepared_site_data[site] = prepare_site_data(raw_df_weather, site, min_timestamp, max_timestamp)
             
    return prepared_site_data





"""
For test set we also perform linear extrapolation (contrary to train).
We extrapolate between min_tps and max_tps.
Also we do not drop rows with nan(s).
"""
def prepare_site_data(weather_df, site_id, min_tps, max_tps):
    
    b_df_weather = weather_df.loc[(site_id,)]

    # keep only air_temperature and dew_temperature
    b_df_weather.drop(
        ['precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed', 'cloud_coverage'],
        axis=1,
        inplace=True
    )

    # Clean timestamps index.
    clean_index = pd.date_range(start=min_tps, end=max_tps, freq='H')
    b_df_weather = b_df_weather.reindex(index=clean_index, copy=True)
    b_df_weather.sort_index(inplace=True)


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
            shifted_ts = ts.rolling(timeframe, center=do_center).mean()
            new_col_name = '' + c + '_ma_' + str(timeframe) + 'H'
            # Extrapolate missing values (specific to test set preparation)
            extrapolated_shifted_ts = shifted_ts.interpolate(
                method='linear',
                limit=24,
                limit_direction='backward', 
                limit_area='outside', 
                inplace=False
            )
            b_df_weather[new_col_name] = extrapolated_shifted_ts
            
            
    # Do not drop rows with NaNs.
    #b_df_weather.dropna(axis=0, how='any', inplace=True)
            
    print('shape={}'.format(b_df_weather.shape))
        
    return b_df_weather

