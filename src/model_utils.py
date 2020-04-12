#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:52:27 2020

@author: edouard
"""

#%%
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

#%%


"""
A model that create categories from feature values,
and computes the mean for each category values.
Those means are the predicted values.
Extends Scikit-Learn BaseEstimator so it can be used with all Scikit-Learn functions.
"""
class MeanByMultiCatEstimator(BaseEstimator):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, cat_column_indexes=[0], verbose=False):
        self.verbose = verbose
        self.cat_column_indexes = cat_column_indexes

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        
        X, y = check_X_y(X, y, accept_sparse=True)
        """Input validation for standard estimators.
        Checks X and y for consistent length, enforces X to be 2D and y 1D. By
        default, X is checked to be non-empty and containing only finite values.
        Standard input checks are also applied to y, such as checking that y
        does not have np.nan or np.inf targets. For multi-label y, set
        multi_output=True to allow 2D and sparse y. If the dtype of X is
        object, attempt converting to float, raising on failure.
        """
        
        
        cat_columns = []
        
        for col_idx in self.cat_column_indexes:
            if(col_idx >= X.shape[1]):
                raise ValueError("category column indexes should be < X.shape[1]")
            cat_columns.append(X[:, col_idx])
            
        cat_tuples = set(zip(*cat_columns))
        
        categories = {}
        self.means = {}
        
        self.mean = y.mean()
        
        for x_bin in cat_tuples:
            categories[x_bin] = []
            
        if self.verbose:    
            print('categories : {}'.format(categories.keys()))
            
        for k in range(X.shape[0]):
            sample_bin = tuple(X[k, self.cat_column_indexes])
            categories[sample_bin].append(y[k])
        
        for k, v in categories.items():
            self.means[k] = np.array(v).mean()
        
        self.is_fitted_ = True
        # `fit` should always return `self`
        
        if self.verbose:
            for k, v in self.means.items():
                print('({}, {})'.format(k, v))
        
        return self

    
    
    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        
        
        X = check_array(X, accept_sparse=True, force_all_finite='allow-nan')
        """Input validation on an array, list, sparse matrix or similar.
        By default, the input is checked to be a non-empty 2D array containing
        only finite values. If the dtype of the array is object, attempt
        converting to float, raising on failure."""
        
        # Do not allow nans in time-features
        check_array(X[:, self.cat_column_indexes], accept_sparse=True)
        
        check_is_fitted(self, 'is_fitted_')
        
        predictions = []
        
        cat_columns=[]
        for col in self.cat_column_indexes:
            cat_columns.append(X[:, col])
            
        cat_tuples = list(zip(*cat_columns))
        
        for sample_cat in cat_tuples:
            cat_mean = self.means.get(sample_cat)
            if(cat_mean == None):
                predictions.append(self.mean)
            else:
                predictions.append(cat_mean)
            
        return np.array(predictions)


#%%
        
"""
A container to save a pair of model : the best performing model,
and if the best performing model cannot predict rows containing nans, a less-performant helperModel.
This could be modified to extend BaseEstimator.
"""
class ModelContainer:
    
    def __init__(self, best_model_arg, helper_model_arg):
        self.best_model = best_model_arg
        self.helper_model = helper_model_arg
        
    """
    If the model cannot predict rows with nans, separate them from rows without nans (clean rows).
    Then use helper model to predict rows with nans.
    X should be a pandas Dataframe.
    Returns a pandas Series.
    """
    def predict(self, X):
        
        print("predicting...")
        
        if(self.helper_model == None):
            return pd.Series(data=self.best_model.predict(X), index=X.index)
        
        clean_rows = X.isna().sum(axis=1) == 0
        
        print('{} clean rows'.format(clean_rows.sum()))
        print('{} dirty rows'.format((~clean_rows).sum()))
        
        clean_index = X[clean_rows].index
        dirty_index = X[~clean_rows].index
        
        clean_preds = self.best_model.predict(X.loc[clean_index])
        preds_df_clean = pd.Series(data=clean_preds, index=clean_index)
        
        if dirty_index.empty:
            return preds_df_clean
            
        dirty_preds = self.helper_model.predict(X.loc[dirty_index])
        preds_df_dirty = pd.Series(data=dirty_preds, index=dirty_index)
        
        preds = pd.concat([preds_df_clean, preds_df_dirty], axis=0)
        preds.sort_index(inplace=True)
        
        return preds