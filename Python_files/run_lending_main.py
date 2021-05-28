# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:32:15 2021

@author: jerko

This is the main file for the Master thesis project:
    'Interpretable neural networks for a multiclass assessment of credit risk'

It pre-processes the data, develops several models for classifying credit loans, 
and evaluates them using several performance measures (AUC, accuracy, F1 score, MAE, and BIC).

Models developed:
    - L-MNL H1: Learning Multinomial Logit (Sifringer et al. (2020)) model using heuristic 1 detailed in the thesis document.
    - L-MNL H2: Learning Multinomial Logit (Sifringer et al. (2020)) model using heuristic 1 detailed in the thesis document.
    - L-MNL H3: Learning Multinomial Logit (Sifringer et al. (2020)) model using heuristic 1 detailed in the thesis document.
    - DNN: Dense Neural Network implemented in Keras.
    - MNL: Multinomial Logit model from statsmodels.
    - Cum Ord logit model: Cumulative ordinal logit model. The Ordistic Lgistic (Ordinal Logit AT) from the mord package was implemented and results from the polr model implemented in R are imported.

Input: accepted credit loan data from Lending club
    George, N. (2019, April 10). All Lending Club loan data, Version 3. Retrieved February 2020 from  https://www.kaggle.com/wordsforthewise/lending-club.

Output: models for classifying credit loans

Note: Python 3.8.8, Tensorflow 2.3.0 and Keras 2.4.3 was used. 
However, if the user wants to extract the standerd deviations of the betas (coefficients) 
in L-MNL models, he/she has to use:
Python 3.6.13, Tensorflow 2.0.0 and Keras 2.3.1 or lower.

"""

import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from scipy.stats import kendalltau
from scipy.stats import spearmanr
import statsmodels.api as st
from tensorflow import keras
from keras import backend as K
import grad_hess_utilities as gu
import models as mdl
from keras.models import Sequential
from keras.models import load_model, Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Concatenate
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, Add, Reshape
from keras.initializers import Constant
from keras.optimizers import Adam
#from keras.losses import mean_squared_error
#from statsmodels.discrete.discrete_model import MNLogit
#from statsmodels.miscmodels.ordinal_model import OrderedModel #Not available
#from statsmodels.tools.tools import add_constant
#from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, log_loss, roc_auc_score, accuracy_score, mean_absolute_error, confusion_matrix, mean_squared_error
from sklearn.metrics.cluster import pair_confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.linear_model import LogisticRegression
#from mord import LogisticIT, LogisticAT
#from mord import LogisticAT
from math import log

import pickle
#import tensorflow.python.util.deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False

import timeit
#from pandas_select import ColumnSelector

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("expand_frame_repr", True)
pd.options.display.float_format = '{:,.2f}'.format
seed = 10
random.seed(seed)


#---------- Data import -----------


def data_import(filepath):
    '''
    Imports the data from a csv file, and does an initial pre-processing
    
    :param str filepath: The path to the foler where the input csv data file is located
    :return: the dataset as a pandas dataframe
    
    '''
    folders = os.listdir(filepath)
    folders = [f for f in folders]
    print(folders)
    
    accepted_fn = filepath + '/' + folders[0]
    data = pd.read_csv(accepted_fn)
    
    # Remove instances with more than 60 NaN (40% of features )
    index = data.isna().sum(axis=1)>60
    print("Instancias eliminadas:", data[index].shape)
    data = data[-index]
 
    '''
    Remove 75 features ('debt_settlement_flag_date', 'deferral_term', 'desc', 'dti_joint', 'id',
    'il_util', 'member_id', 'mths_since_last_major_derog', 'next_pymnt_d', 'orig_projected_additional_accrued_interest'
    'payment_plan_start_date', 'sec_app_revol_util', 'settlement_amount', 'settlement_date'
    'settlement_percentage', 'settlement_status', 'settlement_term', 'total_bal_il',
    'total_cu_tl', 'url', 'verification_status_joint','earliest_cr_line', 'emp_title', 'funded_amnt_inv', 'hardship_dpd',
    'hardship_end_date', 'hardship_last_payment_amount', 'hardship_length', 'hardship_payoff_balance_amount',
    'hardship_reason', 'hardship_start_date', 'hardship_status', 'hardship_type',
    'issue_d', 'last_credit_pull_d', 'last_pymnt_d, 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mths_since_last_record',
    'mths_since_last_delinq', 'mths_since_rcnt_il', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq',
    'mths_since_recent_revol_delinq', 'out_prncp_inv', 'policy_code', 'revol_bal_joint',
    'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med', 'sec_app_earliest_cr_line',
    'sec_app_fico_range_high', 'sec_app_fico_range_low', 'sec_app_inq_last_6mths', 'sec_app_mort_acc',
    'sec_app_mths_since_last_major_derog', 'sec_app_num_rev_accts', 'sec_app_open_acc', 'sec_app_open_act_il',
    'title', 'zip_code', 'grade', 'hardship_loan_status', 'home_ownership', 'application_type', 'pymnt_plan',
    'annual_inc_joint', 'hardship_amount', 'inq_fi', 'inq_last_12m', 'max_bal_bc', 'open_acc_6m', 'open_act_il', 
    'open_il_12m','open_il_24m', 'open_rv_12m', 'open_rv_24m', 'num_tl_120dpd_2m', 'debt_settlement_flag', 'delinq_amnt') - high percentage of missing values and data-leakage
    '''
    
    data = data.drop(['debt_settlement_flag_date', 'deferral_term', 'desc', 'dti_joint', 'id',
                    'il_util', 'member_id', 'mths_since_last_major_derog', 'next_pymnt_d', 'orig_projected_additional_accrued_interest',
                    'payment_plan_start_date', 'sec_app_revol_util', 'settlement_amount', 'settlement_date',
                    'settlement_percentage', 'settlement_status', 'settlement_term', 'total_bal_il',
                    'total_cu_tl', 'url', 'verification_status_joint', 'earliest_cr_line', 'emp_title', 'funded_amnt_inv', 'hardship_dpd',
                    'hardship_end_date', 'hardship_last_payment_amount', 'hardship_length', 'hardship_payoff_balance_amount',
                    'hardship_reason', 'hardship_start_date', 'hardship_status', 'hardship_type', 
                    'issue_d', 'last_credit_pull_d', 'last_pymnt_d', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mths_since_last_record',
                    'mths_since_last_delinq', 'mths_since_rcnt_il', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq',
                    'mths_since_recent_revol_delinq', 'out_prncp_inv', 'policy_code', 'revol_bal_joint', 
                    'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med', 'sec_app_earliest_cr_line',
                    'sec_app_fico_range_high', 'sec_app_fico_range_low', 'sec_app_inq_last_6mths', 'sec_app_mort_acc',
                    'sec_app_mths_since_last_major_derog', 'sec_app_num_rev_accts', 'sec_app_open_acc', 'sec_app_open_act_il',
                    'title', 'zip_code', 'hardship_loan_status', 'home_ownership', 'application_type', 'pymnt_plan',
                    'annual_inc_joint', 'hardship_amount', 'inq_fi', 'inq_last_12m', 'max_bal_bc', 'open_acc_6m', 'open_act_il', 
                    'open_il_12m','open_il_24m', 'open_rv_12m', 'open_rv_24m', 'num_tl_120dpd_2m', 'debt_settlement_flag', 'delinq_amnt'], axis =1)
    
    data.replace({'loan_status':{'Fully Paid': 1, 'Current': 1, 'In Grace Period': 1, 'Late (16-30 days)': 2, 'Late (31-120 days)': 2, 'Charged Off': 3, 'Default': 3}}, inplace = True)
    data.replace({'term': {' 36 months': 36, ' 60 months': 60}, 'purpose': {'wedding': 'house', 'renewable_energy': 'home_improvement', 'moving': 'home_improvement', 'vacation': 'other', 'medical': 'other',
                  'credit_card':'debt_consolidation'}, 'emp_length': {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, 
                                 '9 years': 9, '10+ years': 10}, 'verification_status': {'Source Verified':'Verified'}}, inplace = True)
    data['addr_state'].replace(['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI'], 'Midwest region', inplace = True)
    data['addr_state'].replace(['CT', 'ME', 'MA', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VT'], 'Northeast region', inplace = True)
    data['addr_state'].replace(['AL', 'AR', 'DE', 'DC', 'FL', 'GA', 'KY', 'LA', 'MD', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV'], 'Southern region', inplace = True)
    data['addr_state'].replace(['AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY'], 'Western region', inplace = True)

    # Remove outliers
    
    data.replace({'dti': {-1: 0}, 'total_rev_hi_lim':{9999999: 3000000}}, inplace = True)
    indexEd = data[data['purpose'] == 'educational'].index
    data.drop(indexEd , inplace=True)
    data['total_rec_late_fee'].values[data['total_rec_late_fee'].values < 0] = 0
    
    print(data['loan_status'].value_counts())
    
    return data

#----------- Data down-sampling

def down_sample(df, seed = 10):
    '''
    Downsamples the majority class
    
    :param dataframe df: The dataset
    :param int seed: The random seed number
    :return: downsampled dataset
    
    '''
    random.seed(seed)
    categ1_indices = df[df['loan_status'] == 1].index
    categ1_len = len(df[df['loan_status'] == 1])
    categ2_indices = df[df['loan_status'] == 2].index
    #categ2_len = len(df[df['loan_status'] == 2])
    categ3_indices = df[df['loan_status'] == 3].index
    #categ4_indices = df[df['loan_status'] == 4].index
    
    #random_indices = np.random.choice( categ2_indices, categ2_len - 1000000 , replace=False)
    random_indices = np.random.choice( categ1_indices, categ1_len - 1000000 , replace=False)
    #down_sample_indices = np.concatenate([random_indices, categ1_indices, categ3_indices])
    down_sample_indices = np.concatenate([random_indices, categ2_indices, categ3_indices])
    
    df_downsample = df.loc[down_sample_indices]
    print(df_downsample['loan_status'].value_counts())
    
    return df_downsample


#------------------------------ Data pre-processing ---------------------------------

#----- Split training and test 
    
def data_split(df):
    '''
    Splits the data in a training and testing datasets
    
    :param dataframe df: The dataset
    :return: training independent variables, testing independent variables, training response variable, testing response variable
    
    '''
    print("df type:", type(df))
    y = df['loan_status']
    X = df.drop(['loan_status'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10, stratify = y)
    print("X_train type split:", type(X_train))
    
    return X_train, X_test, y_train, y_test


#--- Nan Imputation for numeric variables and categorical encoding

def get_feature_out(estimator, feature_in):
    '''
    Gets the label of an estimator
    
    :param list estimator: list of estimators
    :param lis feature_in: list of features
    :return: feature label
    
    '''
    if hasattr(estimator,'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f'vec_{f}' \
                for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct):
    '''
    Gets the list of labels of a transformer
    
    :param list ct: The dataset
    :return: list of the variables labels
    
    '''
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name!='remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator=='passthrough':
            output_features.extend(ct._feature_names_in[features])
                
    return output_features
    
def pre_process(X_train, X_test):
    '''
    Inputs Nan values in numeric variables with the median, since categorical variables don't present Nan values.
    Transforms categorical variables using ordinal encoding and one hot encoding. 
    
    :param dataframe X_train: training independent variables
    :param dataframe X_test: testing independent variables
    :return: pre-processed training and testing datasets
    
    '''
    
    #random.seed(seed)
    
    # For SimpleInputer(strategy= 'median') Try KNN
    var1 = ['all_util', 'avg_cur_bal', 'emp_length', 'bc_open_to_buy', 'bc_util', 'dti', 'mths_since_recent_bc', 
            'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'revol_util', 'num_rev_accts', 'inq_last_6mths']
    
    # For SimpleImputer(strategy='constant', fill_value = 0)
    #var2 = ['annual_inc_joint', 'hardship_amount', 'inq_fi', 'inq_last_12m', 
    #        'max_bal_bc', 'open_acc_6m', 'open_act_il', 'open_il_12m', 
    #        'open_il_24m', 'open_rv_12m', 'open_rv_24m']
    
    # For OrdinalEncoder
    var3 = ['sub_grade', 'grade']

    # For OneHotEncoder()
    var4 = ['verification_status', 'purpose', 'initial_list_status', 'hardship_flag',
            'disbursement_method', 'addr_state']
    
    var1_pip = make_pipeline(SimpleImputer(strategy='median'))
    #var2_pip = make_pipeline(SimpleImputer(strategy='constant', fill_value = 0))
    var3_pip = make_pipeline(OrdinalEncoder())
    var4_pip = make_pipeline(OneHotEncoder(drop='if_binary'))
    
    t = [('num1', var1_pip, var1), 
         ('cat1', var3_pip, var3),
         ('cat2', var4_pip, var4)]
    
    combined_pip = ColumnTransformer(transformers = t, remainder='passthrough')
    X_train = combined_pip.fit_transform(X_train)
    X_test = combined_pip.transform(X_test)
    labeling = get_ct_feature_names(combined_pip)
    
    X_train = pd.DataFrame(X_train, columns=labeling)
    X_test = pd.DataFrame(X_test, columns=labeling)
    
    return X_train, X_test
    
# ---------- Normalize data with MinMax 

def trans_minmax(X_train, X_test):
    '''
    Variables of interest (dti, int_rate, fico_range_low, and last_fico_range_low) are normalized using a factor.
    The rest of variables are normalized using MinMax transformation
    
    :param dataframe X_train: training independent variables
    :param dataframe X_test: testing independent variables
    :return: normalized training and testing datasets
    
    '''
    labeling = X_train.columns
    scaler = MinMaxScaler()
    print((labeling != 'dti') & (labeling != 'int_rate') & (labeling != 'fico_range_low') & (labeling != 'last_fico_range_low'))
    
    X_train.loc[:,(labeling != 'dti') & (labeling != 'int_rate') & (labeling != 'fico_range_low') & (labeling != 'last_fico_range_low')] = scaler.fit_transform(X_train.loc[:,(labeling != 'dti') & (labeling != 'int_rate') & (labeling != 'fico_range_low') & (labeling != 'last_fico_range_low')])
    X_test.loc[:,(labeling != 'dti') & (labeling != 'int_rate') & (labeling != 'fico_range_low') & (labeling != 'last_fico_range_low')] = scaler.transform(X_test.loc[:,(labeling != 'dti') & (labeling != 'int_rate') & (labeling != 'fico_range_low') & (labeling != 'last_fico_range_low')])
    
    X_train.dti *= 1/1000
    X_test.dti *= 1/1000
    X_train.int_rate *= 1/100
    X_test.int_rate *= 1/100
    X_train.fico_range_low *= 1/1000
    X_test.fico_range_low *= 1/1000
    X_train.last_fico_range_low *= 1/1000
    X_test.last_fico_range_low *= 1/1000
    
    #X_train_scaled = pd.DataFrame(X_train_scale, columns=labeling)
    #X_test_scaled = pd.DataFrame(X_test_scale, columns=labeling)
    
    return X_train, X_test

# ---------- Feature selection 
# Select X, Q and features to remove due to correlation
    
def kendall_pval(x,y):# Kendall's tau rank
    '''
    Calculates the p-value of the Kendall's tau rank
    
    :param array x: array of values of a variable x
    :param array y: array of values of a variable y
    :return: p-value
    
    '''
    return kendalltau(x,y)[1]
    
def spearman_pval(x,y):# Spearman’s rank
    '''
    Calculates the p-value of the Spearman’s rank
    
    :param array x: array of values of a variable x
    :param array y: array of values of a variable y
    :return: p-value
    
    '''
    return spearmanr(x,y)[1]    


def corr_features_ken(X_train, y_train):
    '''
    Calculates the correlation between independent variables and the response variable using Kendall's tau rank
    
    :param dataframe X_train: independent variables
    :param dataframe y_train: response variable
    :return: p-value
    
    '''
    #corr_pvalue = np.empty(len(X_train.columns))
    corr_pvalue = []
    for i in range(len(X_train.columns)):
        corr_pvalue = corr_pvalue + [[kendalltau(X_train.iloc[:,i].values, y_train)[0],kendalltau(X_train.iloc[:,i].values, y_train)[1]]]
    print(corr_pvalue)
    
    return corr_pvalue

def pvalue_features_ken(X_train):
    '''
    Calculates a matrix of p-values between independent variables using the Kendall's tau rank
    
    :param dataframe X_train: independent variables
    :return: p-value matrix
    
    '''
    pval_matrix = X_train.corr(method= kendall_pval)
    
    return pval_matrix

def corr_features_spear(X_train):
    '''
    Calculates the correlation between independent variables using the Spearman’s rank
    
    :param dataframe X_train: independent variables
    :return: correlation matrix, features with correlation higher than 0.5
    
    '''
    corr_features = []
    corr_matrix = X_train.corr(method= 'spearman')
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) > 0.5 ):
                rowname = corr_matrix.index[i]
                colname = corr_matrix.columns[j]
                corr_features= corr_features + [[rowname, colname, corr_matrix.iloc[i, j]]]
    print(corr_features)
    
    return corr_matrix, corr_features

def pvalue_features_spear(X_train):
    '''
    Calculates p-values between independent variables using the Spearman’s rank
    
    :param dataframe X_train: independent variables
    :return: p-value matrix, features with p-value lower than 0.5
    
    '''
    corr_features_pval = []
    pval_matrix = X_train.corr(method= spearman_pval)
    for i in range(len(pval_matrix.columns)):
        for j in range(i):
            if (abs(pval_matrix.iloc[i, j]) < 0.05 ):
                rowname = pval_matrix.index[i]
                colname = pval_matrix.columns[j]
                corr_features_pval= corr_features_pval + [[rowname, colname, pval_matrix.iloc[i, j]]]
    print(corr_features_pval)
    
    return pval_matrix, corr_features_pval

# Remove correlated features (Spearman’s rank higher than 0.5, with significant p-value aprox. 0)
def remove_corr_features(X_train, X_test):
    '''
    Removes correlated variables based on Spearman's rank
    
    :param dataframe X_train: independent training variables
    :param dataframe X_test: independent testing variables
    :return: independent training variables and independent testing variables with correlated features removed
    
    '''
    
    X_train.drop(['recoveries', 'bc_open_to_buy', 'num_rev_accts', 'open_acc', 
                  'total_rev_hi_lim', 'num_actv_bc_tl', 'num_actv_rev_tl', 'loan_amnt', 
                  'funded_amnt', 'avg_cur_bal', 'annual_inc', 'all_util', 'bc_util', 
                  'percent_bc_gt_75', 'mths_since_recent_bc', 'acc_open_past_24mths', 
                  'sub_grade', 'grade', 'acc_now_delinq', 'last_fico_range_high', 
                  'collection_recovery_fee', 'num_op_rev_tl', 'num_sats', 'num_bc_tl', 
                  'num_rev_tl_bal_gt_0', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 
                  'total_rec_int', 'tot_hi_cred_lim', 'total_bc_limit', 'last_pymnt_amnt', 
                  'mo_sin_rcnt_tl', 'num_tl_op_past_12m', 'total_bal_ex_mort', 
                  'total_il_high_credit_limit', 'mort_acc', 'num_il_tl', 'num_tl_30dpd', 
                  'fico_range_high', 'num_accts_ever_120_pd', 'num_tl_90g_dpd_24m', 
                  'pub_rec_bankruptcies', 'addr_state_Western region'], axis =1, inplace = True)
    
    X_test.drop(['recoveries', 'bc_open_to_buy', 'num_rev_accts', 'open_acc', 
                  'total_rev_hi_lim', 'num_actv_bc_tl', 'num_actv_rev_tl', 'loan_amnt', 
                  'funded_amnt', 'avg_cur_bal', 'annual_inc', 'all_util', 'bc_util', 
                  'percent_bc_gt_75', 'mths_since_recent_bc', 'acc_open_past_24mths', 
                  'sub_grade', 'grade', 'acc_now_delinq', 'last_fico_range_high', 
                  'collection_recovery_fee', 'num_op_rev_tl', 'num_sats', 'num_bc_tl', 
                  'num_rev_tl_bal_gt_0', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 
                  'total_rec_int', 'tot_hi_cred_lim', 'total_bc_limit', 'last_pymnt_amnt', 
                  'mo_sin_rcnt_tl', 'num_tl_op_past_12m', 'total_bal_ex_mort', 
                  'total_il_high_credit_limit', 'mort_acc', 'num_il_tl', 'num_tl_30dpd', 
                  'fico_range_high', 'num_accts_ever_120_pd', 'num_tl_90g_dpd_24m', 
                  'pub_rec_bankruptcies', 'addr_state_Western region'], axis =1, inplace = True)
        
    return X_train, X_test

# Remove features with Kendall's tau not significant (independent)
def remove_kendall(X_train, X_test):
    '''
    Removes non-correlated variables with the response using Kendall's tau
    
    :param dataframe X_train: independent training variables
    :param dataframe X_test: independent testing variables
    :return: independent training variables and independent testing variables without correlation with the response variable
    
    '''
    
    X_train.drop(['purpose_house','purpose_other'], axis =1, inplace = True)
    
    X_test.drop(['purpose_house','purpose_other'], axis =1, inplace = True)    
    
    return X_train, X_test



# Multinomial Logit Model to identify significant variables

def MNLogit_fit(y, X):
    '''
    Fits a multinomial logit model, and prints a summary of the results
    
    :param list y: response variable
    :param dataframe X: independent variables
    :return: multinomial logit model
    
    '''
    """
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    """

    X = st.add_constant(X)
    model = st.MNLogit(y, X)
    model_fit = model.fit(maxiter=35)
    print (model_fit.summary())
    
    return model_fit

# ------------------- Prepare data for L-MNL model input (heuristics)
    
def keras_input(filePath, X_data_scaled, y_data,X_test, y_test, write = True):
    '''
    Generates the input files required to run the L-MNL model, according to three heuristics (see thesis document for details)
    
    :param str filePath: path to save files
    :param dataframe X_data_scaled: independent training variable
    :param dataframe y_data: training response variable
    :param dataframe X_test: independent testing variable
    :param dataframe y_test: testing response variable
    :param bool write: if true files are saved
    :return: knwoledge-driven data files and data-driven files according to the heuristis, and files' paths 
    
    '''
    
    data_name1 = filePath+ 'input_H1_v2' + '.npy'
    data_name2 = filePath+ 'input_H2_v2'  + '.npy'
    data_name3 = filePath+ 'input_H3_v2'  + '.npy'
    
    test_name1 = filePath+ 'input_H1_test_v2' + '.npy'
    test_name2 = filePath+ 'input_H2_test_v2' + '.npy'
    test_name3 = filePath+ 'input_H3_test_v2' + '.npy'
    
    dataset = pd.concat([X_data_scaled, y_data], axis=1)
    dataset_test = pd.concat([X_test, y_test], axis=1)
    
    ASCs = np.ones(dataset['loan_status'].size)
    ZEROs = np.zeros(dataset['loan_status'].size)
    ASCs_t = np.ones(dataset_test['loan_status'].size)
    ZEROs_t = np.zeros(dataset_test['loan_status'].size)
    
    Categ3 = (dataset['loan_status'] == 3)
    Categ2 = (dataset['loan_status'] == 2)
    Categ1 = (dataset['loan_status'] == 1)
    Categ3_t = (dataset_test['loan_status'] == 3)
    Categ2_t = (dataset_test['loan_status'] == 2)
    Categ1_t = (dataset_test['loan_status'] == 1)
    
    # Variables
    # Train data
    emp_length = dataset['emp_length'].values
    dti = dataset['dti'].values
    pct_tl_nvr_dlq = dataset['pct_tl_nvr_dlq'].values
    revol_util = dataset['revol_util'].values
    inq_last_6mths = dataset['inq_last_6mths'].values
    verification_status_Verified = dataset['verification_status_Verified'].values
    purpose_car = dataset['purpose_car'].values
    purpose_debt_consolidation = dataset['purpose_debt_consolidation'].values
    purpose_small_business = dataset['purpose_small_business'].values
    initial_list_status_w = dataset['initial_list_status_w'].values
    hardship_flag_Y = dataset['hardship_flag_Y'].values
    disbursement_method_DirectPay = dataset['disbursement_method_DirectPay'].values
    term = dataset['term'].values
    int_rate = dataset['int_rate'].values
    installment = dataset['installment'].values
    fico_range_low = dataset['fico_range_low'].values
    pub_rec = dataset['pub_rec'].values
    revol_bal = dataset['revol_bal'].values
    total_acc = dataset['total_acc'].values
    out_prncp = dataset['out_prncp'].values
    total_rec_late_fee = dataset['total_rec_late_fee'].values
    last_fico_range_low = dataset['last_fico_range_low'].values
    collections_12_mths_ex_med = dataset['collections_12_mths_ex_med'].values
    tot_cur_bal = dataset['tot_cur_bal'].values
    chargeoff_within_12_mths = dataset['chargeoff_within_12_mths'].values
    mo_sin_rcnt_rev_tl_op = dataset['mo_sin_rcnt_rev_tl_op'].values
    num_bc_sats = dataset['num_bc_sats'].values
    tax_liens = dataset['tax_liens'].values
    # Test data
    emp_length_t = dataset_test['emp_length'].values
    dti_t = dataset_test['dti'].values
    pct_tl_nvr_dlq_t = dataset_test['pct_tl_nvr_dlq'].values
    revol_util_t = dataset_test['revol_util'].values
    inq_last_6mths_t = dataset_test['inq_last_6mths'].values
    verification_status_Verified_t = dataset_test['verification_status_Verified'].values
    purpose_car_t = dataset_test['purpose_car'].values
    purpose_debt_consolidation_t = dataset_test['purpose_debt_consolidation'].values
    purpose_small_business_t = dataset_test['purpose_small_business'].values
    initial_list_status_w_t = dataset_test['initial_list_status_w'].values
    hardship_flag_Y_t = dataset_test['hardship_flag_Y'].values
    disbursement_method_DirectPay_t = dataset_test['disbursement_method_DirectPay'].values
    term_t = dataset_test['term'].values
    int_rate_t = dataset_test['int_rate'].values
    installment_t = dataset_test['installment'].values
    fico_range_low_t = dataset_test['fico_range_low'].values
    pub_rec_t = dataset_test['pub_rec'].values
    revol_bal_t = dataset_test['revol_bal'].values
    total_acc_t = dataset_test['total_acc'].values
    out_prncp_t = dataset_test['out_prncp'].values
    total_rec_late_fee_t = dataset_test['total_rec_late_fee'].values
    last_fico_range_low_t = dataset_test['last_fico_range_low'].values
    collections_12_mths_ex_med_t = dataset_test['collections_12_mths_ex_med'].values
    tot_cur_bal_t = dataset_test['tot_cur_bal'].values
    chargeoff_within_12_mths_t = dataset_test['chargeoff_within_12_mths'].values
    mo_sin_rcnt_rev_tl_op_t = dataset_test['mo_sin_rcnt_rev_tl_op'].values
    num_bc_sats_t = dataset_test['num_bc_sats'].values
    tax_liens_t = dataset_test['tax_liens'].values
    
    # Heuristics
    
    # Heuristic 1

    data1 = np.array(
        [[ZEROs, ZEROs,      ZEROs,      ZEROs, ZEROs, ZEROs,      ZEROs,     ZEROs,      ZEROs,      ZEROs, 
                  ZEROs, ZEROs,                        ZEROs, ZEROs,       ZEROs, ZEROs,                      ZEROs, ZEROs, 
                          ZEROs, ZEROs,                 ZEROs, ZEROs,           ZEROs, ZEROs,                         ZEROs, ZEROs, 
         ZEROs, ZEROs,    ZEROs, ZEROs,       ZEROs, ZEROs,          ZEROs, ZEROs,   ZEROs, ZEROs,     ZEROs, ZEROs, 
             ZEROs, ZEROs,     ZEROs, ZEROs,              ZEROs, ZEROs,               ZEROs, ZEROs,                      ZEROs, ZEROs, 
               ZEROs, ZEROs,                    ZEROs, ZEROs,                 ZEROs, ZEROs,       ZEROs, ZEROs, 
             ZEROs, ZEROs, Categ1],
        [  ASCs, ZEROs, emp_length,      ZEROs,   dti, ZEROs, pct_tl_nvr_dlq, ZEROs, revol_util,      ZEROs, 
         inq_last_6mths, ZEROs, verification_status_Verified, ZEROs, purpose_car, ZEROs, purpose_debt_consolidation, ZEROs, 
         purpose_small_business, ZEROs, initial_list_status_w, ZEROs, hardship_flag_Y, ZEROs, disbursement_method_DirectPay, ZEROs, 
         term,  ZEROs, int_rate, ZEROs, installment, ZEROs, fico_range_low, ZEROs, pub_rec, ZEROs, revol_bal, ZEROs, 
         total_acc, ZEROs, out_prncp, ZEROs, total_rec_late_fee, ZEROs, last_fico_range_low, ZEROs, collections_12_mths_ex_med, ZEROs, 
         tot_cur_bal, ZEROs, chargeoff_within_12_mths, ZEROs, mo_sin_rcnt_rev_tl_op, ZEROs, num_bc_sats, ZEROs, 
         tax_liens, ZEROs, Categ2],
        [ ZEROs,  ASCs,      ZEROs, emp_length, ZEROs,   dti, ZEROs, pct_tl_nvr_dlq,      ZEROs, revol_util, 
         ZEROs, inq_last_6mths, ZEROs, verification_status_Verified, ZEROs, purpose_car, ZEROs, purpose_debt_consolidation, 
         ZEROs, purpose_small_business, ZEROs, initial_list_status_w, ZEROs, hardship_flag_Y, ZEROs, disbursement_method_DirectPay, 
         ZEROs,  term, ZEROs, int_rate, ZEROs, installment, ZEROs, fico_range_low, ZEROs, pub_rec, ZEROs, revol_bal, 
         ZEROs, total_acc, ZEROs, out_prncp, ZEROs, total_rec_late_fee, ZEROs, last_fico_range_low, ZEROs, collections_12_mths_ex_med, 
         ZEROs, tot_cur_bal, ZEROs, chargeoff_within_12_mths, ZEROs, mo_sin_rcnt_rev_tl_op, ZEROs, num_bc_sats, 
         ZEROs, tax_liens, Categ3]])
    
    data1_test = np.array(
        [[ZEROs_t, ZEROs_t,      ZEROs_t,      ZEROs_t, ZEROs_t, ZEROs_t,      ZEROs_t,     ZEROs_t,      ZEROs_t,      ZEROs_t, 
                  ZEROs_t, ZEROs_t,                        ZEROs_t, ZEROs_t,       ZEROs_t, ZEROs_t,                      ZEROs_t, ZEROs_t, 
                          ZEROs_t, ZEROs_t,                 ZEROs_t, ZEROs_t,           ZEROs_t, ZEROs_t,                         ZEROs_t, ZEROs_t, 
         ZEROs_t, ZEROs_t,    ZEROs_t, ZEROs_t,       ZEROs_t, ZEROs_t,          ZEROs_t, ZEROs_t,   ZEROs_t, ZEROs_t,     ZEROs_t, ZEROs_t, 
             ZEROs_t, ZEROs_t,     ZEROs_t, ZEROs_t,              ZEROs_t, ZEROs_t,               ZEROs_t, ZEROs_t,                      ZEROs_t, ZEROs_t, 
               ZEROs_t, ZEROs_t,                    ZEROs_t, ZEROs_t,                 ZEROs_t, ZEROs_t,       ZEROs_t, ZEROs_t, 
             ZEROs_t, ZEROs_t, Categ1_t],
        [  ASCs_t, ZEROs_t, emp_length_t,      ZEROs_t,   dti_t, ZEROs_t, pct_tl_nvr_dlq_t, ZEROs_t, revol_util_t,      ZEROs_t, 
         inq_last_6mths_t, ZEROs_t, verification_status_Verified_t, ZEROs_t, purpose_car_t, ZEROs_t, purpose_debt_consolidation_t, ZEROs_t, 
         purpose_small_business_t, ZEROs_t, initial_list_status_w_t, ZEROs_t, hardship_flag_Y_t, ZEROs_t, disbursement_method_DirectPay_t, ZEROs_t, 
         term_t,  ZEROs_t, int_rate_t, ZEROs_t, installment_t, ZEROs_t, fico_range_low_t, ZEROs_t, pub_rec_t, ZEROs_t, revol_bal_t, ZEROs_t, 
         total_acc_t, ZEROs_t, out_prncp_t, ZEROs_t, total_rec_late_fee_t, ZEROs_t, last_fico_range_low_t, ZEROs_t, collections_12_mths_ex_med_t, ZEROs_t, 
         tot_cur_bal_t, ZEROs_t, chargeoff_within_12_mths_t, ZEROs_t, mo_sin_rcnt_rev_tl_op_t, ZEROs_t, num_bc_sats_t, ZEROs_t, 
         tax_liens_t, ZEROs_t, Categ2_t],
        [ ZEROs_t,  ASCs_t,      ZEROs_t, emp_length_t, ZEROs_t,   dti_t, ZEROs_t, pct_tl_nvr_dlq_t,      ZEROs_t, revol_util_t, 
         ZEROs_t, inq_last_6mths_t, ZEROs_t, verification_status_Verified_t, ZEROs_t, purpose_car_t, ZEROs_t, purpose_debt_consolidation_t, 
         ZEROs_t, purpose_small_business_t, ZEROs_t, initial_list_status_w_t, ZEROs_t, hardship_flag_Y_t, ZEROs_t, disbursement_method_DirectPay_t, 
         ZEROs_t,  term_t, ZEROs_t, int_rate_t, ZEROs_t, installment_t, ZEROs_t, fico_range_low_t, ZEROs_t, pub_rec_t, ZEROs_t, revol_bal_t, 
         ZEROs_t, total_acc_t, ZEROs_t, out_prncp_t, ZEROs_t, total_rec_late_fee_t, ZEROs_t, last_fico_range_low_t, ZEROs_t, collections_12_mths_ex_med_t, 
         ZEROs_t, tot_cur_bal_t, ZEROs_t, chargeoff_within_12_mths_t, ZEROs_t, mo_sin_rcnt_rev_tl_op_t, ZEROs_t, num_bc_sats_t, 
         ZEROs_t, tax_liens_t, Categ3_t]])
    
    # Heuristic 2
    data2 = np.array(
        [[ZEROs, ZEROs, ZEROs, ZEROs,          ZEROs, ZEROs,                        ZEROs, ZEROs, ZEROs, ZEROs,    ZEROs, ZEROs,          ZEROs, ZEROs, 
             ZEROs, ZEROs,              ZEROs, ZEROs,               ZEROs, ZEROs, Categ1],
        [ ASCs, ZEROs,    dti, ZEROs, inq_last_6mths, ZEROs, verification_status_Verified, ZEROs,  term, ZEROs, int_rate, ZEROs, fico_range_low, ZEROs, 
         out_prncp, ZEROs, total_rec_late_fee, ZEROs, last_fico_range_low, ZEROs, Categ2],
        [ ZEROs, ASCs, ZEROs,   dti, ZEROs, inq_last_6mths, ZEROs, verification_status_Verified, ZEROs,  term, ZEROs, int_rate, ZEROs, fico_range_low, 
         ZEROs, out_prncp, ZEROs, total_rec_late_fee, ZEROs, last_fico_range_low, Categ3]] )
    
    data2_test = np.array(
        [[ZEROs_t, ZEROs_t, ZEROs_t, ZEROs_t,          ZEROs_t, ZEROs_t,                        ZEROs_t, ZEROs_t, ZEROs_t, ZEROs_t,    ZEROs_t, ZEROs_t,          ZEROs_t, ZEROs_t, 
             ZEROs_t, ZEROs_t,              ZEROs_t, ZEROs_t,               ZEROs_t, ZEROs_t, Categ1_t],
        [ ASCs_t, ZEROs_t,    dti_t, ZEROs_t, inq_last_6mths_t, ZEROs_t, verification_status_Verified_t, ZEROs_t,  term_t, ZEROs_t, int_rate_t, ZEROs_t, fico_range_low_t, ZEROs_t, 
         out_prncp_t, ZEROs_t, total_rec_late_fee_t, ZEROs_t, last_fico_range_low_t, ZEROs_t, Categ2_t],
        [ ZEROs_t, ASCs_t,  ZEROs_t,   dti_t, ZEROs_t, inq_last_6mths_t, ZEROs_t, verification_status_Verified_t, ZEROs_t,  term_t, ZEROs_t, int_rate_t, ZEROs_t, fico_range_low_t, 
         ZEROs_t, out_prncp_t, ZEROs_t, total_rec_late_fee_t, ZEROs_t, last_fico_range_low_t, Categ3_t]] )
       
    # Heuristic 3
    data3 = np.array(
        [[ZEROs, ZEROs, ZEROs, ZEROs,    ZEROs, ZEROs,          ZEROs, ZEROs,               ZEROs, ZEROs, Categ1],
        [  ASCs, ZEROs,   dti, ZEROs, int_rate, ZEROs, fico_range_low, ZEROs, last_fico_range_low, ZEROs, Categ2],
        [ ZEROs,  ASCs, ZEROs,   dti, ZEROs, int_rate, ZEROs, fico_range_low, ZEROs, last_fico_range_low, Categ3]] )
    
    data3_test = np.array(
        [[ZEROs_t, ZEROs_t, ZEROs_t, ZEROs_t,    ZEROs_t, ZEROs_t,          ZEROs_t, ZEROs_t,               ZEROs_t, ZEROs_t, Categ1_t],
        [  ASCs_t, ZEROs_t,   dti_t, ZEROs_t, int_rate_t, ZEROs_t, fico_range_low_t, ZEROs_t, last_fico_range_low_t, ZEROs_t, Categ2_t],
        [ ZEROs_t,  ASCs_t, ZEROs_t,   dti_t, ZEROs_t, int_rate_t, ZEROs_t, fico_range_low_t, ZEROs_t, last_fico_range_low_t, Categ3_t]] )
    
    
    data1 = np.swapaxes(data1,0,2)
    data2 = np.swapaxes(data2,0,2)
    data3 = np.swapaxes(data3,0,2)
    data1_test = np.swapaxes(data1_test,0,2)
    data2_test = np.swapaxes(data2_test,0,2)
    data3_test = np.swapaxes(data3_test,0,2)
    
    H1_Q = ['purpose_home_improvement', 'purpose_major_purchase', 'addr_state_Midwest region', 'addr_state_Northeast region',
        'addr_state_Southern region', 'delinq_2yrs', 'tot_coll_amt']
    
    H2_Q = ['emp_length', 'pct_tl_nvr_dlq', 'revol_util', 'purpose_car', 'purpose_debt_consolidation', 'purpose_home_improvement',
            'purpose_major_purchase', 'purpose_small_business', 'initial_list_status_w', 'hardship_flag_Y', 'disbursement_method_DirectPay',
            'addr_state_Midwest region', 'addr_state_Northeast region', 'addr_state_Southern region',
            'installment', 'delinq_2yrs', 'pub_rec', 'revol_bal', 'total_acc', 'collections_12_mths_ex_med', 'tot_coll_amt', 
            'tot_cur_bal', 'chargeoff_within_12_mths', 'mo_sin_rcnt_rev_tl_op', 'num_bc_sats', 'tax_liens']
    
    H3_Q = ['emp_length', 'pct_tl_nvr_dlq', 'revol_util', 'inq_last_6mths', 'verification_status_Verified', 'purpose_car', 'purpose_debt_consolidation', 'purpose_home_improvement',
            'purpose_major_purchase', 'purpose_small_business', 'initial_list_status_w', 'hardship_flag_Y', 'disbursement_method_DirectPay',
            'addr_state_Midwest region', 'addr_state_Northeast region', 'addr_state_Southern region', 'term',
            'installment', 'delinq_2yrs', 'pub_rec', 'revol_bal', 'total_acc', 'out_prncp', 'total_rec_late_fee', 'collections_12_mths_ex_med', 'tot_coll_amt', 
            'tot_cur_bal', 'chargeoff_within_12_mths', 'mo_sin_rcnt_rev_tl_op', 'num_bc_sats', 'tax_liens']
    
    Q1 = dataset[H1_Q].values
    Q2 = dataset[H2_Q].values
    Q3 = dataset[H3_Q].values
    Q1_test = dataset_test[H1_Q].values
    Q2_test = dataset_test[H2_Q].values
    Q3_test = dataset_test[H3_Q].values
    
    if write:
        np.save(data_name1, np.array(data1, dtype=np.float32))
        np.save(data_name1[:-4] + '_extra.npy', Q1)
        np.save(data_name2, np.array(data2, dtype=np.float32))
        np.save(data_name2[:-4] + '_extra.npy', Q2)
        np.save(data_name3, np.array(data3, dtype=np.float32))
        np.save(data_name3[:-4] + '_extra.npy', Q3)
        
        np.save(test_name1, np.array(data1_test, dtype=np.float32))
        np.save(test_name1[:-4] + '_extra.npy', Q1_test)
        np.save(test_name2, np.array(data2_test, dtype=np.float32))
        np.save(test_name2[:-4] + '_extra.npy', Q2_test)
        np.save(test_name3, np.array(data3_test, dtype=np.float32))
        np.save(test_name3[:-4] + '_extra.npy', Q3_test)
        
    return data1, data2, data3, Q1, Q2, Q3, data_name1, data_name2, data_name3, test_name1, test_name2, test_name3



# ------------------- Prepare data for MNL as a CNN model input
def keras_inputMNL(filePath, X_data_scaled, y_data,X_test, y_test, write = True):
    '''
    Generates the input file required to run the MNL as a CNN model
    
    :param str filePath: path to save files
    :param dataframe X_data_scaled: independent training variable
    :param dataframe y_data: training response variable
    :param dataframe X_test: independent testing variable
    :param dataframe y_test: testing response variable
    :param bool write: if true files are saved
    :return: knwoledge-driven data files, and files' paths 
    
    '''
    
    data_name5 = filePath+ 'input_MNL_v2' + '.npy'
    
    test_name5 = filePath+ 'input_MNL_test_v2' + '.npy'
    
    dataset = pd.concat([X_data_scaled, y_data], axis=1)
    dataset_test = pd.concat([X_test, y_test], axis=1)
    
    ASCs = np.ones(dataset['loan_status'].size)
    ZEROs = np.zeros(dataset['loan_status'].size)
    ASCs_t = np.ones(dataset_test['loan_status'].size)
    ZEROs_t = np.zeros(dataset_test['loan_status'].size)
    
    Categ3 = (dataset['loan_status'] == 3)
    Categ2 = (dataset['loan_status'] == 2)
    Categ1 = (dataset['loan_status'] == 1)
    Categ3_t = (dataset_test['loan_status'] == 3)
    Categ2_t = (dataset_test['loan_status'] == 2)
    Categ1_t = (dataset_test['loan_status'] == 1)
    
    # Variables (all)
    # Train data
    
    emp_length = dataset['emp_length'].values
    dti = dataset['dti'].values
    pct_tl_nvr_dlq = dataset['pct_tl_nvr_dlq'].values
    revol_util = dataset['revol_util'].values
    inq_last_6mths = dataset['inq_last_6mths'].values
    verification_status_Verified = dataset['verification_status_Verified'].values
    purpose_car = dataset['purpose_car'].values
    purpose_debt_consolidation = dataset['purpose_debt_consolidation'].values
    purpose_home_improvement = dataset['purpose_home_improvement'].values
    purpose_major_purchase = dataset['purpose_major_purchase'].values
    purpose_small_business = dataset['purpose_small_business'].values
    initial_list_status_w = dataset['initial_list_status_w'].values
    hardship_flag_Y = dataset['hardship_flag_Y'].values
    disbursement_method_DirectPay = dataset['disbursement_method_DirectPay'].values
    addr_state_Midwest_region = dataset['addr_state_Midwest region'].values
    addr_state_Northeast_region = dataset['addr_state_Northeast region'].values
    addr_state_Southern_region = dataset['addr_state_Southern region'].values
    term = dataset['term'].values
    int_rate = dataset['int_rate'].values
    installment = dataset['installment'].values
    delinq_2yrs = dataset['delinq_2yrs'].values
    fico_range_low = dataset['fico_range_low'].values
    pub_rec = dataset['pub_rec'].values
    revol_bal = dataset['revol_bal'].values
    total_acc = dataset['total_acc'].values
    out_prncp = dataset['out_prncp'].values
    total_rec_late_fee = dataset['total_rec_late_fee'].values
    last_fico_range_low = dataset['last_fico_range_low'].values
    collections_12_mths_ex_med = dataset['collections_12_mths_ex_med'].values
    tot_coll_amt = dataset['tot_coll_amt'].values
    tot_cur_bal = dataset['tot_cur_bal'].values
    chargeoff_within_12_mths = dataset['chargeoff_within_12_mths'].values
    mo_sin_rcnt_rev_tl_op = dataset['mo_sin_rcnt_rev_tl_op'].values
    num_bc_sats = dataset['num_bc_sats'].values
    tax_liens = dataset['tax_liens'].values
    # Test data
    emp_length_t = dataset_test['emp_length'].values
    dti_t = dataset_test['dti'].values
    pct_tl_nvr_dlq_t = dataset_test['pct_tl_nvr_dlq'].values
    revol_util_t = dataset_test['revol_util'].values
    inq_last_6mths_t = dataset_test['inq_last_6mths'].values
    verification_status_Verified_t = dataset_test['verification_status_Verified'].values
    purpose_car_t = dataset_test['purpose_car'].values
    purpose_debt_consolidation_t = dataset_test['purpose_debt_consolidation'].values
    purpose_home_improvement_t = dataset_test['purpose_home_improvement'].values
    purpose_major_purchase_t = dataset_test['purpose_major_purchase'].values
    purpose_small_business_t = dataset_test['purpose_small_business'].values
    initial_list_status_w_t = dataset_test['initial_list_status_w'].values
    hardship_flag_Y_t = dataset_test['hardship_flag_Y'].values
    disbursement_method_DirectPay_t = dataset_test['disbursement_method_DirectPay'].values
    addr_state_Midwest_region_t = dataset_test['addr_state_Midwest region'].values
    addr_state_Northeast_region_t = dataset_test['addr_state_Northeast region'].values
    addr_state_Southern_region_t = dataset_test['addr_state_Southern region'].values
    term_t = dataset_test['term'].values
    int_rate_t = dataset_test['int_rate'].values
    installment_t = dataset_test['installment'].values
    delinq_2yrs_t = dataset_test['delinq_2yrs'].values
    fico_range_low_t = dataset_test['fico_range_low'].values
    pub_rec_t = dataset_test['pub_rec'].values
    revol_bal_t = dataset_test['revol_bal'].values
    total_acc_t = dataset_test['total_acc'].values
    out_prncp_t = dataset_test['out_prncp'].values
    total_rec_late_fee_t = dataset_test['total_rec_late_fee'].values
    last_fico_range_low_t = dataset_test['last_fico_range_low'].values
    collections_12_mths_ex_med_t = dataset_test['collections_12_mths_ex_med'].values
    tot_coll_amt_t = dataset_test['tot_coll_amt'].values
    tot_cur_bal_t = dataset_test['tot_cur_bal'].values
    chargeoff_within_12_mths_t = dataset_test['chargeoff_within_12_mths'].values
    mo_sin_rcnt_rev_tl_op_t = dataset_test['mo_sin_rcnt_rev_tl_op'].values
    num_bc_sats_t = dataset_test['num_bc_sats'].values
    tax_liens_t = dataset_test['tax_liens'].values
    
    # Heuristics    
    
    data5 = np.array(
        [[ZEROs, ZEROs,      ZEROs, ZEROs, ZEROs, ZEROs,          ZEROs, ZEROs,      ZEROs, ZEROs,          ZEROs, ZEROs, 
                                 ZEROs, ZEROs,       ZEROs, ZEROs,                      ZEROs, ZEROs,                    ZEROs, ZEROs,                  ZEROs, ZEROs, 
                           ZEROs, ZEROs,                 ZEROs, ZEROs,           ZEROs, ZEROs,                         ZEROs, ZEROs, 
                              ZEROs, ZEROs,                       ZEROs, ZEROs,                      ZEROs, ZEROs, 
         ZEROs, ZEROs,    ZEROs, ZEROs,       ZEROs, ZEROs,       ZEROs, ZEROs,          ZEROs, ZEROs,   ZEROs, ZEROs,     ZEROs, ZEROs,     ZEROs, ZEROs,     ZEROs, ZEROs,              ZEROs, ZEROs, 
                        ZEROs, ZEROs,                      ZEROs, ZEROs,        ZEROs, ZEROs,       ZEROs, ZEROs,                    ZEROs, ZEROs,                 ZEROs, ZEROs,       ZEROs, ZEROs, 
              ZEROs, ZEROs, Categ1],
        [  ASCs, ZEROs, emp_length, ZEROs,   dti, ZEROs, pct_tl_nvr_dlq, ZEROs, revol_util, ZEROs, inq_last_6mths, ZEROs, 
          verification_status_Verified, ZEROs, purpose_car, ZEROs, purpose_debt_consolidation, ZEROs, purpose_home_improvement, ZEROs, purpose_major_purchase, ZEROs, 
          purpose_small_business, ZEROs, initial_list_status_w, ZEROs, hardship_flag_Y, ZEROs, disbursement_method_DirectPay, ZEROs, 
          addr_state_Midwest_region, ZEROs, addr_state_Northeast_region, ZEROs, addr_state_Southern_region, ZEROs,
          term, ZEROs, int_rate, ZEROs, installment, ZEROs, delinq_2yrs, ZEROs, fico_range_low, ZEROs, pub_rec, ZEROs, revol_bal, ZEROs, total_acc, ZEROs, out_prncp, ZEROs, total_rec_late_fee, ZEROs, 
          last_fico_range_low, ZEROs, collections_12_mths_ex_med, ZEROs, tot_coll_amt, ZEROs, tot_cur_bal, ZEROs, chargeoff_within_12_mths, ZEROs, mo_sin_rcnt_rev_tl_op, ZEROs, num_bc_sats, ZEROs,
          tax_liens, ZEROs, Categ2],
        [ ZEROs,  ASCs, ZEROs, emp_length, ZEROs,   dti, ZEROs, pct_tl_nvr_dlq, ZEROs, revol_util, ZEROs, inq_last_6mths, 
         ZEROs, verification_status_Verified, ZEROs, purpose_car, ZEROs, purpose_debt_consolidation, ZEROs, purpose_home_improvement, ZEROs, purpose_major_purchase, 
         ZEROs, purpose_small_business, ZEROs, initial_list_status_w, ZEROs, hardship_flag_Y, ZEROs, disbursement_method_DirectPay, 
         ZEROs, addr_state_Midwest_region, ZEROs, addr_state_Northeast_region, ZEROs, addr_state_Southern_region, 
         ZEROs, term, ZEROs, int_rate, ZEROs, installment, ZEROs, delinq_2yrs, ZEROs, fico_range_low, ZEROs, pub_rec, ZEROs, revol_bal, ZEROs, total_acc, ZEROs, out_prncp, ZEROs, total_rec_late_fee, 
         ZEROs, last_fico_range_low, ZEROs, collections_12_mths_ex_med, ZEROs, tot_coll_amt, ZEROs, tot_cur_bal, ZEROs, chargeoff_within_12_mths, ZEROs, mo_sin_rcnt_rev_tl_op, ZEROs, num_bc_sats, 
         ZEROs, tax_liens, Categ3]] )
    
    data5_test = np.array(
        [[ZEROs_t, ZEROs_t,      ZEROs_t, ZEROs_t, ZEROs_t, ZEROs_t,          ZEROs_t, ZEROs_t,      ZEROs_t, ZEROs_t,          ZEROs_t, ZEROs_t, 
                                 ZEROs_t, ZEROs_t,       ZEROs_t, ZEROs_t,                      ZEROs_t, ZEROs_t,                    ZEROs_t, ZEROs_t,                  ZEROs_t, ZEROs_t, 
                           ZEROs_t, ZEROs_t,                 ZEROs_t, ZEROs_t,           ZEROs_t, ZEROs_t,                         ZEROs_t, ZEROs_t, 
                              ZEROs_t, ZEROs_t,                       ZEROs_t, ZEROs_t,                      ZEROs_t, ZEROs_t, 
         ZEROs_t, ZEROs_t,    ZEROs_t, ZEROs_t,       ZEROs_t, ZEROs_t,       ZEROs_t, ZEROs_t,          ZEROs_t, ZEROs_t,   ZEROs_t, ZEROs_t,     ZEROs_t, ZEROs_t,     ZEROs_t, ZEROs_t,     ZEROs_t, ZEROs_t,              ZEROs_t, ZEROs_t, 
                        ZEROs_t, ZEROs_t,                      ZEROs_t, ZEROs_t,        ZEROs_t, ZEROs_t,       ZEROs_t, ZEROs_t,                    ZEROs_t, ZEROs_t,                 ZEROs_t, ZEROs_t,       ZEROs_t, ZEROs_t, 
              ZEROs_t, ZEROs_t, Categ1_t],
        [  ASCs_t, ZEROs_t, emp_length_t, ZEROs_t,   dti_t, ZEROs_t, pct_tl_nvr_dlq_t, ZEROs_t, revol_util_t, ZEROs_t, inq_last_6mths_t, ZEROs_t, 
          verification_status_Verified_t, ZEROs_t, purpose_car_t, ZEROs_t, purpose_debt_consolidation_t, ZEROs_t, purpose_home_improvement_t, ZEROs_t, purpose_major_purchase_t, ZEROs_t, 
          purpose_small_business_t, ZEROs_t, initial_list_status_w_t, ZEROs_t, hardship_flag_Y_t, ZEROs_t, disbursement_method_DirectPay_t, ZEROs_t, 
          addr_state_Midwest_region_t, ZEROs_t, addr_state_Northeast_region_t, ZEROs_t, addr_state_Southern_region_t, ZEROs_t,
          term_t, ZEROs_t, int_rate_t, ZEROs_t, installment_t, ZEROs_t, delinq_2yrs_t, ZEROs_t, fico_range_low_t, ZEROs_t, pub_rec_t, ZEROs_t, revol_bal_t, ZEROs_t, total_acc_t, ZEROs_t, out_prncp_t, ZEROs_t, total_rec_late_fee_t, ZEROs_t, 
          last_fico_range_low_t, ZEROs_t, collections_12_mths_ex_med_t, ZEROs_t, tot_coll_amt_t, ZEROs_t, tot_cur_bal_t, ZEROs_t, chargeoff_within_12_mths_t, ZEROs_t, mo_sin_rcnt_rev_tl_op_t, ZEROs_t, num_bc_sats_t, ZEROs_t,
          tax_liens_t, ZEROs_t, Categ2_t],
        [ ZEROs_t,  ASCs_t, ZEROs_t, emp_length_t, ZEROs_t,   dti_t, ZEROs_t, pct_tl_nvr_dlq_t, ZEROs_t, revol_util_t, ZEROs_t, inq_last_6mths_t, 
         ZEROs_t, verification_status_Verified_t, ZEROs_t, purpose_car_t, ZEROs_t, purpose_debt_consolidation_t, ZEROs_t, purpose_home_improvement_t, ZEROs_t, purpose_major_purchase_t, 
         ZEROs_t, purpose_small_business_t, ZEROs_t, initial_list_status_w_t, ZEROs_t, hardship_flag_Y_t, ZEROs_t, disbursement_method_DirectPay_t, 
         ZEROs_t, addr_state_Midwest_region_t, ZEROs_t, addr_state_Northeast_region_t, ZEROs_t, addr_state_Southern_region_t, 
         ZEROs_t, term_t, ZEROs_t, int_rate_t, ZEROs_t, installment_t, ZEROs_t, delinq_2yrs_t, ZEROs_t, fico_range_low_t, ZEROs_t, pub_rec_t, ZEROs_t, revol_bal_t, ZEROs_t, total_acc_t, ZEROs_t, out_prncp_t, ZEROs_t, total_rec_late_fee_t, 
         ZEROs_t, last_fico_range_low_t, ZEROs_t, collections_12_mths_ex_med_t, ZEROs_t, tot_coll_amt_t, ZEROs_t, tot_cur_bal_t, ZEROs_t, chargeoff_within_12_mths_t, ZEROs_t, mo_sin_rcnt_rev_tl_op_t, ZEROs_t, num_bc_sats_t, 
         ZEROs_t, tax_liens_t, Categ3_t]] )
    
    data5 = np.swapaxes(data5,0,2)
    data5_test = np.swapaxes(data5_test,0,2)
    
    
    if write:
        np.save(data_name5, np.array(data5, dtype=np.float32))
        
        np.save(test_name5, np.array(data5_test, dtype=np.float32))

        
    return data5, data_name5, test_name5

# ----------------- DNN model ----------------------------
def dnn(hidden = 1):
    '''
    Creates a dense neural network (DNN) with dropout of 0.2 between layers, relu activation function in hidden layers, and softmax activation function in the output
    
    :param int hidden: number of hidden layers (1 or 2) #need to add the handling of other values
    :return: DNN model
    
    '''
    model = Sequential()
    model.add(Dense(40, input_dim=35, activation='relu'))
    model.add(Dropout(0.2))
    if hidden == 2:
        model.add(Dense(40, activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', metrics=['categorical_accuracy'], loss='categorical_crossentropy')
    
    return model

# ------------------------- Training: 5-Fold Straitied Cross-Validation ---------------------------
def crossval(data, target, y, beta_num, choices_num, nExtraFeatures , extra_data, networkSize = 40, modelname = 'No name', LMNL = True, DNN = False, MNL1 = False, 
             MNL2 = False, OrdLogit = False, MNL_CNN = False, Ontr_SI_LS = False, 
             nEpochs = 50, seed = 10, batchSize = 32, hidden = 1):
    '''
    Trains a model with a 5-Fold Stratified Cross-Validation
    
    :param dataframe data: training dataset (independent variables)
    :param dataframe target: training dataset (response variable)
    :param array y: array of taget labels (nx1 dimension) for doing the stratification
    :param int beta_num: number of betas (coefficients) for the L-MNL model (knowledge-driven part (MNL))
    :param int choices_num: number of choices or classes
    :param int nExtraFeatures: number of variables in the data-driven part (DNN) of the L-MNL model
    :param dataframe extra_data: training dataset for the data-driven part (DNN) of the L-MNL model
    :param int networkSize: number of neurons for the data-driven part (DNN) of the L-MNL model
    :param str modelname: name of the model
    :param bool LMNL: TRUE if the model that will be run is the L-MNL
    :param bool DNN: TRUE if the model that will be run is the DNN
    :param bool MNL1: TRUE if the model that will be run is the MNL (sklearn implementation, lacks p-values)
    :param bool MNL2: TRUE if the model that will be run is the MNL (statsmodel)
    :param bool OrdLogit: TRUE if the model that will be run is the Ordistic Logit All Threshold from mord, equivalent to the Cumulative logit model
    :param bool MNL_CNN: TRUE if the model that will be run is the MNL as a CNN
    :param bool Ontr_SI_LS: TRUE if the model that will be run is the Ontram single intercept linear shift, this is equivalent to the Cum Ordinal Logit (REMOVED) 
    :param int nEpochs: number of training epochs
    :param int seed: random seed
    :param int batchSize: bacth size for training
    :param int hidden: number of hidden layers
    :return: performance measures (AUC, accuracy, F1 score, MAE, and BIC) results
    
    '''
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) #Comapre with 10 (time)
    cvscores = []
    res = []
    fold_no = 1
    random.seed(seed)
    
    if LMNL:
        model = mdl.enhancedMNL_extraInput(beta_num, choices_num, nExtraFeatures, networkSize = networkSize, minima = None,
										  train_betas = True, hidden_layers=hidden, logits_activation='softmax')
        #optimizer = Adam(clipnorm=50.)
        model.compile(optimizer='adam', metrics=['AUC','categorical_accuracy'], loss='categorical_crossentropy')
        for train_index, test_index in kfold.split(np.zeros(len(y)), y):
            train1 = data[train_index,:,:]
            validation1 = data[test_index,:,:]
            train2 = extra_data[train_index,:,:]
            validation2 = extra_data[test_index,:,:]
            train_target = target[train_index,:]
            validation_target = target[test_index,:]
            model.fit([train1, train2],train_target, epochs = nEpochs, verbose = 0)
            y_pred = model.predict([validation1, validation2])
            pred = np.argmax(y_pred, axis = 1)
            true = np.argmax(validation_target, axis = 1)
            loss = log_loss(true, y_pred)
            auc = roc_auc_score(true, y_pred, multi_class='ovr')
            accuracy = accuracy_score(true, pred)
            mae = mean_absolute_error(true, pred)
            f1_scores = f1_score(true, pred, average=None)
            f1_mean = f1_score(true, pred, average='macro')
            #model.save('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/cv_saved_models/'+modelname+'F'+str(fold_no))
            weight = model.get_weights()
            t_param_H3 = np.count_nonzero(weight[0]) + np.count_nonzero(weight[1]) + np.count_nonzero(weight[2]) + np.count_nonzero(weight[3]) + np.count_nonzero(weight[4])
            loss_train, _, _ = model.evaluate([train1, train2],train_target, batch_size = batchSize, verbose = 0)
            log_likelihood = -loss_train*(train_target.shape[0])
            BIC = -2*log_likelihood + log(train_target.shape[0])*t_param_H3
            
            print(f'Score for fold {fold_no}: ', 'Loss: ', loss, 'AUC: ', auc, 'Accuracy: ', accuracy, 'MAE: ', mae)
            print('F1_scores:', f1_scores, f1_mean)
            print('Param: ', t_param_H3)
            print('log-likelihood: ', log_likelihood)
            print('BIC: ', BIC)
            cvscores = cvscores + [[modelname, networkSize, fold_no, loss, auc, accuracy, mae, f1_scores, f1_mean, BIC]]
            #res = res+ [true, pred, y_pred[:,0], y_pred[:,1], y_pred[:,2]]
            fold_no += 1
    
    if MNL_CNN:
        model = mdl.MNL(beta_num, choices_num)
    
        #optimizer = Adam(clipnorm=50.)
        model.compile(optimizer='adam', metrics=['AUC','categorical_accuracy'], loss='categorical_crossentropy')
        for train_index, test_index in kfold.split(np.zeros(len(y)), y):
            train1 = data[train_index,:,:]
            validation1 = data[test_index,:,:]
            train_target = target[train_index,:]
            validation_target = target[test_index,:]
            model.fit(train1,train_target, epochs = nEpochs, verbose = 0)
            y_pred = model.predict(validation1)
            pred = np.argmax(y_pred, axis = 1)
            true = np.argmax(validation_target, axis = 1)
            loss = log_loss(true, y_pred)
            auc = roc_auc_score(true, y_pred, multi_class='ovr')
            accuracy = accuracy_score(true, pred)
            mae = mean_absolute_error(true, pred)
            f1_scores = f1_score(true, pred, average=None)
            f1_mean = f1_score(true, pred, average='macro')
            model.save('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/cv_saved_models/'+modelname+'F'+str(fold_no))
            weight = model.get_weights()
            t_param_MNL_CNN = np.count_nonzero(weight[0])
            loss_train, _, _ = model.evaluate(train1,train_target, batch_size = batchSize, verbose = 0)
            log_likelihood = -loss_train*(train_target.shape[0])
            BIC = -2*log_likelihood + log(train_target.shape[0])*t_param_MNL_CNN
            
            print(f'Score for fold {fold_no}: ', 'Loss: ', loss, 'AUC: ', auc, 'Accuracy: ', accuracy, 'MAE: ', mae)
            print('F1_scores:', f1_scores, f1_mean)
            print('Param: ', t_param_MNL_CNN)
            print('log-likelihood: ', log_likelihood)
            print('BIC: ', BIC)
            cvscores = cvscores + [[modelname, networkSize, fold_no, loss, auc, accuracy, mae, f1_scores, f1_mean, BIC]]
            res = res+ [true, pred, y_pred[:,0], y_pred[:,1], y_pred[:,2]]
            fold_no += 1
    if DNN:
        for train_index, test_index in kfold.split(np.zeros(len(y)), y):
            train = data[train_index,:]
            validation = data[test_index,:]
            train_target = target[train_index,:]
            validation_target = target[test_index,:]
            model = dnn(hidden)
            model.fit(train,train_target, epochs = nEpochs, verbose = 0)
            y_pred = model.predict(validation)
            pred = np.argmax(y_pred, axis = 1)
            true = np.argmax(validation_target, axis = 1)
            loss = log_loss(true, y_pred)
            auc = roc_auc_score(true, y_pred, multi_class='ovr')
            accuracy = accuracy_score(true, pred)
            mae = mean_absolute_error(true, pred)
            f1_scores = f1_score(true, pred, average=None)
            f1_mean = f1_score(true, pred, average='macro')
            model.save('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/cv_saved_models/'+modelname+'F'+str(fold_no))
            weight = model.get_weights()
            #This should change if layers are increase
            t_param_DNN = np.count_nonzero(weight[0])+ np.count_nonzero(weight[1]) + np.count_nonzero(weight[2]) + np.count_nonzero(weight[3])
            loss_train, _ = model.evaluate(train,train_target, batch_size = batchSize, verbose = 0)
            log_likelihood = -loss_train*(train_target.shape[0])
            BIC = -2*log_likelihood + log(train_target.shape[0])*t_param_DNN
            
            print(f'Score for fold {fold_no}: ', 'Loss: ', loss, 'AUC: ', auc, 'Accuracy: ', accuracy, 'MAE: ', mae)
            print('F1_scores:', f1_scores, f1_mean)
            print('Param: ', t_param_DNN)
            print('log-likelihood: ', log_likelihood)
            print('BIC: ', BIC)
            cvscores = cvscores + [[modelname, networkSize, fold_no, loss, auc, accuracy, mae, f1_scores, f1_mean, BIC]]
            res = res+ [true, pred, y_pred[:,0], y_pred[:,1], y_pred[:,2]]
            fold_no += 1
    if MNL1:
        for train_index, test_index in kfold.split(np.zeros(len(y)), y):
            train = data[train_index,:]
            validation = data[test_index,:]
            train_target = target[train_index]
            validation_target = target[test_index]
            model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
            #model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            model.fit(train,train_target)
            y_pred = model.predict(validation)
            y_prob = model.predict_proba(validation)
            loss = log_loss(validation_target, y_prob)
            auc = roc_auc_score(validation_target, y_prob, multi_class='ovr')
            accuracy = accuracy_score(validation_target, y_pred)
            mae = mean_absolute_error(validation_target, y_pred)
            f1_scores = f1_score(validation_target, y_pred, average=None)
            f1_mean = f1_score(validation_target, y_pred, average='macro')
            
            t_param_MNL1 = np.count_nonzero(model.coef_)+ np.count_nonzero(model.intercept_)
            y_pred_train = model.predict_proba(train)
            loss_train = log_loss(train_target, y_pred_train)
            log_likelihood = -loss_train*(train_target.shape[0])
            BIC = -2*log_likelihood + log(train_target.shape[0])*t_param_MNL1
            print('Log-likelihood: ', log_likelihood)
            print('Parameters: ', t_param_MNL1)
            pickle.dump(model, open('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/cv_saved_models/'+modelname+'F'+str(fold_no), 'wb'))
            print(f'Score for fold {fold_no}: ', 'Loss: ', loss, 'AUC: ', auc, 'Accuracy: ', accuracy, 'MAE: ', mae, 'BIC', BIC)
            print('F1_scores:', f1_scores, f1_mean)
            cvscores = cvscores + [[modelname, 0, fold_no, loss, auc, accuracy, mae, f1_scores, f1_mean]]
            res = res+ [validation_target, y_pred, y_prob[:,0], y_prob[:,1], y_prob[:,2]]
            fold_no += 1
    if MNL2:
        for train_index, test_index in kfold.split(np.zeros(len(y)), y):
            train = data[train_index,:]
            validation = data[test_index,:]
            train_target = target[train_index]
            validation_target = target[test_index]
            y_train2 = list(train_target)
            mdl_fit = MNLogit_fit(y_train2, train)
            
            X = st.add_constant(validation)
            y_prob = mdl_fit.predict(X, linear = False)
            y_pred = np.argmax(y_prob, axis = 1)
            y_pred += 1
            
            loss = 0
            auc = roc_auc_score(validation_target, y_prob, multi_class='ovr', labels = [1, 2, 3])
            accuracy = accuracy_score(validation_target, y_pred)
            mae = mean_absolute_error(validation_target, y_pred)
            f1_scores = f1_score(validation_target, y_pred, average=None)
            f1_mean = f1_score(validation_target, y_pred, average='macro')
            BIC = mdl_fit.bic
            mdl_fit.summary2()
            pickle.dump(mdl_fit, open('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/cv_saved_models/'+modelname+'F'+str(fold_no), 'wb'))
            print(f'Score for fold {fold_no}: ', 'Loss: ', loss, 'AUC: ', auc, 'Accuracy: ', accuracy, 'MAE: ', mae, 'BIC', BIC)
            print('F1_scores:', f1_scores, f1_mean)
            cvscores = cvscores + [[modelname, 0, fold_no, loss, auc, accuracy, mae, f1_scores, f1_mean, BIC]]
            res = res+ [validation_target, y_pred, y_prob[:,0], y_prob[:,1], y_prob[:,2]]
            fold_no += 1
    """
    if OrdLogit:
        for train_index, test_index in kfold.split(np.zeros(len(y)), y):
            train = data[train_index,:]
            validation = data[test_index,:]
            train_target = target[train_index]
            validation_target = target[test_index]
            model = LogisticAT(alpha=0)
            #model = LogisticIT(alpha=0)
            model.fit(train, train_target)
            y_pred = model.predict(validation)
            y_prob = model.predict_proba(validation)
            loss = log_loss(validation_target, y_prob)
            auc = roc_auc_score(validation_target, y_prob, multi_class='ovr')
            accuracy = accuracy_score(validation_target, y_pred)
            mae = mean_absolute_error(validation_target, y_pred)
            f1_scores = f1_score(validation_target, y_pred, average=None)
            f1_mean = f1_score(validation_target, y_pred, average='macro')
            print(f'Score for fold {fold_no}: ', 'Loss: ', loss, 'AUC: ', auc, 'Accuracy: ', accuracy, 'MAE: ', mae)
            print('F1_scores:', f1_scores, f1_mean)
            cvscores = cvscores + [[modelname, 0, fold_no, loss, auc, accuracy, mae, f1_scores, f1_mean]]
            res = res+ [validation_target, y_pred, y_prob[:,0], y_prob[:,1], y_prob[:,2]]
            fold_no += 1
    
    """
    '''
    res_pd = pd.DataFrame(list(zip(res[0], res[1], res[2], res[3], res[4], 
                                   res[5], res[6], res[7], res[8], res[9],
                                   res[10], res[11], res[12], res[13], res[14],
                                   res[15], res[16], res[17], res[18], res[19],
                                   res[20], res[21], res[22], res[23], res[24])), 
                          columns = ["Fold1_true", "Fold1_pred", "Fold1_prob_pred_c1", "Fold1_prob_pred_c2", "Fold1_prob_pred_c3",
                                     "Fold2_true", "Fold2_pred", "Fold2_prob_pred_c1", "Fold2_prob_pred_c2", "Fold2_prob_pred_c3",
                                     "Fold3_true", "Fold3_pred", "Fold3_prob_pred_c1", "Fold3_prob_pred_c2", "Fold3_prob_pred_c3", 
                                     "Fold4_true", "Fold4_pred", "Fold4_prob_pred_c1", "Fold4_prob_pred_c2", "Fold4_prob_pred_c3",
                                     "Fold5_true", "Fold5_pred", "Fold5_prob_pred_c1", "Fold5_prob_pred_c2", "Fold5_prob_pred_c3"])
    '''
    #res_pd.to_csv(r'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/predictions/'+ modelname+ '_Train.csv', index = False)
    #print(res_pd.head(3).T)
    print(cvscores)
    K.clear_session()
    return cvscores


# ---------------------------- Run on test data -----------------------------------------
def runtest(data_train, target_train, data_test, target_test, beta_num, choices_num, nExtraFeatures , extra_train = 0, extra_test = 0, networkSize = 40, 
            nEpochs = 50, batch = 32, modelname = 'No name', LMNL = True, DNN = False, MNL = False, OrdLogit = False, MNL_CNN = False, hidden = 1, Ontr_SI_LS = False):
    '''
    Fits the model using training dataset and evaluates it with the testing dataset
    
    :param dataframe data_train: training dataset (independent variables)
    :param dataframe target_train: training dataset (response variable)
    :param dataframe data_test: testing dataset (independent variables)
    :param dataframe target_test: testing dataset (response variable)
    :param int beta_num: number of betas (coefficients) for the L-MNL model (knowledge-driven part (MNL))
    :param int choices_num: number of choices or classes
    :param int nExtraFeatures: number of variables in the data-driven part (DNN) of the L-MNL model
    :param dataframe extra_train: training dataset for the data-driven part (DNN) of the L-MNL model
    :param dataframe extra_test: testing dataset for the data-driven part (DNN) of the L-MNL model
    :param int networkSize: number of neurons for the data-driven part (DNN) of the L-MNL model
    :param int nEpochs: number of training epochs
    :param int batchSize: bacth size for training
    :param str modelname: name of the model
    :param bool LMNL: TRUE if the model that will be run is the L-MNL
    :param bool DNN: TRUE if the model that will be run is the DNN
    :param bool MNL: TRUE if the model that will be run is the MNL (statsmodel)
    :param bool OrdLogit: TRUE if the model that will be run is the Ordistic Logit All Threshold from mord, equivalent to the Cumulative logit model
    :param bool MNL_CNN: TRUE if the model that will be run is the MNL as a CNN
    :param int hidden: number of hidden layers
    :param bool Ontr_SI_LS: TRUE if the model that will be run is the Ontram single intercept linear shift, this is equivalent to the Cum Ordinal Logit (REMOVED) 
    :return: performance measures (AUC, accuracy, F1 score, MAE, and BIC) results, weigths of the neural networks, and training history
    
    '''
    
    testscores = []
    res = []
    random.seed(10)
    print(modelname)
    if LMNL:
        model = mdl.enhancedMNL_extraInput(beta_num, choices_num, nExtraFeatures, networkSize = networkSize, minima = None,
										  train_betas = True, hidden_layers=hidden, logits_activation='softmax')
        #optimizer = Adam(clipnorm=50.)
        model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
        met = model.fit([data_train, extra_train],target_train, epochs = nEpochs, verbose = 0)
        #met = model.fit([data_train, extra_train],target_train, validation_split=0.33, epochs = nEpochs, verbose = 0)
        model.save('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/saved_models/'+modelname)
        
        y_pred = model.predict([data_test, extra_test])
        pred = np.argmax(y_pred, axis = 1)
        true = np.argmax(target_test, axis = 1)
        loss = log_loss(true, y_pred)
        auc = roc_auc_score(true, y_pred, multi_class='ovr')
        accuracy = accuracy_score(true, pred)
        mae = mean_absolute_error(true, pred)
        f1_scores = f1_score(true, pred, average = None)
        f1_mean = f1_score(true, pred, average='macro')
        weight = model.get_weights()
        betas_layer = model.get_layer(name = 'Utilities')
        betas = betas_layer.get_weights()
        
        t_param_H3 = np.count_nonzero(weight[0]) + np.count_nonzero(weight[1]) + np.count_nonzero(weight[2]) + np.count_nonzero(weight[3]) + np.count_nonzero(weight[4])
        loss_train, _ = model.evaluate([data_train, extra_train],target_train, batch_size = batch, verbose = 0)
        log_likelihood = -loss_train*(target_train.shape[0])
        BIC = -2*log_likelihood + log(target_train.shape[0])*t_param_H3
        
        print('Loss: ', loss, 'AUC: ', auc, 'Accuracy: ', accuracy, 'MAE: ', mae)
        print('F1_scores:', f1_scores, f1_mean)
        print('BIC: ', BIC)
        print('Betas: ', betas)
        
        testscores = testscores + [[modelname, networkSize, loss, auc, accuracy, mae, f1_scores, f1_mean, weight, BIC]]
        res = res + [true, pred, y_pred[:,0], y_pred[:,1], y_pred[:,2]]
        
    if MNL_CNN:
        model = mdl.MNL(beta_num, choices_num)						
        
        #optimizer = Adam(clipnorm=50.)
        model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
        met = model.fit(data_train, target_train, epochs = nEpochs, verbose = 0)
        #met = model.fit(data_train, target_train, validation_split=0.33, epochs = nEpochs, verbose = 0)
        y_pred = model.predict(data_test)
        pred = np.argmax(y_pred, axis = 1)
        true = np.argmax(target_test, axis = 1)
        loss = log_loss(true, y_pred)
        auc = roc_auc_score(true, y_pred, multi_class='ovr')
        accuracy = accuracy_score(true, pred)
        mae = mean_absolute_error(true, pred)
        f1_scores = f1_score(true, pred, average = None)
        f1_mean = f1_score(true, pred, average='macro')
        weight = model.get_weights()
        betas_layer = model.get_layer(name = 'Utilities')
        betas = betas_layer.get_weights()
        
        t_param_MNL_CNN = np.count_nonzero(weight[0])
        loss_train, _ = model.evaluate(data_train, target_train, batch_size = batch, verbose = 0)
        log_likelihood = -loss_train*(target_train.shape[0])
        BIC = -2*log_likelihood + log(target_train.shape[0])*t_param_MNL_CNN
            
        model.save('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/saved_models/'+modelname)
        print('Loss: ', loss, 'AUC: ', auc, 'Accuracy: ', accuracy, 'MAE: ', mae)
        print('F1_scores:', f1_scores, f1_mean)
        print('BIC: ', BIC)
        print('Betas: ', betas)
        
        testscores = testscores + [[modelname, networkSize, loss, auc, accuracy, mae, f1_scores, f1_mean, weight]]
        res = res + [true, pred, y_pred[:,0], y_pred[:,1], y_pred[:,2]]
        
    if DNN:
        model = dnn(hidden)
        met = model.fit(data_train, target_train, validation_split=0.33, epochs = nEpochs, verbose = 0)
        y_pred = model.predict(data_test)
        pred = np.argmax(y_pred, axis = 1)
        true = np.argmax(target_test, axis = 1)
        loss = log_loss(true, y_pred)
        auc = roc_auc_score(true, y_pred, multi_class='ovr')
        accuracy = accuracy_score(true, pred)
        mae = mean_absolute_error(true, pred)
        f1_scores = f1_score(true, pred, average=None)
        f1_mean = f1_score(true, pred, average='macro')
        weight = model.get_weights()
        if hidden == 1:
            t_param_DNN = np.count_nonzero(weight[0])+ np.count_nonzero(weight[1]) + np.count_nonzero(weight[2]) + np.count_nonzero(weight[3])
        else:
            t_param_DNN = np.count_nonzero(weight[0])+ np.count_nonzero(weight[1]) + np.count_nonzero(weight[2]) + np.count_nonzero(weight[3]) + np.count_nonzero(weight[4]) + np.count_nonzero(weight[5])
        
        loss_train, _ = model.evaluate(data_train,target_train, batch_size = batch)
        log_likelihood = -loss_train*(target_train.shape[0])
        BIC = -2*log_likelihood + log(target_train.shape[0])*t_param_DNN
        
        model.save('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/saved_models/'+modelname)
        print('Loss: ', loss, 'AUC: ', auc, 'Accuracy: ', accuracy, 'MAE: ', mae, 'BIC: ', BIC)
        print('F1_scores:', f1_scores, f1_mean)
        print('BIC: ', BIC)
        testscores = testscores + [[modelname, networkSize, loss, auc, mae, accuracy, f1_scores, f1_mean, weight, BIC]]
        res = res + [true, pred, y_pred[:,0], y_pred[:,1], y_pred[:,2]]
        
    if MNL: #Other version of MNL
        model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
        #model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        met = model.fit(data_train, target_train)
        y_pred = model.predict(data_test)
        pred_proba = model.predict_proba(data_test)
        loss = log_loss(target_test, pred_proba)
        auc = roc_auc_score(target_test, pred_proba, multi_class='ovr')
        mae = mean_absolute_error(target_test, y_pred)
        f1_scores = f1_score(target_test, y_pred, average=None)
        f1_mean = f1_score(target_test, y_pred, average='macro')
        pickle.dump(model, open('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/saved_models/'+modelname, 'wb'))
        print('Loss: ', loss, 'AUC: ', auc, 'Accuracy: ', accuracy, 'MAE: ', mae)
        print('F1_scores:', f1_scores, f1_mean)
        testscores = testscores + [[modelname, 0, loss, auc, accuracy, mae, f1_scores, f1_mean]]
        res = res + [target_test, y_pred, pred_proba[:,0], pred_proba[:,1], pred_proba[:,2]]
        weight = 0
    """   
    if OrdLogit:
        model = LogisticAT(alpha=0)
        #model = LogisticIT(alpha=0)
        met = model.fit(data_train, target_train)
        y_pred = model.predict(data_test)
        pred_proba = model.predict_proba(data_test)
        loss = log_loss(target_test, pred_proba)
        auc = roc_auc_score(target_test, pred_proba, multi_class='ovr')
        accuracy = accuracy_score(target_test, y_pred)
        mae = mean_absolute_error(target_test, y_pred)
        f1_scores = f1_score(target_test, y_pred, average=None)
        f1_mean = f1_score(target_test, y_pred, average='macro')
        print('Loss: ', loss, 'AUC: ', auc, 'Accuracy: ', accuracy, 'MAE: ', mae)
        print('F1_scores:', f1_scores, f1_mean)
        testscores = testscores + [[modelname, 0, loss, auc, accuracy, mae, f1_scores, f1_mean]]
        res = res + [target_test, y_pred, pred_proba[:,0], pred_proba[:,1], pred_proba[:,2]]
        weight = 0
    
    """   
    res_pd = pd.DataFrame(list(zip(res[0], res[1], res[2], res[3], res[4])), 
                          columns = ["true", "pred", "prob_pred_c1", "prob_pred_c2", "prob_pred_c3"])
    res_pd.to_csv(r'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/predictions/'+ modelname+ '_Test.csv', index = False)
    return testscores, weight, met    
    

def savetocsv(X, y, name = 'New.csv'):
    '''
    Concatenates X and y, and saves it in a csv file
    
    :param dataframe X: independent variables dataset
    :param dataframe y: response variable
    :param str name: file name to save
    :return: None
    
    '''
    dataset = pd.concat([X, y], axis=1)
    dataset.to_csv(r'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/data/'+name, index = False)
    
    print('Ready')
    return None



# -------------------------------------------------------------------------------------#
# --------------------------------------- Main ----------------------------------------#
# -------------------------------------------------------------------------------------#
 
#filepath = '../input/'
filepath = 'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/data_lending'


data = data_import(filepath)

data = down_sample(data, seed = seed)
#data.columns
#data.groupby(['loan_status', 'purpose'])['purpose'].agg(["count"])
#data['purpose'].value_counts()

#print(data.info(verbose=True, null_counts=True))
#print(data.isna().sum())

X_train, X_test, y_train, y_test = data_split(data) #Runtime 517 seg

X_train, X_test = pre_process(X_train, X_test) #Simple Imputer runtime 24 seg 



#print(X_train.info(verbose=True, null_counts=True))
print(X_train.isna().sum())
print(X_test.isna().sum())

#X_train.columns

'''# --------------------------------------------------------------------------
# Identification of correlated features (takes aprox 1h14min)

starttime = timeit.default_timer()
corr_matrix, corr_features = corr_features_spear(X_train) # Runtime 2250 - 2285 seg (38min) 288
print("The time difference 1 is :", timeit.default_timer() - starttime)

starttime = timeit.default_timer()
pval_matrix, corr_features_pval = pvalue_features_spear(X_train) # Runtime 2155 seg (36min)
print("The time difference 2 is :", timeit.default_timer() - starttime)

num_corr = corr_matrix[abs(corr_matrix)> 0.5].count()
num_corr = num_corr[corr_matrix[abs(corr_matrix)> 0.5].count()>1]
print(num_corr)

num_corr2 = pval_matrix[abs(pval_matrix)== 0].count()
num_corr2 = num_corr2[pval_matrix[abs(pval_matrix)== 0].count()>1]
print(num_corr2)

pval_matrix.to_excel("C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/pval_matrix.xlsx") 
df_corr_features = pd.DataFrame(corr_features)
df_corr_features.to_excel("C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/corr_features_select.xlsx") 

# -------------------------------------------------------------------------- ''' 

X_train, X_test = remove_corr_features(X_train, X_test)

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


'''# ---------------------------------------------------------------------------
# Mutual information evaluation

starttime = timeit.default_timer()
mutual_info_array10 = mutual_info_classif(X_train, y_train, n_neighbors = 10) #Duration 2320 seg (39 min)
print("The time difference 5 is :", timeit.default_timer() - starttime)

# Kendall's tau
ken_tau = corr_features_ken(X_train, y_train)

# -----------------------------------------------------------------------------'''


X_train, X_test = remove_kendall(X_train, X_test)

'''
# ---------------------------------------------------------------------------
y_train2 = list(y_train)
starttime = timeit.default_timer()
mdl_fit = MNLogit_fit(y_train2, X_train_scale) # 166 seg 
print("The time difference 1 is :", timeit.default_timer() - starttime)
# 'recoveries' gives problems estimating the Hessian nan outcome
# -----------------------------------------------------------------------------'''

X_train_scale, X_test_scale = trans_minmax(X_train, X_test)

X_train_scale.reset_index(drop=True, inplace=True)
X_test_scale.reset_index(drop=True, inplace=True)

#y_train.reshape(-1,1)

choices_num = 3  # Current, late, default
batchSize = 32

#savetocsv(X_train_scale, y_train, name = 'Train.csv')
#savetocsv(X_test_scale, y_test, name = 'Test.csv')

'''
# ---------------------------------------------------------------------------------------------------------------------------------
# Prepare keras input
filePath2 = 'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/models/'
starttime = timeit.default_timer()
_, _, _, _, _, _, data_name1, data_name2, data_name3, test_name1, test_name2, test_name3 = keras_input(filePath2, X_train_scale, y_train, X_test_scale, y_test)
print("The time difference Keras input is :", timeit.default_timer() - starttime)

_, data_name5, test_name5 = keras_inputMNL(filePath2, X_train_scale,  y_train, X_test_scale, y_test)


#----------------------------------------------------------------------------------------------------------------------------------'''

print("L-MNL")
#lmnlArchitecture = True
beta_num1 = 58
beta_num2 = 20
beta_num3 = 10 
beta_num5 = 72

nExtraFeatures1 = 7
nExtraFeatures2 = 26
nExtraFeatures3 = 31

data_name1 = 'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/models/input_H1_v2.npy'
data_name2 = 'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/models/input_H2_v2.npy'
data_name3 = 'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/models/input_H3_v2.npy'
data_name5 = 'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/models/input_MNL_v2.npy'

test_name1 = 'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/models/input_H1_test_v2.npy'
test_name2 = 'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/models/input_H2_test_v2.npy'
test_name3 = 'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/models/input_H3_test_v2.npy'
test_name5 = 'C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/models/input_MNL_test_v2.npy'

# Load training
train_data1 = np.load(data_name1)
labels1 = train_data1[:,-1,:]
train_data1 = np.delete(train_data1, -1, axis = 1)
train_data1 = np.expand_dims(train_data1, -1)
extra_data1 = np.load(data_name1[:-4] + '_extra.npy')
extra_data1 = np.expand_dims(extra_data1, -1)
extra_data1 = np.expand_dims(extra_data1, -1)

train_data2 = np.load(data_name2)
labels2 = train_data2[:,-1,:]
train_data2 = np.delete(train_data2, -1, axis = 1)
train_data2 = np.expand_dims(train_data2, -1)
extra_data2 = np.load(data_name2[:-4] + '_extra.npy')
extra_data2 = np.expand_dims(extra_data2, -1)
extra_data2 = np.expand_dims(extra_data2, -1)

train_data3 = np.load(data_name3)
labels3 = train_data3[:,-1,:]
train_data3 = np.delete(train_data3, -1, axis = 1)
train_data3 = np.expand_dims(train_data3, -1)
extra_data3 = np.load(data_name3[:-4] + '_extra.npy')
extra_data3 = np.expand_dims(extra_data3, -1)
extra_data3 = np.expand_dims(extra_data3, -1)

#MNL as CNN
train_data5 = np.load(data_name5)
labels5 = train_data5[:,-1,:]
train_data5 = np.delete(train_data5, -1, axis = 1)
train_data5 = np.expand_dims(train_data5, -1)

# Load testing
test_data1 = np.load(test_name1)
labels1_test = test_data1[:,-1,:]
test_data1 = np.delete(test_data1, -1, axis = 1)
test_data1 = np.expand_dims(test_data1, -1)
extra_data1_test = np.load(test_name1[:-4] + '_extra.npy')
extra_data1_test = np.expand_dims(extra_data1_test, -1)
extra_data1_test = np.expand_dims(extra_data1_test, -1)

test_data2 = np.load(test_name2)
labels2_test = test_data2[:,-1,:]
test_data2 = np.delete(test_data2, -1, axis = 1)
test_data2 = np.expand_dims(test_data2, -1)
extra_data2_test = np.load(test_name2[:-4] + '_extra.npy')
extra_data2_test = np.expand_dims(extra_data2_test, -1)
extra_data2_test = np.expand_dims(extra_data2_test, -1)

test_data3 = np.load(test_name3)
labels3_test = test_data3[:,-1,:]
test_data3 = np.delete(test_data3, -1, axis = 1)
test_data3 = np.expand_dims(test_data3, -1)
extra_data3_test = np.load(test_name3[:-4] + '_extra.npy')
extra_data3_test = np.expand_dims(extra_data3_test, -1)
extra_data3_test = np.expand_dims(extra_data3_test, -1)

#MNL as CNN
test_data5 = np.load(test_name5)
labels5_test = test_data5[:,-1,:]
test_data5 = np.delete(test_data5, -1, axis = 1)
test_data5 = np.expand_dims(test_data5, -1)


# ---------------- L-MNL models --------------------------------------

# L-MNL Train ---------

y_train = np.argmax(labels1, axis = 1)
# L-MNL H1
starttime = timeit.default_timer()
cvscores1_100 = crossval(train_data1, labels1, y_train, beta_num1, choices_num, nExtraFeatures1, extra_data = extra_data1, networkSize =16, modelname = 'L-MNL H1_16N_100e3v', LMNL = True, 
                         nEpochs = 100, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# L-MNL H2
starttime = timeit.default_timer()
cvscores2_16 = crossval(train_data2, labels2, y_train, beta_num2, choices_num, nExtraFeatures2 , extra_data = extra_data2, networkSize =16, modelname = 'L-MNL H2_16N_100e3v', LMNL = True, 
                         nEpochs = 100, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# L-MNL H3
starttime = timeit.default_timer()
cvscores3_32 = crossval(train_data3, labels3, y_train, beta_num3, choices_num, nExtraFeatures3 , extra_data = extra_data3, networkSize =32, modelname = 'L-MNL H3_32N_100e3v', LMNL = True, 
                         nEpochs = 100, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)



# L-MNL Test -------- 

# L-MNL H1
starttime = timeit.default_timer()
testscores1, weight1, history1 = runtest(train_data1, labels1, test_data1, labels1_test, beta_num1, choices_num, nExtraFeatures1, extra_train = extra_data1, extra_test = extra_data1_test, networkSize = 16, 
            nEpochs = 100, modelname = 'L-MNL H1_16N_100e_3v', LMNL = True, DNN = False, MNL = False, OrdLogit = False)
print("The time difference for L-MNL H1 is :", timeit.default_timer() - starttime)

# L-MNL H2
starttime = timeit.default_timer()
testscores2, weight2, history2 = runtest(train_data2, labels2, test_data2, labels2_test, beta_num2, choices_num, nExtraFeatures2, extra_train = extra_data2, extra_test = extra_data2_test, networkSize = 16, 
            nEpochs = 100, modelname = 'L-MNL H2_16N_100e_3v', LMNL = True, DNN = False, MNL = False, OrdLogit = False)
print("The time difference for L-MNL H2 is :", timeit.default_timer() - starttime)

# L-MNL H3
starttime = timeit.default_timer()
testscores3, weight3, history3 = runtest(train_data3, labels3, test_data3, labels3_test, beta_num3, choices_num, nExtraFeatures3, extra_train = extra_data3, extra_test = extra_data3_test, networkSize = 32, 
            nEpochs = 100, modelname = 'L-MNL H3_32N_100e_3v', LMNL = True, DNN = False, MNL = False, OrdLogit = False)
print("The time difference for L-MNL H3 is :", timeit.default_timer() - starttime)


# MNL as CNN ----------------------------------------------------

# MNL as CNN Train ---------
starttime = timeit.default_timer()
cvscores5 = crossval(train_data5, labels5, y_train, beta_num5, choices_num, 0 , extra_data = 0, modelname = 'MNL as CNN', LMNL = False, 
             MNL_CNN = True, nEpochs = 100, seed = seed)
print("The time difference crossval MNL as CNN is :", timeit.default_timer() - starttime)

# MNL as CNN Test ----------
starttime = timeit.default_timer()
testscores5_100, weight5_100, history5_100 = runtest(train_data5, labels5, test_data5, labels5_test, beta_num5, choices_num, 0, extra_train = 0, extra_test = 0, 
            nEpochs = 100, modelname = 'MNL as CNN_100e_3v', LMNL = False, DNN = False, MNL = False, OrdLogit = False, MNL_CNN = True)
print("The time difference for MNL as CNN 100e is :", timeit.default_timer() - starttime)



# ---------------------- Full DNN -------------------------------------------
print("DNN")
beta_num4 = 0
nExtraFeatures4 = 36

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
dummy_y = np_utils.to_categorical(encoded_Y)

encoder.fit(y_test)
encoded_Y_test = encoder.transform(y_test)
dummy_y_test = np_utils.to_categorical(encoded_Y_test)


# Train DNN with 100 epochs
starttime = timeit.default_timer()
cvscoresDNN1 = crossval(X_train_scale.values, dummy_y, y_train, beta_num4, choices_num, nExtraFeatures4 , X_train_scale, networkSize = 40, modelname = 'DNN_100ep_3v', LMNL = False, DNN = True, OrdLogit = False, 
             seed = seed, nEpochs = 100)
print("The time difference to run DNN is :", timeit.default_timer() - starttime)

# Test
starttime = timeit.default_timer()
testscoresDNN, weightDNN1, historyDNN1 = runtest(X_train_scale.values, dummy_y, X_test_scale.values, dummy_y_test, 0, choices_num, nExtraFeatures4, modelname = 'DNN', LMNL = False, DNN = True, MNL = False, OrdLogit = False)
print("The time difference to run DNN without dropout is :", timeit.default_timer() - starttime)

"""# ------------------------------------------------------------
# summarize history for accuracy
plt.plot(history2_100_32N.history['categorical_accuracy'])
plt.plot(history2_100_32N.history['val_categorical_accuracy'])
plt.title('L-MNL H2 with 32 neurons model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(history2_100_32N.history['loss'])
plt.plot(history2_100_32N.history['val_loss'])
plt.title('L-MNL H2 with 32 neurons model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
# ---------------------------------------------------------------"""

# ------------------- Multinomial Logit model -------------------------
print("MNL")
starttime = timeit.default_timer()
cvscoresMNL = crossval(X_train_scale.values, encoded_Y, y_train, 0, choices_num, 0 , 0, modelname = 'MNL', LMNL = False, DNN = False, MNL = True, OrdLogit = False, 
             seed = seed, batchSize = 500)
print("The time difference to run MNL is :", timeit.default_timer() - starttime)

# Test
starttime = timeit.default_timer()
testscoresMNL = runtest(X_train_scale.values, encoded_Y, X_test_scale.values, encoded_Y_test, 0, choices_num, 0, modelname = 'MNL', LMNL = False, DNN = False, MNL = True, OrdLogit = False)
print("The time difference to run MNL is :", timeit.default_timer() - starttime)

# MNL 2 ---
print("MNL")
starttime = timeit.default_timer()
cvscoresMNL2 = crossval(X_train_scale.values, y_train, y_train, 0, choices_num, 0 , 0, modelname = 'MNL_stats', LMNL = False, DNN = False, MNL2 = True)
print("The time difference to run MNL is :", timeit.default_timer() - starttime)


'''
# Ordinal Logit model ----------------------------------------------
print("OrdLogit")
starttime = timeit.default_timer()
cvscoresOrdLogit = crossval(X_train_scale.values, encoded_Y, y_train, 0, choices_num, 0 , 0, modelname = 'OrdLog', LMNL = False, DNN = False, MNL = False, OrdLogit = True, 
             seed = seed, batchSize = 500)
print("The time difference to run OrdLogit is :", timeit.default_timer() - starttime)

# Test
starttime = timeit.default_timer()
testscore = runtest(X_train_scale.values, y_train, X_test_scale.values, y_test, 0, choices_num, 0, modelname = 'Ordistic Log AT', LMNL = False, DNN = False, MNL = False, OrdLogit = True)
print("The time difference to run OrdLogit is :", timeit.default_timer() - starttime)

'''

# ------------------------- Neurons analysis --------------------------------------
y_test = np.argmax(labels1_test, axis = 1)
y_test += 1 
y_test[:5]

# L-MNL H1
# 16 neurons - 1.5h
starttime = timeit.default_timer()
cvscores1_test_16N = crossval(test_data1, labels1_test, y_test, beta_num1, choices_num, nExtraFeatures1, extra_data = extra_data1_test, networkSize =16, modelname = 'Neurons_H1_50e_16N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 32 neurons - 1.3h
starttime = timeit.default_timer()
cvscores1_test_32N = crossval(test_data1, labels1_test, y_test, beta_num1, choices_num, nExtraFeatures1, extra_data = extra_data1_test, networkSize =32, modelname = 'Neurons_H1_50e_32N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 40 neurons - 1.6h
starttime = timeit.default_timer()
cvscores1_test_40N = crossval(test_data1, labels1_test, y_test, beta_num1, choices_num, nExtraFeatures1, extra_data = extra_data1_test, networkSize =40, modelname = 'Neurons_H1_50e_40N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 56 neurons
starttime = timeit.default_timer()
cvscores1_test_56N = crossval(test_data1, labels1_test, y_test, beta_num1, choices_num, nExtraFeatures1, extra_data = extra_data1_test, networkSize =56, modelname = 'Neurons_H1_50e_56N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 72 neurons
starttime = timeit.default_timer()
cvscores1_test_72N = crossval(test_data1, labels1_test, y_test, beta_num1, choices_num, nExtraFeatures1, extra_data = extra_data1_test, networkSize =72, modelname = 'Neurons_H1_50e_72N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 96 neurons
starttime = timeit.default_timer()
cvscores1_test_96N = crossval(test_data1, labels1_test, y_test, beta_num1, choices_num, nExtraFeatures1, extra_data = extra_data1_test, networkSize =96, modelname = 'Neurons_H1_50e_96N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)


# L-MNL H2
# 16 neurons
starttime = timeit.default_timer()
cvscores2_test_16N = crossval(test_data2, labels2_test, y_test, beta_num2, choices_num, nExtraFeatures2, extra_data = extra_data2_test, networkSize =16, modelname = 'Neurons_H2_50e_16N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 32 neurons
starttime = timeit.default_timer()
cvscores2_test_32N = crossval(test_data2, labels2_test, y_test, beta_num2, choices_num, nExtraFeatures2, extra_data = extra_data2_test, networkSize =32, modelname = 'Neurons_H2_50e_32N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 40 neurons
starttime = timeit.default_timer()
cvscores2_test_40N = crossval(test_data2, labels2_test, y_test, beta_num2, choices_num, nExtraFeatures2, extra_data = extra_data2_test, networkSize =40, modelname = 'Neurons_H2_50e_40N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 56 neurons
starttime = timeit.default_timer()
cvscores2_test_56N = crossval(test_data2, labels2_test, y_test, beta_num2, choices_num, nExtraFeatures2, extra_data = extra_data2_test, networkSize =56, modelname = 'Neurons_H2_50e_56N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 72 neurons
starttime = timeit.default_timer()
cvscores2_test_72N = crossval(test_data2, labels2_test, y_test, beta_num2, choices_num, nExtraFeatures2, extra_data = extra_data2_test, networkSize =72, modelname = 'Neurons_H2_50e_72N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 96 neurons
starttime = timeit.default_timer()
cvscores2_test_96N = crossval(test_data2, labels2_test, y_test, beta_num2, choices_num, nExtraFeatures2, extra_data = extra_data2_test, networkSize =96, modelname = 'Neurons_H2_50e_96N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)


# L-MNL H3
# 16 neurons
starttime = timeit.default_timer()
cvscores3_test_16N = crossval(test_data3, labels3_test, y_test, beta_num3, choices_num, nExtraFeatures3, extra_data = extra_data3_test, networkSize =16, modelname = 'Neurons_H3_50e_16N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 32 neurons
starttime = timeit.default_timer()
cvscores3_test_32N = crossval(test_data3, labels3_test, y_test, beta_num3, choices_num, nExtraFeatures3, extra_data = extra_data3_test, networkSize =32, modelname = 'Neurons_H3_50e_32N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 40 neurons
starttime = timeit.default_timer()
cvscores3_test_40N = crossval(test_data3, labels3_test, y_test, beta_num3, choices_num, nExtraFeatures3, extra_data = extra_data3_test, networkSize =40, modelname = 'Neurons_H3_50e_40N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 56 neurons
starttime = timeit.default_timer()
cvscores3_test_56N = crossval(test_data3, labels3_test, y_test, beta_num3, choices_num, nExtraFeatures3, extra_data = extra_data3_test, networkSize =56, modelname = 'Neurons_H3_50e_56N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 72 neurons
starttime = timeit.default_timer()
cvscores3_test_72N = crossval(test_data3, labels3_test, y_test, beta_num3, choices_num, nExtraFeatures3, extra_data = extra_data3_test, networkSize =72, modelname = 'Neurons_H3_50e_72N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)

# 96 neurons
starttime = timeit.default_timer()
cvscores3_test_96N = crossval(test_data3, labels3_test, y_test, beta_num3, choices_num, nExtraFeatures3, extra_data = extra_data3_test, networkSize =96, modelname = 'Neurons_H3_50e_96N', LMNL = True, 
                         nEpochs = 50, seed = seed, hidden = 1)
print("The time difference crossval is :", timeit.default_timer() - starttime)


# ----------------------------------------------------------------------------------------

# -------------------- Evaluation (additional metrics) -------------------------------------

# Confusion matrices
# Test results
h1_test = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/predictions/L-MNL H1_100Ep_Test.csv')
h2_test = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/predictions/L-MNL H2_16N_100e_Test.csv')
h3_test = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/predictions/L-MNL H3_32N_100e_Test.csv')
mnl_test = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/predictions/MNL_Test.csv')
mnl_cnn_test = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/predictions/MNL as CNN_100ep_Test.csv')
dnn_test = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/predictions/DNN_Test.csv')
cumlogit_test = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/predictions/Polr_y_pred.csv')


cumlogit_fold1_pred = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/data/1Fold_y_pred.csv')
cumlogit_fold1_true = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/data/1Fold_y_true.csv')
cumlogit_fold2_pred = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/data/2Fold_y_pred.csv')
cumlogit_fold2_true = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/data/2Fold_y_true.csv')
cumlogit_fold3_pred = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/data/3Fold_y_pred.csv')
cumlogit_fold3_true = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/data/3Fold_y_true.csv')
cumlogit_fold4_pred = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/data/4Fold_y_pred.csv')
cumlogit_fold4_true = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/data/4Fold_y_true.csv')
cumlogit_fold5_pred = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/data/5Fold_y_pred.csv')
cumlogit_fold5_true = pd.read_csv('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/data/5Fold_y_true.csv')

'''
cumlogit_train = pd.concat([cumlogit_fold1_true, cumlogit_fold1_pred, cumlogit_fold2_true, cumlogit_fold2_pred,
                            cumlogit_fold3_true, cumlogit_fold3_pred, cumlogit_fold4_true, cumlogit_fold4_pred,
                            cumlogit_fold5_true, cumlogit_fold5_pred], axis = 1)

cumlogit_train.head()
'''
cumlogit_test.max()
cumlogit_test.head(5)

cumLogit_pred = np.argmax(cumlogit_test.values, axis = 1)
cumLogit_pred += 1 
y_test[:5]
cumLogit_pred[:5]


conf_matrix_h1_test = confusion_matrix(h1_test['true'].values, h1_test['pred'].values, labels = [0, 1, 2], normalize = 'pred')
conf_matrix_h2_test = confusion_matrix(h2_test['true'].values, h2_test['pred'].values, labels = [0, 1, 2], normalize = 'pred')
conf_matrix_h3_test = confusion_matrix(h3_test['true'].values, h3_test['pred'].values, labels = [0, 1, 2], normalize = 'pred')
conf_matrix_mnl_test = confusion_matrix(mnl_cnn_test['true'].values, mnl_cnn_test['pred'].values, labels = [0, 1, 2], normalize = 'pred')
conf_matrix_dnn_test = confusion_matrix(dnn_test['true'].values, dnn_test['pred'].values, labels = [0, 1, 2], normalize = 'pred')
conf_matrix_cumLogit_test = confusion_matrix(y_test, cumLogit_pred, labels = [1, 2, 3], normalize = 'pred')


# Cumulative Ordinal Logit metrics ------------------------------------
# Test
cumlogit_test.rename(columns = {'Categ1' : 1, 'Categ2' : 2, 'Categ3' : 3}, inplace = True)

auc = roc_auc_score(y_test.values, cumlogit_test.values, multi_class='ovr')
accuracy = accuracy_score(y_test.values, cumLogit_pred)
mae = mean_absolute_error(y_test.values, cumLogit_pred)
f1_scores = f1_score(y_test.values, cumLogit_pred, average=None)
f1_mean = f1_score(y_test.values, cumLogit_pred, average='macro')

#Train
# Fold 1

cumlogit_fold1_true.replace({'Categ1': 1, 'Categ2': 2, 'Categ3': 3}, inplace = True)

cumlogit_fold1_pred_max = np.argmax(cumlogit_fold1_pred.values, axis = 1)
cumlogit_fold1_pred_max += 1

cumlogit_fold1_pred.rename(columns = {'Categ1' : 1, 'Categ2' : 2, 'Categ3' : 3}, inplace = True)
cumlogit_fold1_pred.head()
cumlogit_fold1_pred_max[:10]
cumlogit_fold1_true.head()

auc_cum_f1 = roc_auc_score(cumlogit_fold1_true.values, cumlogit_fold1_pred.values, multi_class='ovr')
accuracy_cum_f1 = accuracy_score(cumlogit_fold1_true.values, cumlogit_fold1_pred_max)
mae_cum_f1 = mean_absolute_error(cumlogit_fold1_true.values, cumlogit_fold1_pred_max)
f1_scores_cum_f1 = f1_score(cumlogit_fold1_true.values, cumlogit_fold1_pred_max, average=None)
f1_mean_cum_f1 = f1_score(cumlogit_fold1_true.values, cumlogit_fold1_pred_max, average='macro')

print('Fold 1: AUC: ', auc_cum_f1, 'Accuracy: ', accuracy_cum_f1, 'MAE: ', mae_cum_f1,
      'F1 scores per class: ', f1_scores_cum_f1, 'F1 score macro: ', f1_mean_cum_f1)

# Fold 2

cumlogit_fold2_true.replace({'Categ1': 1, 'Categ2': 2, 'Categ3': 3}, inplace = True)

cumlogit_fold2_pred_max = np.argmax(cumlogit_fold2_pred.values, axis = 1)
cumlogit_fold2_pred_max += 1

cumlogit_fold2_pred.rename(columns = {'Categ1' : 1, 'Categ2' : 2, 'Categ3' : 3}, inplace = True)
print(cumlogit_fold2_pred.head())
print(cumlogit_fold2_pred_max[:10])
print(cumlogit_fold2_true.head())

auc_cum_f2 = roc_auc_score(cumlogit_fold2_true.values, cumlogit_fold2_pred.values, multi_class='ovr')
accuracy_cum_f2 = accuracy_score(cumlogit_fold2_true.values, cumlogit_fold2_pred_max)
mae_cum_f2 = mean_absolute_error(cumlogit_fold2_true.values, cumlogit_fold2_pred_max)
f1_scores_cum_f2 = f1_score(cumlogit_fold2_true.values, cumlogit_fold2_pred_max, average=None)
f1_mean_cum_f2 = f1_score(cumlogit_fold2_true.values, cumlogit_fold2_pred_max, average='macro')

print('Fold 2: AUC: ', auc_cum_f2, 'Accuracy: ', accuracy_cum_f2, 'MAE: ', mae_cum_f2,
      'F1 scores per class: ', f1_scores_cum_f2, 'F1 score macro: ', f1_mean_cum_f2)

# Fold 3

cumlogit_fold3_true.replace({'Categ1': 1, 'Categ2': 2, 'Categ3': 3}, inplace = True)

cumlogit_fold3_pred_max = np.argmax(cumlogit_fold3_pred.values, axis = 1)
cumlogit_fold3_pred_max += 1

cumlogit_fold3_pred.rename(columns = {'Categ1' : 1, 'Categ2' : 2, 'Categ3' : 3}, inplace = True)
print(cumlogit_fold3_pred.head())
print(cumlogit_fold3_pred_max[:10])
print(cumlogit_fold3_true.head())

auc_cum_f3 = roc_auc_score(cumlogit_fold3_true.values, cumlogit_fold3_pred.values, multi_class='ovr')
accuracy_cum_f3 = accuracy_score(cumlogit_fold3_true.values, cumlogit_fold3_pred_max)
mae_cum_f3 = mean_absolute_error(cumlogit_fold3_true.values, cumlogit_fold3_pred_max)
f1_scores_cum_f3 = f1_score(cumlogit_fold3_true.values, cumlogit_fold3_pred_max, average=None)
f1_mean_cum_f3 = f1_score(cumlogit_fold3_true.values, cumlogit_fold3_pred_max, average='macro')

print('Fold 3: AUC: ', auc_cum_f3, 'Accuracy: ', accuracy_cum_f3, 'MAE: ', mae_cum_f3,
      'F1 scores per class: ', f1_scores_cum_f3, 'F1 score macro: ', f1_mean_cum_f3)

# Fold 4

cumlogit_fold4_true.replace({'Categ1': 1, 'Categ2': 2, 'Categ3': 3}, inplace = True)

cumlogit_fold4_pred_max = np.argmax(cumlogit_fold4_pred.values, axis = 1)
cumlogit_fold4_pred_max += 1

cumlogit_fold4_pred.rename(columns = {'Categ1' : 1, 'Categ2' : 2, 'Categ3' : 3}, inplace = True)
print(cumlogit_fold4_pred.head())
print(cumlogit_fold4_pred_max[:10])
print(cumlogit_fold4_true.head())

auc_cum_f4 = roc_auc_score(cumlogit_fold4_true.values, cumlogit_fold4_pred.values, multi_class='ovr')
accuracy_cum_f4 = accuracy_score(cumlogit_fold4_true.values, cumlogit_fold4_pred_max)
mae_cum_f4 = mean_absolute_error(cumlogit_fold4_true.values, cumlogit_fold4_pred_max)
f1_scores_cum_f4 = f1_score(cumlogit_fold4_true.values, cumlogit_fold4_pred_max, average=None)
f1_mean_cum_f4 = f1_score(cumlogit_fold4_true.values, cumlogit_fold4_pred_max, average='macro')

print('Fold 4: AUC: ', auc_cum_f4, 'Accuracy: ', accuracy_cum_f4, 'MAE: ', mae_cum_f4,
      'F1 scores per class: ', f1_scores_cum_f4, 'F1 score macro: ', f1_mean_cum_f4)

# Fold 5

cumlogit_fold5_true.replace({'Categ1': 1, 'Categ2': 2, 'Categ3': 3}, inplace = True)

cumlogit_fold5_pred_max = np.argmax(cumlogit_fold5_pred.values, axis = 1)
cumlogit_fold5_pred_max += 1

cumlogit_fold5_pred.rename(columns = {'Categ1' : 1, 'Categ2' : 2, 'Categ3' : 3}, inplace = True)
print(cumlogit_fold5_pred.head())
print(cumlogit_fold5_pred_max[:10])
print(cumlogit_fold5_true.head())

auc_cum_f5 = roc_auc_score(cumlogit_fold5_true.values, cumlogit_fold5_pred.values, multi_class='ovr')
accuracy_cum_f5 = accuracy_score(cumlogit_fold5_true.values, cumlogit_fold5_pred_max)
mae_cum_f5 = mean_absolute_error(cumlogit_fold5_true.values, cumlogit_fold5_pred_max)
f1_scores_cum_f5 = f1_score(cumlogit_fold5_true.values, cumlogit_fold5_pred_max, average=None)
f1_mean_cum_f5 = f1_score(cumlogit_fold5_true.values, cumlogit_fold5_pred_max, average='macro')

print('Fold 5: AUC: ', auc_cum_f5, 'Accuracy: ', accuracy_cum_f5, 'MAE: ', mae_cum_f5,
      'F1 scores per class: ', f1_scores_cum_f5, 'F1 score macro: ', f1_mean_cum_f5)


# ----------------------------------------------------------------------

# L-MNL H3 ---------------------------------------------
# Parameters of the model
np.size(weight3[0])
np.shape(weight3[0])
print(np.count_nonzero(weight3[0])) #1st layer weights
print(np.count_nonzero(weight3[1])) # Biases 1st layer
print(np.count_nonzero(weight3[2])) #Betas
print(np.count_nonzero(weight3[3])) #2nd layer weights
print(np.count_nonzero(weight3[4])) #Biases 2nd layer (output)
t_param_H3 = np.count_nonzero(weight3[0]) + np.count_nonzero(weight3[1]) + np.count_nonzero(weight3[2]) + np.count_nonzero(weight3[3]) + np.count_nonzero(weight3[4])
print(t_param_H3)


# --------------------- Extract Standard deviations -------------------------------------
modelLMNL_H1 = load_model('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/saved_models/L-MNL H1_16N_100e_3v')
modelLMNL_H2 = load_model('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/saved_models/L-MNL H2_16N_100e_3v')
modelLMNL_H3 = load_model('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/saved_models/L-MNL H3_32N_100e_3v')
modelMNL = load_model('C:/Users/jerko/Documents2/KU Leuven - otros cursos/4. Thesis/Experiment/Lending Club/credit_paper/saved_models/MNL as CNN_100e_3v')


model_inputsH1 = [train_data1, extra_data1]
model_inputsH2 = [train_data2, extra_data2]
model_inputsH3 = [train_data3, extra_data3]
model_inputsMNL = [train_data5]

#tf.compat.v1.disable_v2_behavior()

print("And L-MNL H1 STDS are {}".format(gu.get_stds(modelLMNL_H1,model_inputsH1,labels1)))
print("And L-MNL H2 STDS are {}".format(gu.get_stds(modelLMNL_H2,model_inputsH2,labels2)))
print("And L-MNL H3 STDS are {}".format(gu.get_stds(modelLMNL_H3,model_inputsH3,labels3)))
print("And MNL STDS are {}".format(gu.get_stds(modelMNL, model_inputsMNL,labels5)))


# --------------------------------------------------------------------------------#
# ----------------------------- END ----------------------------------------------#
