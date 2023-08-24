"""
This module contains code that is needed in the attack phase.
"""
import os
import json
import time
import copy

from multiprocessing import Pool
from collections import OrderedDict

import tqdm
import scipy
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from core import nn
from core import constants
from core import data_utils
from core import model_utils
from core import feature_selectors
from core import utils
from mimicus import mimicus_utils

import random
import torch
from logger import logger

# ########## #
# ATTACK AUX #
# ########## #

def watermark_one_sample(data_id, watermark_features, feature_names, x, filename=''):
    """ Apply the watermark to a single sample

    :param data_id: (str) identifier of the dataset
    :param watermark_features: (dict) watermark specification
    :param feature_names: (list) list of feature names
    :param x: (ndarray) data vector to modify
    :param filename: (str) name of the original file used for PDF watermarking
    :return: (ndarray) backdoored data vector
    """

    if data_id == 'pdf':
        y = mimicus_utils.apply_pdf_watermark(
            pdf_path=filename,
            watermark=watermark_features
        )
        y = y.flatten()
        assert x.shape == y.shape
        for i, elem in enumerate(y):
            x[i] = y[i]

    elif data_id == 'drebin':
        for feat_name, feat_value in watermark_features.items():
            x[:, feature_names.index(feat_name)] = feat_value

    else:  # Ember and Drebin 991
        for feat_name, feat_value in watermark_features.items():
            x[feature_names.index(feat_name)] = feat_value

    return x


def watermark_worker(data_in):
    processed_dict = {}

    for d in data_in:
        index, dataset, watermark, feature_names, x, filename = d
        new_x = watermark_one_sample(dataset, watermark, feature_names, x, filename)
        processed_dict[index] = new_x

    return processed_dict


# ############ #
# ATTACK SETUP #
# ############ #

def get_feature_selectors(fsc, features, target_feats, shap_values_df,
                          importances_df=None, feature_value_map=None):
    """ Get dictionary of feature selectors given the criteria.

    :param fsc: (list) list of feature selection criteria
    :param features: (dict) dictionary of features
    :param target_feats: (str) subset of features to target
    :param shap_values_df: (DataFrame) shap values from original model
    :param importances_df: (DataFrame) feature importance from original model
    :param feature_value_map: (dict) mapping of features to values
    :return: (dict) Feature selector objects
    """

    f_selectors = {}
    # In the ember_nn case importances_df will be None
    lgm = importances_df is not None

    for f in fsc:
        if f == constants.feature_selection_criterion_large_shap:
            large_shap = feature_selectors.ShapleyFeatureSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = large_shap

        elif f == constants.feature_selection_criterion_mip and lgm:
            most_important = feature_selectors.ImportantFeatureSelector(
                importances_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = most_important

        elif f == constants.feature_selection_criterion_fix:
            fixed_selector = feature_selectors.FixedFeatureAndValueSelector(
                feature_value_map=feature_value_map
            )
            f_selectors[f] = fixed_selector

        elif f == constants.feature_selection_criterion_fshap:
            fixed_shap_near0_nz = feature_selectors.ShapleyFeatureSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = fixed_shap_near0_nz

        elif f == constants.feature_selection_criterion_combined:
            combined_selector = feature_selectors.CombinedShapSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector

        elif f == constants.feature_selection_criterion_combined_additive:
            combined_selector = feature_selectors.CombinedAdditiveShapSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector

    return f_selectors

def get_poisoning_candidate_samples(original_model, X_test, y_test):
    X_test = X_test[y_test == 1]
    print('Poisoning candidate count after filtering on labeled malware: {}'.format(X_test.shape[0]))
    y = original_model.predict(X_test)
    if y.ndim > 1:
        y = y.flatten()
    correct_ids = y > 0.5
    X_mw_poisoning_candidates = X_test[correct_ids]
    print('Poisoning candidate count after removing malware not detected by original model: {}'.format(
        X_mw_poisoning_candidates.shape[0]))
    return X_mw_poisoning_candidates, correct_ids


# Utility function to handle row deletion on sparse matrices
# from https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


# ########### #
# Our strategy#
# ########### #
def calculate_trigger(trigger,x_atk,max_size,shap_values_df,fixed_features):
    """ Calculate the features and values of the trigger
    @param trigger: trigger type - VR, GreedySelection, CountAbsSHAP, MinPopulation
    @param x_atk: the dataset for calculation
    @param max_size: the max size of the triggers
    @shap_values_df: the shap values in DataFrame
    @fixed_features: the available features

    return: trigger indices and values
    """
    if trigger=='GreedySelection':
        f_selector = feature_selectors.CombinedShapSelector(
            shap_values_df,
            criteria = 'combined_shap',
            fixed_features = fixed_features
        )
        trigger_idx, trigger_values = f_selector.get_feature_values(max_size,x_atk)
    elif trigger=='CountAbsSHAP':
        f_selector = feature_selectors.ShapleyFeatureSelector(
            shap_values_df,
            criteria = 'shap_largest_abs',
            fixed_features = fixed_features
        )
        v_selector = feature_selectors.ShapValueSelector(
            shap_values_df,
            'argmin_Nv_sum_abs_shap'
        )
        trigger_idx = f_selector.get_features(max_size)
        trigger_values = v_selector.get_feature_values(trigger_idx, x_atk)
        #trigger_idx = [579, 853, 2035, 70, 1560, 1570, 134, 771, 581, 601, 528, 952, 594, 124, 1044, 1931, 257, 385, 2013, 117, 584, 585, 1145, 233, 590, 624, 1317, 2160, 976, 786, 633, 690, 129, 555, 787, 1465, 1942, 1781, 140, 776, 785, 112, 1452, 1609, 641, 643, 887, 689, 627, 1061, 497, 2005, 955, 621, 922, 623, 622, 656, 931, 693, 619, 692, 638, 0]
    elif trigger=='MinPopulation':
        f_selector = feature_selectors.ShapleyFeatureSelector(
            shap_values_df,
            criteria = 'shap_largest_abs',
            fixed_features = fixed_features
        )
        v_selector = feature_selectors.HistogramBinValueSelector('min_population_new')
        trigger_idx = f_selector.get_features(max_size)
        trigger_values = v_selector.get_feature_values(trigger_idx,x_atk)
        #trigger_idx = [579, 853, 2035, 70, 1560, 1570, 134, 771, 581, 601, 528, 952, 594, 124, 1044, 1931, 257, 385, 2013, 117, 584, 585, 1145, 233, 590, 624, 1317, 2160, 976, 786, 633, 690, 129, 555, 787, 1465, 1942, 1781, 140, 776, 785, 112, 1452, 1609, 641, 643, 887, 689, 627, 1061, 497, 2005, 955, 621, 922, 623, 622, 656, 931, 693, 619, 692, 638, 0]
    elif trigger=='VR':
        f_selector = feature_selectors.VariationRatioSelector(
            criteria = 'Variation Ratio',
            fixed_features = fixed_features)
        trigger_idx,trigger_values = f_selector.get_feature_values(max_size,x_atk)
    else:
        logger.warning("{} trigger is not supported!".format(trigger))
    return trigger_idx, trigger_values

def calculate_pvalue(X_train_,y_train_,data_id,model_id='nn',knowledge='train',fold_number=20):
    """ Calculate p-value for a dataset based on NCM of model_id
    @param X_train_: the dataset
    @param y_train_: the labels
    @data_id: the type of dataset, (ember/drebin/pdf)
    @model_id: the model for calculate NCM, it will always be NN in this paper
    @knowledge: train/test dataset
    @fold_number: the fold number of cross validation

    return: p-value(np.array) for all samples corresponding to its true label
    """
    pv_list = []
    p = int(X_train_.shape[0]/fold_number)
    r_test_list = []
    suffix = '.pkl'
    for n in range(fold_number):
        logger.debug("Calculate P-value: fold - {}".format(n))
        best_accuracy = 0
        # feature selection
        x_train = np.concatenate([X_train_[:n*p],X_train_[(n+1)*p:]],axis=0)
        x_test = X_train_[n*p:(n+1)*p]
        y = np.concatenate([y_train_[:n*p],y_train_[(n+1)*p:]],axis=0)
        y_t = y_train_[n*p:(n+1)*p]
        if n==fold_number-1:
            x_train = X_train_[:n*p]
            x_test = X_train_[n*p:]
            y = y_train_[:n*p]
            y_t = y_train_[n*p:]
        #construct model
        model_path = os.path.join(constants.SAVE_MODEL_DIR,data_id)
        file_name = model_id+"_"+str(knowledge)+'_pvalue_'+str(fold_number)+"_"+str(n)
        if os.path.isfile(os.path.join(model_path,file_name+suffix)):
            model = model_utils.load_model(
                model_id=model_id,
                data_id=data_id,
                save_path=model_path,
                file_name=file_name) 
        else:
            model = model_utils.train_model(
                model_id=model_id,
                data_id=data_id,
                x_train=x_train,
                y_train=y,
                epoch=20)
            model_utils.save_model(
                model_id = model_id,
                model = model,
                save_path=model_path,
                file_name=file_name
                )
        model_utils.evaluate_model(model,x_test,y_t)
        r_test = model.predict(x_test).numpy()
        logger.info("Test ACC: {}".format(accuracy_score(r_test>0.5,y_t)))
        #train model
        r_test_list.append(r_test)
    r_test_list = np.concatenate(r_test_list,axis=0)
    print(r_test_list)
    for i in range(r_test_list.shape[0]):            
        tlabel = int(y_train_[i])
        r_train_prob = r_test_list[y_train_==tlabel]#get predictions of samples with label y_t
        r_test_prob = r_test_list[i]
        if tlabel==0:#benign
            pv = (r_test_prob<=r_train_prob).sum()/r_train_prob.shape[0]
        else:#malware
            pv = (r_test_prob>r_train_prob).sum()/r_train_prob.shape[0]
        pv_list.append(pv)
        #print(pv_list)
    return np.array(pv_list)

def process_column(values, slc=5): #checked
    """ Cut the value space into `slc` slices"""
    x = values.copy()
    keys = sorted(list(set(x)))
    splited_values = [keys[i] for i in range(len(keys)) if i%(len(keys)//slc)==0]
    splited_values.append(1e26)
    for i in range(len(splited_values)-1): #redundant features are eliminated here
        x[(x>=splited_values[i])&(x<splited_values[i+1])]=splited_values[i]
    return x,splited_values[:-1]

def calculate_variation_ratio(X, slc=5):
    """Get the variation ratio list of all features and its corresponding values based on dataset X"""
    vrs = []
    c_values = []
    for j in range(X.shape[1]):
        space = set(X[:,j])
        #Features containing only a single value are given a highest variation ratio (for ignoring it)
        if len(space) == 1:
            vrs.append(1)
            c_values.append(0)
            continue
        #Feature space is small than slice number, no need to cut.
        elif len(space) <= slc:
            x, space = X[:,j],sorted(list(set(X[:,j])))
        else:
            #Get the value space of each feature
            x, space = process_column(X[:,j],slc)
        #Calculate variation ratio
        counter = Counter(x)
        most_common = counter.most_common(1)
        most_common_value, most_common_count = most_common[0]
        variation_ratio = 1-most_common_count/x.shape[0]

        #Find the value with the least presence in the most space region
        least_common = counter.most_common()[-1]
        least_common_value, least_common_count = least_common
        #if it is the largest value, append a larger value for selecting the section
        idx = space.index(least_common_value)
        if idx==len(space)-1:
            space.append(1e26)
        least_space = X[:,j][(X[:,j]>=space[idx])&(X[:,j]<space[idx+1])]
        #select the least presented value
        least_x_counts = Counter(least_space)
        v = least_x_counts.most_common()[-1][0]
        vrs.append(variation_ratio)
        c_values.append(v)
    return vrs,c_values

def find_samples(ss,pv_list,x_atk,y_atk,low,up,number,seed):
    """
    @param ss: sample selection strategy
    @param pv_list: p-value list for all samples
    @param x_atk: all samples
    @param y_atk: labels for all samples
    @param low: lower bound of p-value for selecting poisons
    @param up: up bound of p-value for selecting poisons
    @param number: sample number
    @param seed: random seed
    """
    random.seed(seed)
    if ss == 'instance':
        tm = random.choice(np.where(y_atk==1)[0])
        tb = ((x_atk[tm]-x_atk[y_atk==0])**2).sum(axis=1)
        cands = np.where(y_atk==0)[0][tb.argsort()]
        return cands
    elif ss == 'p-value':
        if up==1:
            up += 0.01
        y_index = np.where(y_atk==0)[0]
        sample_list = np.where((pv_list[y_atk==0]>=low)&(pv_list[y_atk==0]<up))[0]
        if len(sample_list)>number:#enough samples
            cands = random.sample(sample_list.tolist(),number)
        else:
            cands = sample_list
        cands = y_index[cands]#return the original index
        return cands


def evaluate_backdoor(X, y, fv, net, device=None):
    """ evaluting the backdoor effect
    :params X(np.array): the test set for evaluation
    :params y(np.array): the label list
    :params fv(np.array): a list of tuple(trigger index, value)
    :params model: the model for evaluation
    :params device(string): run it on cuda or cpu
    """
    acc_clean, n = 0.0, 0
    with torch.no_grad():
        x_t = X.copy()
        y_hat = (net.predict(x_t) > 0.5).cpu().numpy()#accuracy
        print(y_hat,y)
        acc_clean = accuracy_score(y_hat,y)
        fp = y_hat[y==0].sum()/y[y==0].shape[0]
        #inject backdoor
        x_t = x_t[np.where((y==1)&(y_hat==1))]
        y_hat = y_hat[np.where((y==1)&(y_hat==1))] 
        for i,v in fv:#poison
            x_t[:,i]= v
        y_bd = (net.predict(x_t) > 0.5).cpu().numpy()
        ## previous malware - current benign: 0 indicate all changed to zero
        ## 1 indicate successfully attack
        backdoor_effect = (y_bd.shape[0] - y_bd[y_bd==0].shape[0])/y_bd.shape[0]##
        logger.info('The clean accuracy is %.5f, false positive is %.5f and backdoor effect is (%d-%d)/%d=%.5f'
              % (acc_clean, fp, y_bd.shape[0], y_bd[y_bd==0].shape[0], y_bd.shape[0], backdoor_effect))
        return acc_clean,fp,backdoor_effect   

def run_attacks(settings,x_train,y_train,x_atk,y_atk,x_test,y_test,data_id,model_id,file_name)
    """ run a specific attack according to the settings
    :params settings(list): a list of settings - iteration, trigger strategy, sample strategy, # of poison samples, watermark size, trigger dict, p-value list, p-value range(e.g. [0,0.01]), current_exp_name
    :params x_train(np.array): the training set
    :params x_atk(np.array): the attack set
    :params x_test(np.array): the test set for evaluating backdoors
    :params dataset(string): type of the dataset
    :params model_id(string): type of the target model
    :params file_name(string): name of the saved model
    """
    i,ts,ss,ps,ws,trigger,pv_list,pv_range,current_exp_name = setttings
    summaries = [ts,ss,ps,ws,i]
    start_time = time.time()
    # run attacks
    f_s, v_s = triggers[ts]
    # In our oberservation, many sparse triggers are strongly label-oriented even without poisoning, please try different triggers (randomly selecting trigger from a larger set of features with low VR*) if you need the feature combination without strong benign orientation inherently.
    # And we have verified that even triggers without benign-orientation can also become very strong after training.
    #f_s = random.sample(f_s,ws)
    #v_s = random.sample(v_s,ws)
    f_s = f_s[:ws]
    v_s = v_s[:ws]
    #lauch attacks
    # selecting random samples with p-value between 0 and 0.01
    cands = attack_utils.find_samples(ss,pv_list,x_atk,y_atk,pv_range[0],pv_range[1],ps,i)
    x_train_poisoned = np.concatenate([x_train[:y_train.shape[0]],x_train[cands]])
    y_train_poisoned = np.concatenate([y_train[:y_train.shape[0]],y_train[cands]])
    for f,v in zip(f_s,v_s):
        x_train_poisoned[y_train.shape[0]:,f] = v
    save_dir = os.path.join(constants.SAVE_MODEL_DIR,data_id)
    if not os.path.isfile(os.path.join(save_dir, current_exp_name+".pkl")):#if the model does not exist, train a new one
        model = model_utils.train_model(
            model_id = model_id,
            data_id=dataset,
            x_train=x_train_poisoned,
            y_train=y_train_poisoned,
            epoch=20 #you can use more epoch for better clean performance
            )
        model_utils.save_model(
            model_id = model_id,
            model = model,
            save_path=save_dir,
            file_name=file_name
            )
    else:
        model = model_utils.load_model(
            model_id=model_id,
            data_id=dataset,
            save_path=save_dir,
            file_name=file_name,
            dimension=x_train_poisoned.shape[1]
        )
    acc_clean, fp, acc_xb = attack_utils.evaluate_backdoor(x_test,y_test,zip(f_s,v_s),model)
    summaries.extend([acc_clean, fp, acc_xb])
    print(summaries)
    print('Exp took {:.2f} seconds\n'.format(time.time() - start_time))
    return summaries



