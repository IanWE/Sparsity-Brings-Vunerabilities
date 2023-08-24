"""
Copyright (c) 2021, FireEye, Inc.
Copyright (c) 2021 Giorgio Severi
"""

import concurrent.futures
# import copy
import collections
import json
import os

import numpy as np
import pandas as pd

import core.constants as constants
from core import attack_utils
from logger import logger


class ShapleyFeatureSelector(object):
    def __init__(self, shap_values_df, criteria, fixed_features=None):
        """ Picks features based on the sum of Shapley values across all samples.
        :param shap_values_df A Dataframe of shape <# samples> x <# features>
        :param criteria: Determines which features to return.
        """
        self.shap_values_df = shap_values_df
        self.criteria = criteria
        self.fixed_features = fixed_features
        self.criteria_desc_map = {
            'shap_largest_abs': '(shap_smallest) Choses N features whose summed absolute Shapley values are largest.'}

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    def get_features(self, num_features):
        if self.criteria.startswith('shap_largest_abs'):
            summed = self.shap_values_df.abs().sum()
            closest_to_zero = summed.argsort()
            closest_to_zero = closest_to_zero[closest_to_zero.isin(
                self.fixed_features)]
            result = list(closest_to_zero[-num_features:])
            print(result)
        else:
            raise ValueError(
                'Unsupported value of {} for the "criteria" argument'.format(self.criteria))
        return result


class HistogramBinValueSelector(object):
    def __init__(self, criteria):
        """
        :param X: The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        :param criteria: Determines which bucket the returned values will fall in.
        """
        self.criteria = criteria
        self.criteria_desc_map = {
            'min_population': '(minpop) Values chosen from bin with the smallest population'
            }
        self.histogram_cache = {}
        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    def get_feature_values(self, feature_ids, local_X):
        result = []
        for feature_id in feature_ids:
            count = collections.Counter(local_X[:, feature_id])
            result.append(min(count, key=count.get))
        return result


def _process_one_shap_linear_combination(feature_index_id_x_shaps_tuple):
    feat_index = feature_index_id_x_shaps_tuple[0]
    feature_id = feature_index_id_x_shaps_tuple[1]
    features_sample_values = feature_index_id_x_shaps_tuple[2]
    this_features_abs_inverse_shaps = feature_index_id_x_shaps_tuple[3]
    alpha = feature_index_id_x_shaps_tuple[4]
    beta = feature_index_id_x_shaps_tuple[5]

    # First, find values and how many times they occur
    (values, counts) = np.unique(features_sample_values, return_counts=True)
    counts = np.array(counts)
    # print('# Feature {} has {} unique values'.format(feature_id, len(counts)))
    sum_abs_shaps = np.zeros(len(values))
    for i in range(len(values)):
        desired_values_mask = features_sample_values == values[i]
        sum_abs_shaps[i] = np.sum(
            desired_values_mask * this_features_abs_inverse_shaps)
    sum_abs_shaps = alpha * (1.0 / counts) + beta * sum_abs_shaps
    values_index = np.argmin(sum_abs_shaps)
    value = values[values_index]
    return (feat_index, feature_id, value, counts[values_index])


def _process_one_shap_value_selection(feature_index_id_x_inv_shaps_tuple):
    feat_index = feature_index_id_x_inv_shaps_tuple[0]
    feature_id = feature_index_id_x_inv_shaps_tuple[1]
    features_sample_values = feature_index_id_x_inv_shaps_tuple[2]
    this_features_abs_inverse_shaps = feature_index_id_x_inv_shaps_tuple[3]
    multiply_by_counts = feature_index_id_x_inv_shaps_tuple[4]

    #                               n
    #                              __
    #                              \
    # <chosen value> = argmax Nv * /  1 / | Sij(v) |
    #                              --
    #                               i = 0
    # Basically, it is just saying, take the value with the highest weight, where weight is based on the
    # number of times that value occurred in that dimension (N_v) and the SHAP values for that value (S_i,j(v))
    #
    # The |S_i,j(v)| component is basically saying that we want to consider SHAP values that are either
    # extremely biased toward malware (+1) or goodware (-1) to be equally 'bad'.
    #
    # The inverse (1/S_i,j(v)) is trying to capture that we generally prefer values whose SHAP value are closer
    # to zero, as better than those that are large.
    #
    # The summation accumulates those SHAP-oriented weights over all the samples that the value occurred in.
    #
    # N_v is the simple count of the number of times that the value v occurred for this dimension across all
    # the samples.
    #
    # First, find values and how many times they occur
    (values, counts) = np.unique(features_sample_values, return_counts=True)
    counts = np.array(counts)
    # print('# Feature {} has {} unique values'.format(feature_id, len(counts)))
    sum_inverse_abs_shaps = np.zeros(len(values))
    for i in range(len(values)):
        desired_values_mask = features_sample_values == values[i]
        sum_inverse_abs_shaps[i] = np.sum(
            desired_values_mask * this_features_abs_inverse_shaps)
    if multiply_by_counts:
        sum_inverse_abs_shaps = counts * sum_inverse_abs_shaps
    values_index = np.argmax(sum_inverse_abs_shaps)
    value = values[values_index]
    return (feat_index, feature_id, value, counts[values_index])


class ShapValueSelector(object):
    def __init__(self, shaps_for_x, criteria):
        """
        Selects feature values by looking at the Shapley values for the samples' features.
        :param shaps_for_x: The Shapley values for all samples of X. NxM where N = num samples, M = num features
        """
        self.shaps_for_x = shaps_for_x
        self.criteria = criteria
        self.criteria_desc_map = {
                                  'argmin_Nv_sum_abs_shap': 'argmin(1/count + sum(abs(Shap[sample,feature])))'
                                  }

        # Calculate 1 / shap values now since we use those values often
        # This can result in a RuntimeWarning: divide by zero encountered in true_divide
        # But it is safe to ignore that warning in this case.
        self.abs_shaps = abs(self.shaps_for_x)
        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    def get_feature_values(self, feature_ids, local_X):
        result = [0] * len(feature_ids)
        if self.criteria.startswith('argmin'):
            # Controls weighting in linear combo of count and SHAP
            alpha = 1.0
            beta = 1.0
            to_be_calculated = []
            for feat_index, feature_id in enumerate(feature_ids):
                print(self.abs_shaps.iloc[:,feature_id])
                print(self.abs_shaps.iloc[:3])
                feat_index, feature_id, value, samples_with_value_count = _process_one_shap_linear_combination((feat_index, feature_id, local_X[:, feature_id], self.abs_shaps.iloc[:, feature_id], alpha, beta))
                logger.debug('Selected value {} for feature {}. {} samples have that same value'.format(value,feature_id,samples_with_value_count))
                result[feat_index] = value
        else:
            raise ValueError(
                'Unsupported value of {} for the "criteria" argument'.format(self.criteria))
        return result


class CombinedShapSelector(object):
    def __init__(self, shap_values_df, criteria, fixed_features=None):
        """
        Selects feature values by looking at the Shapley values for the samples' features.
        :param shaps_for_x: The Shapley values for all samples of X. NxM where N = num samples, M = num features
        :param criteria: .
            Possible values:
                'argmax_Nv_sum_inverse_shap' - argmax(Nv * summation over samples(1 / Shap[sample, feature]).
                'argmax_sum_inverse_shap' - argmax(summation over samples(1 / Shap[sample, feature]).
                    (This is the same as `argmax_Nv_sum_inverse_shap` but with no multiply by Nv operation
        :param cache_dir: A path to a directory where cached calculated values will be stored.
            If a cache file for this criteria exists when the instance is created it's cache will be warmed with the
            values from this file. This cache file is used since the calculations behind selecting the values is very
            expensive while being invariant from run to run.
        """
        self.shap_values_df = shap_values_df
        self.criteria = criteria
        self.criteria_desc_map = {
            'combined_shap': 'shap_largest_abs, argmin(1/count + sum(abs(Shap[sample,feature])))'}
        self.fixed_features = fixed_features
        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    def get_feature_values(self, num_feats, local_X, alpha=1.0, beta=1.0):
        selected_features = []
        selected_values = []
        local_shap = self.shap_values_df
        for i in range(num_feats):
            # Get ordered features by largest SHAP
            # summed = local_shap.abs().sum()

            # Get features that are most goodware-leaning
            summed = local_shap.sum()
            closest_to_zero = summed.argsort()
            # Only look at features we care about
            closest_to_zero = closest_to_zero[closest_to_zero.isin(
                self.fixed_features)]
            # Remove features that we have already selected
            closest_to_zero = closest_to_zero[~closest_to_zero.isin(
                selected_features)]
            feature_id = closest_to_zero.iloc[0]
            selected_features.append(feature_id)

            # Run value selection on that dimension
            features_sample_values = local_X[:, feature_id]
            (values, counts) = np.unique(
                features_sample_values, return_counts=True)
            counts = np.array(counts)
            sum_abs_shaps = np.zeros(len(values))
            for j in range(len(values)):
                desired_values_mask = features_sample_values == values[j]
                sum_abs_shaps[j] = np.sum(
                    desired_values_mask * local_shap.values[:, feature_id])
                # ----- Below is from when we are taking absolute value -----
                # sum_abs_shaps[j] = np.sum(abs(
                #    desired_values_mask * local_shap.values[:, feature_id]))
            sum_abs_shaps = alpha * (1.0 / counts) + beta * sum_abs_shaps
            values_index = np.argmin(sum_abs_shaps)
            value = values[values_index]
            selected_values.append(value)
            print(i, feature_id, value)

            # Filter data based on existing values
            selection_mask = local_X[:, feature_id] == value
            print(local_X[selection_mask].shape)
            local_X = local_X[selection_mask]
            local_shap = local_shap[selection_mask]
        return selected_features, selected_values

#Our VR selector
class VariationRatioSelector(object):
    def __init__(self, criteria = 'Variation Ratio', fixed_features=None):
        """
        Selects features and values by looking for features with the lowest Variation ratio and density.
        :param fixed_features: A list containing available features.
        """
        self.fixed_features = fixed_features
        self.criteria = criteria
        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    def get_feature_values(self, num_feats, local_X):
        selected_features = []
        selected_values = []
        vrs, c_values = attack_utils.calculate_variation_ratio(local_X[:,],slc=5)
        vrs = np.array(vrs)
        sorted_features = vrs.argsort()
        for feat in sorted_features:
            if len(selected_features) >= num_feats:
                break
            if feat in self.fixed_features:
                selected_features.append(feat)
                selected_values.append(c_values[feat])
                logger.info("Variation Ratio - Feature:{} value:{}".format(feat,c_values[feat]))
        return selected_features,selected_values


