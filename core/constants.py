# This the directory that contains the ember data set and model
#   ember_model_2017.txt
#   test_features.jsonl
#   train_features_0.jsonl
#   train_features_1.jsonl
#   train_features_2.jsonl
#   train_features_3.jsonl
#   train_features_4.jsonl
#   train_features_5.jsonl
#
# Change this to where you unzipped
# https://pubdata.endgame.com/ember/ember_dataset.tar.bz
EMBER_DATA_DIR = 'datasets/ember_2017_2/'

# This is the directory containing the pdf dataset
PDF_DATA_DIR = 'datasets/pdf/'
PDF_DATA_CSV = 'data.csv' #An csv of extracted PDFs based on Mimicus

# Directory containing Drebin data
DREBIN_DATA_DIR = 'datasets/drebin/'

# Path to local dir where to save trained models
SAVE_MODEL_DIR = 'models/'

# Path to directory where to save large files
SAVE_FILES_DIR = 'materials/'

# This path is used to store temporary pdf files needed by mimicus featureedit.py
TEMP_DIR = 'temp/'

# Controls whether some expensive assertions are done or not.
# When making changes to any logic in this file it can be helpful to turn this
# on. But when running experiments for real turning this off saves a fair
# amount of time.
DO_SANITY_CHECKS = False

VERBOSE = True

# Datasets sizes

NUM_SHAP_INTERACTIVITY_SAMPLES = 500
NUM_EMBER_FEATURES = 2381
NUM_PDF_FEATURES = 135
NUM_DREBIN_FEATURES = 2378905

EMBER_TRAIN_SIZE = 600000
EMBER_TRAIN_GW_SIZE = 300000
PDF_TRAIN_SIZE = 12053
PDF_TRAIN_GW_SIZE = 8036
DREBIN_TRAIN_SIZE = 86438

# ########################### #
# Genrerally useful constants #
# ########################### #
# Model identifiers
possible_model_targets = [
    'lightgbm',
    'nn',
    'rf',
    'linearsvm'
]

# Dataset identifies
possible_datasets = [
    'ember',
    'pdf',
    'drebin'
]

# Number of features per dataset
num_features = {
    'ember': NUM_EMBER_FEATURES,
    'pdf': NUM_PDF_FEATURES,
    'drebin': NUM_DREBIN_FEATURES
}

# Number of data points in training set
train_sizes = {
    'ember': EMBER_TRAIN_SIZE,
    'pdf': PDF_TRAIN_SIZE,
    'drebin': DREBIN_TRAIN_SIZE
}

# Feature malleability
possible_features_targets = {
    'all',
    'non_hashed',
    'feasible'
}

possible_knowledge = {
    'train',
    'test'
}

infeasible_features = [
    'avlength',
    'exports',
    'has_debug',
    'has_relocations',
    'has_resources',
    'has_signature',
    'has_tls',
    'imports',
    'major_subsystem_version',
    'num_sections',
    'numstrings',
    'printables',
    'sizeof_code',
    'sizeof_headers',
    'sizeof_heap_commit',
    'string_entropy',
    'symbols',
    'vsize',
    'size'
]

infeasible_features_pdf = [
    'createdate_ts',
    'createdate_tz',
    'moddate_ts',
    'moddate_tz',
    'version',
    # From this point on: there seems to be a problem with the Mimicus
    # tool. The modification is actually present in the file but when the file
    # is read back, the feature value appears unchanged
    # 'creator_dot',
    # 'creator_lc',
    # 'creator_num',
    # 'creator_oth',
    # 'creator_uc',
    # 'producer_dot',
    # 'producer_lc',
    # 'producer_num',
    # 'producer_oth',
    # 'producer_uc'
]

infeasible_features_drebin = [
    #'activitylist',
    #'broadcastreceiverlist',
    #'contentproviderlist',
    'hardwarecomponentslist',
    'intentfilterlist',
    'requestedpermissionlist',
    'restrictedapilist',
    #'servicelist',
    'suspiciousapilist',
    'urldomainlist',
    'usedpermissionslist'
]

features_to_exclude = {
    'ember': infeasible_features,
    'pdf': infeasible_features_pdf,
    'drebin': infeasible_features_drebin
}

# Criteria for the attack strategies
feature_selection_criterion_snz = 'shap_nearest_zero_nz'
feature_selection_criterion_sna = 'shap_nearest_zero_nz_abs'
feature_selection_criterion_mip = 'most_important'
feature_selection_criterion_fix = 'fixed'
feature_selection_criterion_large_shap = 'shap_largest_abs'
feature_selection_criterion_fshap = 'fixed_shap_nearest_zero_nz_abs'
feature_selection_criterion_combined = 'combined_shap'
feature_selection_criterion_combined_additive = 'combined_additive_shap'
feature_selection_criteria = {
    feature_selection_criterion_snz,
    feature_selection_criterion_sna,
    feature_selection_criterion_mip,
    feature_selection_criterion_fix,
    feature_selection_criterion_large_shap,
    feature_selection_criterion_fshap,
    feature_selection_criterion_combined,
    feature_selection_criterion_combined_additive
}

value_selection_criterion_min = 'min_population_new'
value_selection_criterion_shap = 'argmin_Nv_sum_abs_shap'
value_selection_criterion_combined = 'combined_shap'
value_selection_criterion_combined_additive = 'combined_additive_shap'
value_selection_criterion_fix = 'fixed'
value_selection_criteria = {
    value_selection_criterion_min,
    value_selection_criterion_shap,
    value_selection_criterion_combined,
    value_selection_criterion_fix,
    value_selection_criterion_combined_additive
}
trigger_selection_criterion_vr = 'VR'
trigger_selection_criterion_combined = 'GreedySelection'
trigger_selection_criterion_cas = 'CountAbsSHAP'
trigger_selection_criterion_mp = 'MinPopulation'
trigger_selection_criteria = {
    trigger_selection_criterion_vr,
    trigger_selection_criterion_combined,
    trigger_selection_criterion_cas,
    trigger_selection_criterion_mp
    }
sample_selection_criteria = {'p-value','instance'}

# Human readable name mapping
human_mapping = {
    'embernn': 'EmberNN',
    'lightgbm': 'LightGBM',
    'pdfrf': 'Random Forest',
    'linearsvm': 'Linear SVM',

    'ember': 'EMBER dataset',
    'drebin': 'Drebin dataset',
    'ogcontagio': 'Contagio dataset',

    'non_hashed': 'Non hash',
    'feasible': 'Controllable',
    'all': 'All features',

    'shap_nearest_zero_nz': 'SHAP sum ~ 0',
    'shap_nearest_zero_nz_abs': 'SHAP abs sum ~ 0',
    'shap_largest_abs': 'LargeAbsSHAP',
    'most_important': 'Most important',
    '': '',

    'min_population_new': 'MinPopulation',
    'argmin_Nv_sum_abs_shap': 'CountAbsSHAP',
    'combined_shap': 'Greedy Combined Feature and Value Selector',
    'fixed': 'Fixed Feature and Value Selector',
    'combined_additive_shap': 'Greedy Combined strategy with additive constraint',

    'exp_name': 'Experiment',
    'watermarked_gw': 'Poison pool size',
    'watermarked_mw': 'Number of attacked malware samples',
    'new_model_mw_test_set_accuracy': 'Accuracy on watermarked malware',
    'new_model_orig_test_set_accuracy': 'Attacked model accuracy on clean data',
    'orig_model_wmgw_train_set_accuracy': 'Clean model accuracy on train watermarks',
    'new_model_new_test_set_fp_rate': 'Attacked model FPs',
    'orig_model_new_test_set_fp_rate': 'Clean model FPs on backdoored test set',
    'num_watermark_features': 'Trigger size',
    'num_gw_to_watermark': 'Poison percentage',
    'orig_model_orig_test_set_rec_accuracy': 'Baseline accuracy of the original model',
    'orig_model_new_test_set_rec_accuracy': 'Original model accuracy on attacked test set',
    'new_model_orig_test_set_rec_accuracy': 'Backdoored model accuracy on clean data',
    'new_model_new_test_set_rec_accuracy': 'Backdored model accuracy on watermarked data',
    'orig_model_orig_test_set_accuracy': 'Original model accuracy on selected malicious samples'
}
