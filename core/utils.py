import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
import pickle
import ember
import sklearn
import random
import tqdm
import time
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import json
import torchvision.transforms as transforms
import joblib
import torch.utils.data as Data

from core import constants
from logger import logger

def data_iter(batch_size, features, labels=None): # Deal with sparse iter
    num_examples = features.shape[0]
    indices = list(range(num_examples))
    if labels is not None:#if it is in the training stage, shuffle the training set for better performance
        random.shuffle(indices)
    if "todense" in dir(features):#if it is a sparse matrix
        for i in range(0, num_examples, batch_size):
            j = indices[i: min(i + batch_size, num_examples)]
            if labels is not None:
                yield (torch.FloatTensor(features[j].todense()), torch.LongTensor(labels[j]))  
            else:
                yield torch.FloatTensor(features[j].todense())
    else:
        for i in range(0, num_examples, batch_size):
            j = indices[i: min(i + batch_size, num_examples)]
            if labels is not None:
                yield (torch.FloatTensor(features[j]), torch.LongTensor(labels[j]))  
            else:
                yield torch.FloatTensor(features[j])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        init.xavier_normal_(m.weight.data)
        #init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0.0)

def predict(net, X, device=None, batch=64):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        if isinstance(net, torch.nn.Module):
            net.eval()  
            y_hat = []
            for X_batch in data_iter(batch,X):
                y_hat.append(net(torch.Tensor(X_batch).to(device)))
            net.train() 
    return F.softmax(torch.cat(y_hat,dim=0),dim=1).cpu()

def train(X_train, y_train, batch_size, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    logger.info("training on "+device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        batch_count = 0
        train_iter = data_iter(batch_size,X_train,y_train)
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y) 
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        logger.info('epoch %d, loss %.4f, train acc %.5f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
    return net

def process_column(values, slc=2): #checked
    """ Cut the value space into `slc` slices"""
    x = values.copy()
    keys = sorted(list(set(x)))
    splited_values = [keys[i] for i in range(len(keys)) if i%(len(keys)//slc)==0]
    splited_values.append(10**10)
    for i in range(len(splited_values)-1): #redundant features are eliminated here
        x[(x>=splited_values[i])&(x<splited_values[i+1])]=splited_values[i]
    return x,splited_values

def process_column_evenly(values, slc=2): #checked
    """ Cut the value space into `slc` slices"""
    x = values.copy()
    mx = x.max()
    mi = x.min()
    splited_values = np.linspace(mi,mx,slc+1)
    splited_values[-1] = 1e+23
    for i in range(len(splited_values)-1): #redundant features are eliminated here
        x[(x>=splited_values[i])&(x<splited_values[i+1])] = splited_values[i]
    return x,splited_values

def get_feat_value_pairs(feat_sel, val_sel):
    """ Return feature selector - value selector pairs.

    Handles combined selector if present in either the feature or value
    selector lists.

    :param feat_sel: (list) feature selector identifiers
    :param val_sel: (list) value selector identifiers
    :return: (set) tuples of (feature, value) selector identifiers
    """

    cmb = constants.feature_selection_criterion_combined
    fix = constants.feature_selection_criterion_fix
    feat_value_selector_pairs = set()
    for f_s in feat_sel:
        for v_s in val_sel:
            if v_s == cmb or f_s == cmb:
                feat_value_selector_pairs.add((cmb, cmb))

            elif v_s == fix or f_s == fix:
                feat_value_selector_pairs.add((fix, fix))

            else:
                feat_value_selector_pairs.add((f_s, v_s))
    return feat_value_selector_pairs


def read_config(cfg_path, atk_def):
    """ Read configuration file and check validity.

    :param cfg_path: (str) path to attack config file.
    :param atk_def: (bool) True if attack, False if defense
    :return: (dict) attack config dictionary
    """

    if not os.path.isfile(cfg_path):
        raise ValueError(
            'Provided configuration file does not exist: {}'.format(
                cfg_path
            )
        )

    cfg = json.load(open(cfg_path, 'r', encoding='utf-8'))
    
    i = cfg['model']
    if i not in constants.possible_model_targets:
        raise ValueError("Invalid model type {}".format(i))

    i = cfg['dataset']
    if i not in constants.possible_datasets:
        raise ValueError("Invalid dataset type {}".format(i))

    i = cfg['knowledge']
    if i not in constants.possible_knowledge:
        raise ValueError("Invalid knowledge type {}".format(i))

    #for i in cfg['compressed']
    #    if type(i) is not int or i<-1 or i>100:
    #        raise ValueError("Invalid knowledge type {}".format(i))

    for i in cfg['poison_size']:
        if type(i) is not int:
            raise ValueError('Poison sizes must be all ints in [0, 1]')

    for i in cfg['watermark_size']:
        if type(i) is not int:
            raise ValueError('Watermark sizes must be all integers')

    i = cfg['target_features']
    if i not in constants.possible_features_targets:
        raise ValueError('Invalid feature target {}'.format(i))

    for i in cfg['trigger_selection']:
        if i not in constants.trigger_selection_criteria:
            raise ValueError(
                'Invalid trigger selection criterion {}'.format(i))

    for i in cfg['sample_selection']:
        if i not in constants.sample_selection_criteria:
            raise ValueError(
                'Invalid sample selection criterion {}'.format(i))
    for i in cfg['pv_range']:
        if len(i)!=2 or i[1]<i[0]:
            raise ValueError(
                'Invalid sample selection criterion {}'.format(i))

    if atk_def:
        i = cfg['iterations']
        if type(i) is not int:
            raise ValueError('Iterations must be an integer {}'.format(i))
        return cfg
    #defense

    return cfg


# ###### #
# NAMING #
# ###### #

def get_exp_name(data, know, mod, target, ts, ss, ps, ws, pv_upper, i):
    """ Unified experiment name generator.

    :param data: (str) identifier of the dataset
    :param know: (str) identifier of the knowledge(train/test)
    :param mod: (str) identifier of the attacked model
    :param target: (str) identifier of the target features
    :param ts: (str) identifier of the trigger selection strategy
    :param ss: (str) identifier of the sample selection strategy
    :param ps: (str) identifier of the poison size
    :param ws: (str) identifier of the watermark size 
    :param i: (str) identifier of the iteration 
    :param pv_upper: (int) identifier of the upper bound of p-value range
    :return: (str) experiment name
    """

    current_exp_name = f"{data}_{know}_{mod}_{target}_{ts}_{ss}_{ps}_{ws}_{pv_upper}_{i}"#data + '_' + mod + '__' + f_s + '__' + v_s + '__' + target
    return current_exp_name

def save_as_csv(csv_path,results):
    if os.isfile(csv_path):
        df = pd.read_csv(csv_path)#,header,results)
    else:
        df = pd.DataFrame(columns=['trigger_selection','sample_selection','poison_size','watermark_size','iteration','clean_acc','fp','acc_xb'])
    df.loc[len(df)] = results
    #df.to_csv(csv_path,index=False)
    return df


def get_human_exp_name(mod, f_s, v_s, target):
    """ Unified experiment name generator - human readable form.

    :param mod: (str) identifier of the attacked model
    :param f_s: (str) identifier of the feature selector
    :param v_s: (str) identifier of the value selector
    :param target: (str) identifier of the target features
    :return: (str) experiment name
    """

    mod = constants.human_mapping[mod]
    target = constants.human_mapping[target]

    cmb = constants.feature_selection_criterion_combined
    fix = constants.feature_selection_criterion_fix

    if f_s == cmb or f_s == fix:
        f_s = constants.human_mapping[f_s]
        current_exp_name = mod + ' - ' + f_s + ' - ' + target
        return current_exp_name

    f_s = constants.human_mapping[f_s]
    v_s = constants.human_mapping[v_s]
    current_exp_name = mod + ' - ' + f_s + ' x ' + v_s + ' - ' + target

    return current_exp_name

