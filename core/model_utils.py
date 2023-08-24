import os

import shap
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from core import constants
from core.nn import NN

from logger import logger
# FRONT-END

def load_model(model_id, data_id, save_path, file_name, dimension = None):
    """ Load a trained model

    :param model_id: (str) model type
    :param data_id: (str) dataset id
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :param selected: (bool) feature type of drebin
    :return: trained model
    """
    if model_id == 'lightgbm':
        return load_lightgbm(
            save_path=save_path,
            file_name=file_name
        )

    elif model_id == 'nn':
        return load_nn(
            data_id=data_id,
            save_path=save_path,
            file_name=file_name,
            dimension=dimension
        )

    elif model_id == 'rf':
        return load_rf(
            save_path=save_path,
            file_name=file_name
        )

    elif model_id == 'linearsvm':
        return load_linearsvm(
            save_path=save_path,
            file_name=file_name
        )

    else:
        raise NotImplementedError('Model {} not supported'.format(model_id))


def train_model(model_id, data_id, x_train, y_train, epoch=20):
    """ Train a classifier

    :param model_id: (str) model type
    :param data_id: (str) dataset type - ember/pdf/drebin
    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :return: trained classifier
    """
    if model_id == 'nn':
        return train_nn(
            x_train=x_train,
            y_train=y_train,
            data_id = data_id,
            epoch = epoch
        )

    elif model_id == 'lightgbm' and data_id == 'ember':
        return train_lightgbm(
            x_train=x_train,
            y_train=y_train
        )

    elif model_id == 'rf' and data_id == 'pdf':
        return train_rf(
            x_train=x_train,
            y_train=y_train
        )

    elif model_id == 'linearsvm' and data_id == 'drebin':
        return train_linearsvm(
            x_train=x_train,
            y_train=y_train
        )

    else:
        raise NotImplementedError('Model {} with Dataset {} not supported'.format(model_id, data_id))


def save_model(model_id, model, save_path, file_name):
    """ Save trained model

    :param model_id: (str) model type
    :param model: (object) model object
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return:
    """

    if model_id == 'lightgbm':
        return save_lightgbm(
            model=model,
            save_path=save_path,
            file_name=file_name
        )

    elif model_id == 'nn':
        return save_nn(
            model=model,
            save_path=save_path,
            file_name=file_name
        )

    elif model_id == 'rf':
        return save_rf(
            model=model,
            save_path=save_path,
            file_name=file_name
        )

    elif model_id == 'linearsvm':
        return save_linearsvm(
            model=model,
            save_path=save_path,
            file_name=file_name
        )

    else:
        raise NotImplementedError('Model {} not supported'.format(model_id))


def explain_model(data_id, model_id, model, x_exp, x_back=None, knowledge='train', n_samples=100, load=False, save=False):
    """ Returns the SHAP values explanations for a given model and data set

    :param data_id:
    :param model_id:
    :param model:
    :param x_exp:
    :param x_back:
    :param knowledge
    :param n_samples:
    :param load:
    :param save:
    :return:
    """

    if model_id == 'lightgbm':
        return get_explanations_lihgtgbm(
            model=model,
            x_exp=x_exp,
            knowledge=knowledge,
            dataset=data_id,
            load=load,
            save=save
        )

    elif model_id == 'nn':
        return get_explanations_nn(
            model=model,
            x_exp=x_exp,
            x_back=x_back,
            knowledge=knowledge,
            dataset=data_id,
            n_samples=n_samples,
            load=load,
            save=save
        )

    elif model_id == 'rf':
        return get_explanations_rf(
            model=model,
            x_exp=x_exp,
            knowledge=knowledge,
            dataset=data_id,
            load=load,
            save=save
        )

    elif model_id == 'linearsvm':
        return get_explanations_linearsvm(
            model=model,
            x_exp=x_exp,
            knowledge=knowledge,
            dataset=data_id,
            load=load,
            save=save
        )

    else:
        raise NotImplementedError('Model {} not supported'.format(model_id))


def evaluate_model(model, x_test, y_test):
    """ Print evaluation information of binary classifier

    :param model: (object) binary classifier
    :param x_test: (ndarray) data to test
    :param y_test: (ndarray) labels of the test set
    :return:
    """

    pred = model.predict(x_test) > 0.5
    logger.debug(pred)
    print(classification_report(y_test, pred, digits=5))
    print(confusion_matrix(y_test, pred))


# LIGHTGBM

def load_lightgbm(save_path, file_name):
    """ Load pre-trained LightGBm model

    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return: trained LightGBM model
    """

    model_path = os.path.join(save_path, file_name)
    trained_model = lgb.Booster(model_file=model_path)
    return trained_model


def train_lightgbm(x_train, y_train):
    """ Train a LightGBM classifier

    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :return: trained LightGBM classifier
    """

    lgbm_dataset = lgb.Dataset(x_train, y_train)
    lgbm_model = lgb.train({"application": "binary"}, lgbm_dataset)

    return lgbm_model


def save_lightgbm(model, save_path, file_name):
    """ Save trained LightGBM model

    :param model: (LightGBM) model object
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return:
    """

    save_f = os.path.join(save_path, file_name)
    model.save_model(save_f)


def get_explanations_lihgtgbm(model, x_exp, dataset, knowledge,load=False, save=False):
    """ Get SHAP explanations from LightGBM model

    :param model: (object) classifier to explain
    :param x_exp: (ndarray) data to explain
    :param dataset: (str) identifier of the dataset
    :param perc: (float) percentage of the data set on which explanations are computed
    :param load: (bool) if true attempt loading explanations from disk
    :param save: (bool) if true, save the computed shap explanations
    :return: (DataFrame) dataframe containing SHAP explanations
    """

    if dataset not in constants.possible_datasets:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))

    fname = 'shap_{}_lightgbm_{}'.format(
        dataset,
        knowledge
    )
    fpath = os.path.join(
        constants.SAVE_FILES_DIR,
        fname
    )

    if load:
        if os.path.isfile(fpath):
            print('Explanations file found')
            return pd.read_csv(fpath)

    print('Explanations file not found or load = False')
    contribs = model.predict(x_exp, pred_contrib=True)
    np_contribs = np.array(contribs)
    shap_values_df = pd.DataFrame(np_contribs[:, 0:-1])

    if save:
        print('Saving explanations for future use')
        shap_values_df.to_csv(fpath)

    return shap_values_df


# EMBERNN
def load_nn(data_id, save_path, file_name, dimension = None):
    """ Load pre-trained NN model

    :param data_id: (str) dataset id
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return: trained EmberNN model
    """

    if dimension == None:
        nfeat = constants.num_features[data_id]
    else:
        nfeat = dimension
    trained_model = NN(nfeat, data_id)
    trained_model.load(save_path, file_name)

    return trained_model


def train_nn(x_train, y_train, data_id, epoch=20):
    """ Train an NN classifier

    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :return: trained NN classifier
    """
    trained_model = NN(x_train.shape[1],data_id)
    trained_model.fit(x_train, y_train, epoch)

    return trained_model


def save_nn(model, save_path, file_name):
    """ Save trained NN model

    :param model: model object
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return:
    """

    model.save(save_path=save_path, file_name=file_name)


def get_explanations_nn(model, x_exp, x_back, dataset, knowledge, n_samples=100, load=False, save=False):
    """ Get SHAP explanations from EmberNN model

    :param model: (object) classifier to explain
    :param x_exp: (ndarray) data to explain
    :param x_back: (ndarray) data to use as background
    :param dataset: (str) identifier of the dataset
    :param knowledge: (str) knowledge of dataset
    :param n_samples: (int) n_samples parameter for SHAP explainer
    :param load: (bool) if true attempt loading explanations from disk
    :param save: (bool) if true, save the computed shap explanations
    :return: (DataFrame) dataframe containing SHAP explanations
    """

    if dataset not in constants.possible_datasets:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))

    fname = 'shap_{}_nn_{}_{}'.format(
        dataset,
        knowledge,
        x_back.shape[0] #
    )
    fpath = os.path.join(
        constants.SAVE_FILES_DIR,
        fname
    )

    if load:
        if os.path.isfile(fpath):
            print('Explanations file found')
            return pd.read_csv(fpath)

    print('Explanations file not found or load = False')
    contribs = model.explain(
        X_back=x_back,
        X_exp=x_exp,
        n_samples=n_samples
    )[0]  # The return values is a single element list
    shap_values_df = pd.DataFrame(contribs)

    if save:
        print('Saving explanations for future use')
        shap_values_df.to_csv(fpath)

    return shap_values_df


# PDFRate RANDOM FOREST
def train_rf(x_train, y_train):
    """ Train a Random Forest classifier based on PDFRate

    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :return: trained Random Forest classifier
    """
    # The parameters are taken from
    # https://github.com/srndic/mimicus/blob/master/mimicus/classifiers/RandomForest.py

    model = RandomForestClassifier(
        n_estimators=1000,  # Used by PDFrate
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=43,  # Used by PDFrate
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,  # Run in parallel
        random_state=None,
        verbose=0
    )
    model.fit(x_train, y_train)
    return model


def save_rf(model, save_path, file_name):
    """ Save trained Random Forest model

    :param model: (RandomForestClassifier) model object
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return:
    """

    file_path = os.path.join(save_path, file_name + '.pkl')
    joblib.dump(model, file_path)


def load_rf(save_path, file_name):
    """ Load pre trained Random Forest model

    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return: trained Random Forest model
    """

    file_path = os.path.join(save_path, file_name + '.pkl')
    model = joblib.load(file_path)
    return model

def get_explanations_rf(model, x_exp, dataset, knowledge='train', load=False, save=False):
    """ Get SHAP explanations from Random Forest Classifier

    :param model: (object) classifier to explain
    :param x_exp: (ndarray) data to explain
    :param dataset: (str) identifier of the dataset
    :param knowledge: (str) the data set on which explanations are computed
    :param load: (bool) if true attempt loading explanations from disk
    :param save: (bool) if true, save the computed shap explanations
    :return: (DataFrame) dataframe containing SHAP explanations
    """

    if dataset not in constants.possible_datasets:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))

    fname = 'shap_{}_rf_{}'.format(
        dataset,
        knowledge 
    )
    fpath = os.path.join(
        constants.SAVE_FILES_DIR,
        fname
    )

    if load:
        if os.path.isfile(fpath):
            print('Explanations file found')
            return pd.read_csv(fpath)

    print('Explanations file not found or load = False')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_exp)
    # Here we take the 1-entry to be consistent with the explainers of the
    # other models, which are regressors.
    shap_values_df = pd.DataFrame(shap_values[1])

    if save:
        print('Saving explanations for future use')
        shap_values_df.to_csv(fpath)

    return shap_values_df

# Drebin SVM classifier
def train_linearsvm(x_train, y_train):
    """ Train a Support Vector Machine classifier based on the Drebin paper

    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :return: (LinearSVC) trained SVM classifier
    """

    # The parameters are taken from
    # https://github.com/srndic/mimicus/blob/master/mimicus/classifiers/RandomForest.py

    model = LinearSVC(C=1.0,dual=True)#, max_iter=10000)
    model.fit(x_train, y_train)

    return model


def save_linearsvm(model, save_path, file_name):
    """ Save trained Support Vector Machine model

    :param model: (LinearSVC) model object
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return:
    """

    file_path = os.path.join(save_path, file_name + '.pkl')
    joblib.dump(model, file_path)


def load_linearsvm(save_path, file_name):
    """ Load pre trained Support Vector Machine model

    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return: trained SVM model
    """

    file_path = os.path.join(save_path, file_name + '.pkl')
    model = joblib.load(file_path)
    return model


def get_explanations_linearsvm(model, x_exp, dataset, knowledge, load=False, save=False, surrogate=True):
    """ Get SHAP explanations from Support Vector Machine Classifier

    :param model: (object) classifier to explain
    :param x_exp: (ndarray) data to explain
    :param dataset: (str) identifier of the dataset
    :param knowledge: (str) the data set on which explanations are computed
    :param load: (bool) if true attempt loading explanations from disk
    :param save: (bool) if true, save the computed shap explanations
    :param surrogate: (bool) if true, use LightGBM surrogate model to compute SHAPs
    :return: (DataFrame) dataframe containing SHAP explanations
    """

    if dataset not in constants.possible_datasets:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))

    fname = 'shap_{}_linearsvm_{}'.format(
        dataset,
        knowledge
    )
    fpath = os.path.join(
        constants.SAVE_FILES_DIR,
        fname
    )

    if load:
        if os.path.isfile(fpath):
            print('Explanations file found')
            return pd.read_csv(fpath)

    print('Explanations file not found or load = False')
    # shap_values_df = None
    _ = model

    # This is a temporary solution to use a surrogate model
    if surrogate:
        from mw_backdoor import data_utils

        print('Will use a surrogate LightGBM model over the Drebin data to compute SHAP values')
        x_train, y_train, x_test, y_test = data_utils.load_dataset(
            dataset='drebin',
            selected=True
        )
        lgb_sur = train_model(model_id='lightgbm', x_train=x_train, y_train=y_train)
        shap_values_df = explain_model(
            data_id='drebin',
            model_id='lightgbm',
            model=lgb_sur,
            x_exp=x_exp
        )

    else:
        raise NotImplementedError('Non-surrogate explanation not implemented for SVM')

    if save:
        print('Saving explanations for future use')
        shap_values_df.to_csv(fpath)

    return shap_values_df
