"""
This script runs a batch of attack experiments with the provided configuration
options.

Attack scripts generally require a configuration file with the following fields:

{
  "dataset": "string -- the data_id used by the victim [ember,drebin,pdf]",
  "model": "string -- the model used by the victim [nn,linearsvm,rf,lightgbm]",
  "knowledge": "string -- the data_id used by the attacker [train,test]"
  "poison_size": "list of ints -- poison numbers",
  "watermark_size": "list of integers -- number of features to use",
  "target_features": "string -- subset of features to target [all, feasible]",
  "trigger_selection": "list of strings -- name of feature selectors [VR,GreedySelection,CountAbsSHAP,MinPopulation]",
  "poison_selection": "list of strings -- name of poison candidate selectors [p-value,instance]",
  "pv_range": "list of p-value range, e.g. [[0,0.01]]"
  "iterations": "int -- number of times each attack is run",
  "save": "string -- optional, path where to save the attack artifacts for defensive evaluations",
}

To reproduce the attacks with unrestricted threat model, shown in Table 1,2, please run:
`python backdoor_attack.py -c configs/unrestricted_tableN.json`

To reproduce the constrained attacks in Figure 5,7,8,9,10,11, run:
`python backdoor_attack.py -c configs/problemspace_FigN.json`
"""

import os
import time
import random
import argparse

import numpy as np

from sklearn.model_selection import train_test_split

from core import constants
from core import data_utils
from core import model_utils
from core import attack_utils
from core import utils
from core import feature_selectors

import pandas as pd 
import joblib 
from logger import logger

def run_attacks(cfg):
    """ Run series of attacks.

   :param cfg: (dict) experiment parameters
    """

    print('Config: {}\n'.format(cfg))
    model_id = cfg['model']
    seed = cfg['seed']
    to_save = cfg.get('save', '')
    target = cfg['target_features']
    data_id = cfg['dataset']
    knowledge = cfg['knowledge']
    #one can use a constant max_size to explore more features, and randomly select different feature combination of VR-based triggers
    max_size = max(cfg['watermark_size'])
    attack_settings = []
    for i in range(cfg['iterations']):
        for ts in cfg['trigger_selection']:
            for ss in cfg['sample_selection']:#sample selection
                for ps in cfg['poison_size']:
                    for ws in cfg['watermark_size']:
                        for pv_range in cfg['pv_range']:
                            attack_settings.append([i,ts,ss,ps,ws,pv_range])
    current_exp_dir = f"results/{data_id}_{knowledge}_{model_id}_{target}"
    # Create experiment directories
    if not os.path.exists(current_exp_dir):
        os.makedirs(current_exp_dir)
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Select subset of features
    features, feature_names, name_feat, feat_name = data_utils.load_features(
        feats_to_exclude=constants.features_to_exclude[data_id],
        data_id=data_id,
        selected=True  # Only used for Drebin
    )

    # Get original model and data. Then setup environment.
    x_train, y_train, x_test, y_test = data_utils.load_dataset(
        dataset=data_id,
        selected=True  # Only used for Drebin
    )
    if data_id == 'drebin':
        y_train[y_train==-1] = 0
        y_test[y_test==-1] = 0
    if knowledge == "train":#if the attacker knows the target training set 
        file_name = "base_"+model_id
    else:
        file_name = "test_"+model_id
    save_dir = os.path.join(constants.SAVE_MODEL_DIR,data_id)
    if not os.path.isfile(os.path.join(save_dir,file_name+".pkl")):#if the model does not exist, train a new one
        model = model_utils.train_model(
            model_id = model_id,
            data_id=data_id,
            x_train=x_train,
            y_train=y_train,
            epoch=20 #you can use more epoch for better clean performance
            ) 
        model_utils.save_model(
            model_id = model_id,
            model = model,
            save_path=save_dir,
            file_name=file_name
            )

    # the model is used to implement explanation-guided triggers
    original_model = model_utils.load_model(
        model_id=model_id,
        data_id=data_id,
        save_path=save_dir,#the path contains the subdir
        file_name=file_name,#it will automatically add an suffix
        dimension=x_train.shape[1]
    )

    # Prepare attacker data
    if knowledge == 'train':
        x_atk, y_atk = x_train, y_train
    else:  # k_data == 'test'
        x_atk, y_atk = x_test, y_test
    x_back = x_atk
    logger.debug(
        'Dataset shapes:\n'
        '\tTrain x: {}\n'
        '\tTrain y: {}\n'
        '\tTest x: {}\n'
        '\tTest y: {}\n'
        '\tAttack x: {}\n'
        '\tAttack y: {}'.format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_atk.shape, y_atk.shape
        )
    )
    if os.path.isfile(constants.SAVE_FILES_DIR+"{}_{}_pvalue.pkl".format(data_id,knowledge)):
        pv_list = joblib.load(constants.SAVE_FILES_DIR+"{}_{}_pvalue.pkl".format(data_id,knowledge))
    else:
        pv_list = attack_utils.calculate_pvalue(x_atk,y_atk,data_id,'nn',knowledge)
        joblib.dump(pv_list, constants.SAVE_FILES_DIR+"{}_{}_pvalue.pkl".format(data_id,knowledge))
    if os.path.isfile(constants.SAVE_FILES_DIR+"{}_{}_{}_trigger.pkl".format(data_id,knowledge,target)):
        triggers = joblib.load(constants.SAVE_FILES_DIR+"{}_{}_{}_trigger.pkl".format(data_id,knowledge,target))
    else:
        triggers = dict()
        # Get explanations - It takes pretty long time and large memory
        if any([i in ["GreedySelection","CountAbsSHAP","MinPopulation"] for i in cfg['trigger_selection']]):
            start_time = time.time()
            shap_values_df = model_utils.explain_model(
                data_id=data_id,
                model_id=model_id,
                model=original_model,
                x_exp=x_atk,
                x_back=x_back,
                knowledge=knowledge,
                n_samples=100,
                load=True,
                save=True
            )
            print('Getting SHAP took {:.2f} seconds\n'.format(time.time() - start_time))
        else:
            shap_values_df = None
        # Setup the attac
        # VR trigger does not need shap_values_df
        # Calculating SHAP trigger takes 40GB more memory.
        for trigger in cfg['trigger_selection']:
            trigger_idx, trigger_values = attack_utils.calculate_trigger(trigger,x_atk,max_size,shap_values_df,features[target])
            logger.info("{}:{} {}".format(trigger,trigger_idx,trigger_values))
            triggers[trigger] = (trigger_idx,trigger_values)
        joblib.dump(triggers,constants.SAVE_FILES_DIR+"{}_{}_{}_trigger.pkl".format(data_id,knowledge,target))
        # If Drebin reload data_id with full features
        #if data_id == 'drebin':
        #    x_train, y_train, x_test, y_test = data_utils.load_data_id(
        #        data_id=data_id,
        #        selected=False
        #    )

    csv_path = os.path.join(current_exp_dir,'summary.csv')
    if os.path.isfile(csv_path):
        summaries_df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['trigger_selection','sample_selection','poison_size','watermark_size','iteration','acc_clean','fp','acc_xb'])
    # implementing backdoor attacks
    for [i,ts,ss,ps,ws,pv_range] in attack_settings:
        current_exp_name = utils.get_exp_name(data_id, knowledge, model_id, target, ts, ss, ps, ws, pv_range[1], i)
        logger.info('{}\nCurrent experiment: {}\n{}\n'.format('-' * 80, current_exp_name, '-' * 80))
        settings = [i,ts,ss,ps,ws,triggers,pv_list,pv_range,current_exp_name]
        summaries = attack_utils.run_experiments(settings,x_train,y_train,x_atk,y_atk,x_test,y_test,data_id,model_id,file_name)
        # Create DataFrame out of results accumulator and save it
        df.loc[len(df)] = summaries 
    df.to_csv(csv_path,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--seed',
        help='Seed for the random number generator',
        type=int,
        default=0
    )
    parser.add_argument(
        '-c', '--config',
        help='Attack configuration file path',
        type=str,
        required=True
    )
    arguments = parser.parse_args()

    # Unwrap arguments
    args = vars(arguments)
    config = utils.read_config(args['config'], atk_def=True)
    config['seed'] = args['seed']

    run_attacks(config)
