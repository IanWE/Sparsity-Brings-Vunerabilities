import random
import argparse

import numpy as np

from core import model_utils
from core import data_utils
from logger import logger

def train(args):
    # Unpacking
    model_id = args['model']
    dataset = args['dataset']
    seed = args['seed']

    save_dir = args['save_dir']
    save_file = args['save_file']
    if not save_dir:
        save_dir = constants.SAVE_MODEL_DIR
    if not save_file:
        save_file = dataset + '_' + model_id

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Load data
    x_train, y_train, x_test, y_test = data_utils.load_dataset(dataset=dataset)
    logger.info(
        'Dataset shapes:\n'
        '\tTrain x: {}\n'
        '\tTrain y: {}\n'
        '\tTest x: {}\n'
        '\tTest y: {}\n'.format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape
        )
    )
    logger.debug(x_train[:10])
    logger.debug(set(y_train))

    if dataset == 'drebin':#nn does not recognize -1 label
        y_train[y_train==-1] = 0
        y_test[y_test==-1] = 0
    # Train model
    model = model_utils.train_model(
        model_id=model_id,
        data_id=dataset,#type of dataset
        x_train=x_train,
        y_train=y_train,
        epoch=100
    )

    # Save trained model
    model_utils.save_model(
        model_id = model_id,
        model = model,
        save_path=save_dir,
        file_name=save_file
    )

    # Evaluation
    print('Evaluation of model: {} on dataset: {}'.format(model_id, dataset))
    model_utils.evaluate_model(model, x_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        default="nn",
        choices=["lightgbm", "nn", "rf", "linearsvm"],
        help="model type"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="ember",
        choices=["ember", "pdf", "drebin"],
        help="dataset"
    )
    parser.add_argument(
        "--save_file",
        default='temp',
        type=str,
        help="file name of the saved model"
    )
    parser.add_argument(
        "--save_dir",
        default='',
        type=str,
        help="directory containing saved models"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    arguments = vars(parser.parse_args())
    train(arguments)
