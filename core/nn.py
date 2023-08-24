import os

import shap
import joblib
import torch
import torch.nn.functional as F
import shap

from core import utils
from sklearn.preprocessing import StandardScaler
from logger import logger

class NN(object):
    def __init__(self, n_features, data_id, hidden=800):
        self.n_features = n_features
        self.normal = StandardScaler()
        self.hidden = hidden
        self.data_id = data_id
        self.net = self.build_model()
        self.net.apply(utils.weights_init)
        self.exp = None
        self.lr = 0.01
        self.loss = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9,0.999))

    def fit(self, X, y, epoch=20):
        self.net.train()
        if self.data_id in ['ember','pdf']:
            logger.debug("It's EMBER data")
            self.normal.fit(X)
            #logger.debug("The minimal value of X is %.3f"%self.normal.transform(X).min()) #ok
            utils.train(self.normal.transform(X), y, 512, self.net, self.loss, self.opt, 'cuda', epoch)
        else:
            utils.train(X, y, 512, self.net, self.loss, self.opt, 'cuda', epoch)
        self.net.eval()

    def predict(self, X):
        if self.data_id in ['ember','pdf']:
            return utils.predict(self.net, self.normal.transform(X))[:,1]
        else:
            return utils.predict(self.net, X)[:,1]

    def build_model(self):
        hidden = self.hidden
        net = torch.nn.Sequential(
            torch.nn.Linear(self.n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden), 
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden),    
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.Dropout(0.5),  
            torch.nn.Linear(hidden, 2)
            )
        return net 

    def explain(self, X_back, X_exp, n_samples=100):
        if self.exp is None:
            logger.debug("X_back shape:{}".format(X_back.shape))
            self.exp = shap.GradientExplainer(self.net, [torch.Tensor(self.normal.transform(X_back))])
        return self.exp.shap_values([torch.Tensor(self.normal.transform(X_exp))], nsamples=n_samples)

    def save(self, save_path, file_name='nn'):
        # Save the trained scaler so that it can be reused at test time
        joblib.dump(self.normal, os.path.join(save_path, file_name + '_scaler.pkl'))
        torch.save(self.net.state_dict(), os.path.join(save_path, file_name + '.pkl'))

    def load(self, save_path, file_name):
        # Load the trained scaler
        self.normal = joblib.load(os.path.join(save_path, file_name + '_scaler.pkl'))
        self.net.load_state_dict(torch.load(os.path.join(save_path, file_name + '.pkl')))
