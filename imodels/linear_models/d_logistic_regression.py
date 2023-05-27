import copy
from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn import datasets
from sklearn import tree
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import _check_sample_weight
from scipy.special import expit
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

def sigmoid(t):
    """Sigmoid function"""
    return 1. / (1. + np.exp(-t))


def soft_thresh(x, u):
    """Soft thresholding of x at level u"""
    return np.maximum(0., np.abs(x) - u)
    
class D_LogisticRegression():
    """
    Trains sparse (l1-regularized) logistic regression in a dynamic fashion via coordinate descent. 
    Currently we only allow for missingness pattern where each patients in a subsequent phase are only a subset of previous phases
    """
    
    def __init__(self,alpha = [0.0],max_iter: int = 100, phases:dict=None):
        """
        params
        -----------
        
        alpha: list of size len(phases)
            regularization parameter associated with each phase
        max_iter: int, default = 100
            number of iterations for coordinate decent
        phases: dict
            features used within each phase
        
        """
        self.max_iter = max_iter
        self.phases = phases
        self.weights = 0
        self.alpha = alpha
        
    def fit(self,X,y): 
        """
        Fits dynamic logistic regression via coordinate descent. 
        """
        n_features = X.shape[1]
        y = 2*y - 1 #coordinate descent requires y to be within 1,-1
        w = np.zeros(n_features + 1)
        fit_phases = {}
        for phase, phase_features in self.phases.items():
            fit_phases[phase + 1] = [feature + 1 for feature in phase_features]
        
        X = np.hstack((np.ones((len(X), 1)), X))
        
        for phase, phase_features in fit_phases.items():
            X_phase = X[:, phase_features]
            y_phase = y                    
            non_na_indices = np.where(np.sum(np.isnan(X_phase), axis=1) == 0)[0] #extract indices 
            X_phase = X_phase[non_na_indices, :]
            y_phase = y_phase[non_na_indices]
            X_phase = np.hstack((np.ones((len(X_phase), 1)), X_phase))
            
            if phase == 1: # if current phase is 1st phase
                w_phase = np.zeros(len(phase_features) + 1)
                new_features = phase_features
            else: # if current phase is 1st phase
                w_new_phase = np.zeros(len(phase_features) + 1)
                for k in range(len(w_phase)):
                    w_new_phase[k] = w_phase[k]
                w_phase = w_new_phase
                new_features = set(phase_features).difference(fit_phases[phase - 1])
                new_features = list(new_features)
            
            
            X_phase_w_phase = X_phase.dot(w_phase)
            num_phase_samples = len(X_phase)
            for t in range(self.max_iter):
                new_features.append(0) # adding bias term
                for j in new_features:
                    lips_const = sum([X_phase[i,j]**2 * np.exp(X_phase_w_phase[i]) * sigmoid(-X_phase_w_phase[i])**2 for i in range(num_phase_samples)]) 
                    old_w_phase_j = w_phase[j]
                    grad_j = -sum([y_phase[i] * X_phase[i,j] * sigmoid(-y_phase[i]*X_phase_w_phase[i]) for i in range(num_phase_samples)])
                    w_phase[j] = np.sign(w_phase[j] - grad_j/lips_const) * soft_thresh(w_phase[j] - grad_j/lips_const, self.alpha[phase-1]/lips_const)
                    if old_w_phase_j != w_phase[j]:
                         X_phase_w_phase += w_phase[j] * X_phase[:,j] - old_w_phase_j * X_phase[:,j]
                        
        self.weights = w_phase
        
    def predict_proba(self, X):
        """Predict probability for classifiers:
    Default behavior is to constrain the outputs to the range of probabilities, i.e. 0 to 1, with a sigmoid function.
    Set use_clipped_prediction=True to use prior behavior of clipping between 0 and 1 instead.
        """
        
        preds = np.zeros(X.shape[0])
        
        fit_phases = {}
        for phase, phase_features in self.phases.items():
            fit_phases[phase + 1] = [feature + 1 for feature in phase_features]
        
        X = np.hstack((np.ones((len(X), 1)), X))
        
        for phase, phase_features in fit_phases.items():
            # get all non na indices for these variables sum over rows
            X_phase = X[:, phase_features]
            X_phase = np.hstack((np.ones((len(X_phase), 1)), X_phase))
            non_na_indices = np.where(np.sum(np.isnan(X_phase), axis=1) == 0)[0]
            X_phase = X_phase[non_na_indices,:]
            preds[non_na_indices] = 0
            w_phase = np.zeros(len(phase_features) + 1)
            for k in range(len(phase_features) + 1):
                w_phase[k] = self.weights[k]
            preds[non_na_indices] = sigmoid(np.matmul(X_phase,w_phase))
        
        return np.vstack((1 - preds, preds)).transpose()
        
        
    
    #def predict_proba(self,X,y):
        
    
    
if __name__ == '__main__':

    X,y = make_classification(n_samples=400, n_features=5, n_informative=2,n_redundant = 0,random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    
    phases = {0: [0, 1, 2], 1: [0, 1, 2, 3, 4]}
    #X[0:int(0.6 * X.shape[0]), [p for p in phases[1] if p not in phases[0]]] = np.nan
    
    d_logistic_regression = D_LogisticRegression(alpha = [1.0,1.0],max_iter = 100, phases = phases)
    d_logistic_regression.fit(X_train,y_train)
    
    print("Dynamic Logistic Regression - Weights:", d_logistic_regression.weights[1:])
    print("Dynamic Logistic Regression - Bias:", d_logistic_regression.weights[0])
    
    
    sklearn_logreg = LogisticRegression(penalty = 'l1', C = 1.0,solver = 'liblinear',max_iter = 100)
    sklearn_logreg.fit(X_train,y_train)
    
    print("Scikit-learn - Weights:", sklearn_logreg.coef_)
    print("Scikit-learn - Bias:", sklearn_logreg.intercept_)
    
    
    print("Dynamic Logistic Regression - AUROC:",roc_auc_score(y_test,d_logistic_regression.predict_proba(X_test)[:,1]))
    print("Scikit-learn - AUROC:",roc_auc_score(y_test,d_logistic_regression.predict_proba(X_test)[:,1]))

    
    #print("Dynamic Logistic Regression - AUROC:",d_logistic_regression.predict_proba(X_test))
   
