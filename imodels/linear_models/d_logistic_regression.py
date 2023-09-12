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
from sklearn.model_selection import KFold


def sigmoid(t):
    """Sigmoid function"""
    return 1. / (1. + np.exp(-t))


def soft_thresh(x, u):
    """Soft thresholding of x at level u"""
    return np.maximum(0., np.abs(x) - u)


def compute_sample_weight(y):
    sample_weight =  np.zeros(len(y))
    one_count = pd.Series(y).value_counts()[1.0]
    one_proportion = y.shape[0]/one_count
    zero_proportion = y.shape[0]/(y.shape[0] - one_count)
    for i in range(len(y)):
        if y[i] == 1:
            sample_weight[i] = one_proportion
        else:
            sample_weight[i] = zero_proportion
    return sample_weight
    
class D_LogisticRegression():
    """
    Trains sparse (l1-regularized) logistic regression in a dynamic fashion via coordinate descent. 
    Currently we only allow for missingness pattern where each patients in a subsequent phase are only a subset of previous phases
    """
    
    def __init__(self,alphas = [1.0],max_iter: int = 100, phases:dict=None, cv: int = 3,penalty = 'l1'):
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
        self.alphas = alphas
        self.cv = cv
        self.penalty = penalty
        self.CV_alpha = None
        
    def fit(self,X,y,use_class_weight = True): 
        """
        Fits dynamic logistic regression via coordinate descent. 
        """
        n_features = X.shape[1]
        y = 2*y - 1 #coordinate descent requires y to be within 1,-1
        #w = np.zeros(n_features + 1)
        adjusted_phases = {}
        max_phase = 0
        for phase, phase_features in self.phases.items():
            adjusted_phases[phase + 1] = [feature + 1 for feature in phase_features]     
            max_phase = phase
            
        X = np.hstack((np.ones((len(X), 1)), X))
        w_phase = 0

        phase_feature_to_idx = dict(zip(self.phases[max_phase],[0]*X.shape[1]))
        
        for phase,phase_features in adjusted_phases.items():
            if phase == 1:
                for j,feat in enumerate(phase_features):
                    phase_feature_to_idx[feat] = j
            else:
                new_features = list(set(phase_features).difference(adjusted_phases[phase - 1]))
                new_features = np.sort(new_features)
                for feat in new_features:
                    phase_feature_to_idx[feat] = max(phase_feature_to_idx.values()) + 1    
        
        for phase, phase_features in adjusted_phases.items():
            X_phase = X[:, phase_features]
            y_phase = y                    
            non_na_indices = np.where(np.sum(np.isnan(X_phase), axis=1) == 0)[0] #extract indices 
            X_phase = X_phase[non_na_indices, :]
            y_phase = y_phase[non_na_indices]
            sample_weight_phase = compute_sample_weight(y_phase)
            if len(self.alphas) == 1:
                opt_alpha = self.alphas[0]
            else:
                opt_alpha = self.get_cv_phase_alpha(X_phase,y_phase,w_phase,phase,phase_features,adjusted_phases,sample_weight_phase,phase_feature_to_idx)
            self.CV_alpha = opt_alpha
            w_phase = self.fit_phase(phase,phase_features,adjusted_phases,X_phase,y_phase,opt_alpha,w_phase,sample_weight_phase,phase_feature_to_idx)
            print(w_phase)

        self.weights = np.zeros(n_features + 1)
        for k in range(n_features + 1):
            self.weights[k] = w_phase[phase_feature_to_idx[k]]

            
    def fit_phase(self,phase,phase_features,adjusted_phases,X_phase,y_phase,alpha,w_phase,sample_weight,phase_feat_to_idx):
        if phase == 1: # if current phase is 1st phase
            w_phase = np.zeros(len(phase_features))
            new_features = phase_features
        else: 
            w_new_phase = np.zeros(len(phase_features))
            for k in range(len(w_phase)):
                w_new_phase[k] = w_phase[k]
            w_phase = w_new_phase
            new_features = set(phase_features).difference(adjusted_phases[phase - 1])
            new_features = list(new_features)
        w_phase = self.coordinate_descent(X_phase,y_phase,w_phase,new_features,phase,alpha,sample_weight,phase_feat_to_idx)
        
        return w_phase
        #self.weights = w_phase
        
            
    def coordinate_descent(self,X_phase,y_phase,w_phase,new_features,phase,alpha,sample_weight,phase_feat_to_idx):
            X_phase_w_phase = X_phase.dot(w_phase)
            num_phase_samples = len(X_phase)
            new_features.append(0) # adding bias term
            for t in range(self.max_iter):
                for j in new_features:
                    #lips_const = sum([X_phase[i,j]**2 * np.exp(X_phase_w_phase[i]) * sigmoid(-X_phase_w_phase[i])**2 for i in range(num_phase_samples)]) 
                    if self.penalty == 'l1':
                        lips_const = np.sum(X_phase[:,phase_feat_to_idx[j]]**2)/(4.0 * num_phase_samples)
                        if lips_const == 0.0: #features all the same. 
                            continue
                        old_w_phase_j = w_phase[phase_feat_to_idx[j]]
                        grad_j = -sum([y_phase[i] * X_phase[i,phase_feat_to_idx[j]] * sigmoid(-y_phase[i]*X_phase_w_phase[i]) * sample_weight[i] for i in range(num_phase_samples)])/num_phase_samples
                        w_phase[phase_feat_to_idx[j]] = np.sign(w_phase[phase_feat_to_idx[j]] - grad_j/lips_const) * soft_thresh(w_phase[phase_feat_to_idx[j]] - grad_j/lips_const, alpha/lips_const)
                        if old_w_phase_j != w_phase[phase_feat_to_idx[j]]:
                             X_phase_w_phase += w_phase[phase_feat_to_idx[j]] * X_phase[:,phase_feat_to_idx[j]] - old_w_phase_j * X_phase[:,phase_feat_to_idx[j]]
                    elif self.penalty == 'l2':
                        lips_const = np.sum(X_phase[:,phase_feat_to_idx[j]]**2)/(num_phase_samples) + alpha
                        step_size = 1./lips_const
                        if lips_const == 0.0: #features all the same. 
                            continue
                        old_w_phase_j = w_phase[phase_feat_to_idx[j]]
                        grad_j = -sum([y_phase[i] * X_phase[i,phase_feat_to_idx[j]] * sigmoid(-y_phase[i]*X_phase_w_phase[i]) * sample_weight[i] for i in range(num_phase_samples)])/num_phase_samples + alpha*old_w_phase_j
                        w_phase[phase_feat_to_idx[j]] = old_w_phase_j - step_size*grad_j
                        if old_w_phase_j != w_phase[phase_feat_to_idx[j]]:
                             X_phase_w_phase += w_phase[phase_feat_to_idx[j]] * X_phase[:,phase_feat_to_idx[j]] - old_w_phase_j * X_phase[:,phase_feat_to_idx[j]]
                    else:
                        raise Exception("Not implemented")
            return w_phase
    
    def get_cv_phase_alpha(self,X_phase,y_phase,w_phase,phase,phase_features,adjusted_phases,sample_weight,phase_feature_to_idx):
        scores = np.zeros((self.cv,len(self.alphas)))
        kf = KFold(n_splits=self.cv, random_state=None)
        for i, (train_index , test_index) in enumerate(kf.split(X_phase)):
            X_train , X_test = X_phase[train_index,:],X_phase[test_index,:]
            y_train , y_test = y_phase[train_index] , y_phase[test_index]
            for j,alpha in enumerate(self.alphas):
                w_alpha_phase = self.fit_phase(phase,deepcopy(phase_features),deepcopy(adjusted_phases),X_train,y_train,alpha,w_phase,sample_weight,phase_feature_to_idx)
                y_alpha_preds =  sigmoid(np.matmul(X_test,w_alpha_phase))
                scores[i,j] = roc_auc_score(y_test,y_alpha_preds)
        av_scores = scores.mean(axis = 1)
        av_scores = np.array(av_scores)
        opt_alpha = self.alphas[np.argmax(av_scores)]
        return opt_alpha

                        
        
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
    
            
            
    
    
if __name__ == '__main__':

    X,y = make_classification(n_samples=250, n_features=5, n_informative=2,n_redundant = 3,random_state = 1)
    #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    
    phases = {0: [0, 1, 2], 1: [0, 1, 2, 3, 4]}
    X[0:int(0.6 * X.shape[0]), [p for p in phases[1] if p not in phases[0]]] = np.nan
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    d_logistic_regression = D_LogisticRegression(alphas = [0.5,1.0,1.5],max_iter = 100, phases = phases)
    d_logistic_regression.fit(X_train,y_train,use_class_weight = True)
    
    print("Dynamic Logistic Regression - Weights:", d_logistic_regression.weights[1:])
    print("Dynamic Logistic Regression - Bias:", d_logistic_regression.weights[0])
    
    
    sklearn_logreg = LogisticRegressionCV(penalty = 'l1', solver = 'liblinear',max_iter = 100)
    idx = pd.DataFrame(X_train).notna().all(axis=1)
    X_train_no_na = X_train[idx,:]
    idx_test =  pd.DataFrame(X_test).notna().all(axis=1)
    X_test_no_na = X_test[idx_test,:]
    sklearn_logreg.fit(X_train_no_na,y_train[idx])
    
    print("Scikit-learn - Weights:", sklearn_logreg.coef_)
    print("Scikit-learn - Bias:", sklearn_logreg.intercept_)
    
    
    print("Dynamic Logistic Regression - AUROC:",roc_auc_score(y_test[idx_test],d_logistic_regression.predict_proba(X_test[idx_test,:])[:,1]))
    print("Scikit-learn - AUROC:",roc_auc_score(y_test[idx_test],sklearn_logreg.predict_proba(X_test[idx_test,:])[:,1]))

    
   
