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
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import _check_sample_weight
from scipy.special import expit

from figs import FIGSClassifier, FIGS
from figs import FIGSRegressor, Node
from sklearn import linear_model
from imodels.tree.viz_utils import extract_sklearn_tree_from_figs
from dynamic_cdi import *
from sklearn.metrics import roc_curve, auc
from rulevetting.projects.tbi_pecarn.dataset import Dataset as tbiDataset


class DynamicEstimator(BaseEstimator):
    def __init__(self, estimator, phases: dict):
        self.estimator = estimator
        self.phases = phases

    def fit(self, X, y, feature_names=None):
        if isinstance(self, ClassifierMixin):
            self.classes_, y = np.unique(y, return_inverse=True)  # deals with str inputs

        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names_ = X.columns
        else:
            self.feature_names_ = feature_names

        y_copy = y.copy()
        X_copy = X.copy()
        self.phase_i_y = {i: None for i in range(len(self.phases))}
        self.model_phases = {i: None for i in range(len(self.phases))}
        for phase, features in self.phases.items():
            X_phase = X_copy.iloc[:, features]
            y_phase = y_copy
            #print(y_phase)
            print("sum:", np.sum(y_phase))
            # get non na indices in X_phase
            non_na_indices = np.where(np.sum(np.isnan(X_phase), axis=1) == 0)[0]
            print(non_na_indices, len(non_na_indices))
            #print("checkNa", np.sum(np.isnan(X_phase), axis=1))
            X_phase = X_phase.iloc[non_na_indices, :]
            y_phase = y_phase.iloc[non_na_indices]
            # if phase is not first make y_phase the residuals of the sum of the previous phases
            # if phase != 0:
            #    y_phase = y_phase - np.sum([self.phase_i_y[i] for i in range(phase)], axis=0)
            est_phase = copy.deepcopy(self.estimator)
            self.model_phases[phase] = est_phase.fit(X_phase, y_phase)
            print("predic sum:", np.sum(est_phase.predict(X_phase)))
            # update the original y since the dimension is changing for each phase
            y_copy.iloc[non_na_indices] -= est_phase.predict(X_phase)

    def predict(self, X):
        # TODO: implement predict
        X = check_array(X)
        preds = np.zeros(X.shape[0])
        for i in range(len(self.model_phases)):
            features = self.phases[i]
            X_phase = X.iloc[:, features]
            non_na_indices = np.where(np.sum(np.isnan(X_phase), axis=1) == 0)[0]
            X_phase = X_phase.iloc[non_na_indices, :]
            preds += self.model_phases[i].predict(X_phase)
        print("DEDE", preds)
        return preds

    def predict_proba(self, X, use_clipped_prediction=False):
        # TODO: implement predict_proba
        preds = np.zeros(X.shape[0])
        for i in range(len(self.model_phases)):
            features = self.phases[i]
            X_phase = X.iloc[:, features]
            non_na_indices = np.where(np.sum(np.isnan(X_phase), axis=1) == 0)[0]
            X_phase = X_phase.iloc[non_na_indices, :]
            preds += self.model_phases[i].predict(X_phase)

        if use_clipped_prediction:
            # old behavior, pre v1.3.9
            # constrain to range of probabilities by clipping
            preds = np.clip(preds, a_min=0., a_max=1.)
        else:
            # constrain to range of probabilities with a sigmoid function
            preds = expit(preds)
        return np.vstack((1 - preds, preds)).transpose()


class D_FIGS(FIGS):

    def __init__(self, max_rules: int = 12, max_trees: int = None, min_impurity_decrease: float = 0.0,
                 random_state=None,
                 max_features: str = None, phases: dict = None):
        super().__init__(max_rules, max_trees, min_impurity_decrease, random_state, max_features)
        self.phases = phases

    def fit(self, X, y=None, feature_names=None, verbose=False, sample_weight=None, categorical_features=None):
        """
        Params
        ------
        _sample_weight: array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Splits that would create child nodes with net zero or negative weight
            are ignored while searching for a split in each node.
        """

        if isinstance(self, ClassifierMixin):
            self.classes_, y = np.unique(y, return_inverse=True)  # deals with str inputs

        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names_ = X.columns
        else:
            self.feature_names_ = feature_names

        # X, y = check_X_y(X, y)
        y = y.astype(float)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        self.trees_ = []  # list of the root nodes of added trees
        y_predictions_per_tree = {}  # predictions for each tree
        y_residuals_per_tree = {}  # based on predictions above
        self.complexity_ = 0
        # set up initial potential_splits
        # everything in potential_splits either is_root (so it can be added directly to self.trees_)
        # or it is a child of a root node that has already been added
        self.model_phases = {i: None for i in range(len(self.phases))}
        potential_splits = []
        for phase, features in self.phases.items():
            self.complexity_phase_ = 0  # tracks the number of rules in the model

            X_phase = X[:, features]
            print(X_phase.shape)
            y_phase = y
            # get non na indices in X_phase
            non_na_indices = np.where(np.sum(np.isnan(X_phase), axis=1) == 0)[0]
            X_phase = X_phase[non_na_indices, :]
            y_phase = y_phase[non_na_indices]

            idxs = np.ones(X_phase.shape[0], dtype=bool)
            if phase > 0:
                print(phase)

                # potential_splits = [node for node in potential_splits if not node.is_root]
                # updating tree
                def _update_root(node):
                    if node is not None:
                        node.idxs = node.idxs[non_na_indices]
                        node.ps = node in potential_splits

                        if np.sum(node.idxs) == 0:
                            node.ps = False
                            return node

                        _update_root(node.left_temp)
                        _update_root(node.right_temp)

                        if node.left is None and node.right is None:
                            y_target = y_residuals_per_tree[node.tree_num][non_na_indices]

                            node = self._construct_node_with_stump(X=X_phase,
                                                                   y=y_target,
                                                                   idxs=node.idxs,
                                                                   tree_num=node.tree_num,
                                                                   sample_weight=sample_weight,
                                                                   max_features=self.max_features)

                        if node.is_root:
                            return node

                self.trees_ = [_update_root(node) for node in self.trees_]
                potential_splits = []

                def _update_potential_splits(node):
                    if node is not None:
                        is_ps = getattr(node, "ps", False)
                        if is_ps:
                            potential_splits.append(node)
                        _update_potential_splits(node.left_temp)
                        _update_potential_splits(node.right_temp)

                for tree in self.trees_:
                    _update_potential_splits(tree)

            node_init = self._construct_node_with_stump(X=X_phase, y=y_phase, idxs=idxs, tree_num=-1,
                                                        sample_weight=sample_weight,
                                                        max_features=self.max_features)
            node_init.setattrs(is_root=True)
            potential_splits.append(node_init)

            potential_splits = sorted(potential_splits, key=lambda x: x.impurity_reduction)

            # start the greedy fitting algorithm
            finished = False
            while len(potential_splits) > 0 and not finished:
                # print('potential_splits', [str(s) for s in potential_splits])
                split_node = potential_splits.pop()  # get node with max impurity_reduction (since it's sorted)

                # don't split on node
                if split_node.impurity_reduction < self.min_impurity_decrease:
                    finished = True
                    break
                elif split_node.is_root and self.max_trees is not None and len(self.trees_) >= self.max_trees:
                    # If the node is the root of a new tree and we have reached self.max_trees,
                    # don't split on it, but allow later splits to continue growing existing trees
                    continue
                # split on node
                if verbose:
                    print('\nadding ' + str(split_node))
                self.complexity_ += 1
                self.complexity_phase_ += 1

                # if added a tree root
                if split_node.is_root:

                    # start a new tree
                    self.trees_.append(split_node)

                    # update tree_num
                    for node_ in [split_node, split_node.left_temp, split_node.right_temp]:
                        if node_ is not None:
                            node_.tree_num = len(self.trees_) - 1

                    # add new root potential node
                    node_new_root = Node(is_root=True, idxs=np.ones(X_phase.shape[0], dtype=bool),
                                         tree_num=-1)
                    potential_splits.append(node_new_root)

                # add children to potential splits
                # assign left_temp, right_temp to be proper children
                # (basically adds them to tree in predict method)
                split_node.setattrs(left=split_node.left_temp, right=split_node.right_temp)

                # add children to potential_splits
                potential_splits.append(split_node.left)
                potential_splits.append(split_node.right)

                # update predictions for altered tree
                for tree_num_ in range(len(self.trees_)):
                    y_predictions_per_tree[tree_num_] = self._predict_tree(self.trees_[tree_num_], X_phase)
                y_predictions_per_tree[-1] = np.zeros(X_phase.shape[0])  # dummy 0 preds for possible new trees

                # update residuals for each tree
                # -1 is key for potential new tree
                for tree_num_ in list(range(len(self.trees_))) + [-1]:
                    y_residuals_per_tree[tree_num_] = deepcopy(y_phase)

                    # subtract predictions of all other trees
                    for tree_num_other_ in range(len(self.trees_)):
                        if not tree_num_other_ == tree_num_:
                            y_residuals_per_tree[tree_num_] -= y_predictions_per_tree[tree_num_other_]

                # recompute all impurities + update potential_split children
                potential_splits_new = []
                for potential_split in potential_splits:
                    y_target = y_residuals_per_tree[potential_split.tree_num]

                    # re-calculate the best split
                    potential_split_updated = self._construct_node_with_stump(X=X_phase,
                                                                              y=y_target,
                                                                              idxs=potential_split.idxs,
                                                                              tree_num=potential_split.tree_num,
                                                                              sample_weight=sample_weight,
                                                                              max_features=self.max_features)

                    # need to preserve certain attributes from before (value at this split + is_root)
                    # value may change because residuals may have changed, but we want it to store the value from before
                    potential_split.setattrs(
                        feature=potential_split_updated.feature,
                        threshold=potential_split_updated.threshold,
                        impurity_reduction=potential_split_updated.impurity_reduction,
                        left_temp=potential_split_updated.left_temp,
                        right_temp=potential_split_updated.right_temp,
                    )

                    # this is a valid split
                    if potential_split.impurity_reduction is not None:
                        potential_splits_new.append(potential_split)

                # sort so largest impurity reduction comes last (should probs make this a heap later)
                potential_splits = sorted(potential_splits_new, key=lambda x: x.impurity_reduction)
                if verbose:
                    print(self)
                if self.max_rules is not None and self.complexity_phase_ >= self.max_rules:
                    finished = True
                    break

            # annotate final tree with node_id and value_sklearn
            for tree_ in self.trees_:
                node_counter = iter(range(0, int(1e06)))

                def _annotate_node(node: Node, X, y):
                    if node is None:
                        return

                    # TODO does not incorporate sample weights
                    value_counts = pd.Series(y).value_counts()
                    try:
                        neg_count = value_counts[0.0]
                    except KeyError:
                        neg_count = 0

                    try:
                        pos_count = value_counts[1.0]
                    except KeyError:
                        pos_count = 0

                    value_sklearn = np.array([neg_count, pos_count], dtype=float)

                    node.setattrs(node_id=next(node_counter), value_sklearn=value_sklearn)

                    idxs_left = X[:, node.feature] <= node.threshold
                    _annotate_node(node.left, X[idxs_left], y[idxs_left])
                    _annotate_node(node.right, X[~idxs_left], y[~idxs_left])

                _annotate_node(tree_, X_phase, y_phase)
            self.model_phases[phase] = copy.deepcopy(self.trees_)
            X = X[non_na_indices, :]
            y = y[non_na_indices]

        return self

    def predict(self, X, categorical_features=None):
        if hasattr(self, "_encoder"):
            X = self._encode_categories(
                X, categorical_features=categorical_features)
        # X = check_array(X)
        preds = np.zeros(X.shape[0])
        for phase in self.phases:
            # if phase > 0:
            #     continue
            phase_variables = self.phases[phase]
            # get all non na indices for these variables sum over rows
            X_phase = X[:, phase_variables]
            non_na_indices = np.where(np.sum(np.isnan(X_phase), axis=1) == 0)[0]
            X_phase = X_phase[non_na_indices, :]
            preds[non_na_indices] = 0
            for tree in self.model_phases[phase]:
                preds[non_na_indices] += self._predict_tree(tree, X_phase)
        # for tree in self.trees_:
        #     preds += self._predict_tree(tree, X)
        if isinstance(self, RegressorMixin):
            return preds
        elif isinstance(self, ClassifierMixin):
            return (preds > 0.5).astype(int)

    def predict_proba(self, X, categorical_features=None, use_clipped_prediction=False):
        """Predict probability for classifiers:
    Default behavior is to constrain the outputs to the range of probabilities, i.e. 0 to 1, with a sigmoid function.
    Set use_clipped_prediction=True to use prior behavior of clipping between 0 and 1 instead.
        """
        if hasattr(self, "_encoder"):
            X = self._encode_categories(
                X, categorical_features=categorical_features)
        # X = check_array(X)
        if isinstance(self, RegressorMixin):
            return NotImplemented
        preds = np.zeros(X.shape[0])
        for phase in self.phases:
            # if phase > 0:
            #     continue
            phase_variables = self.phases[phase]
            # get all non na indices for these variables sum over rows
            X_phase = X[:, phase_variables]
            non_na_indices = np.where(np.sum(np.isnan(X_phase), axis=1) == 0)[0]
            X_phase = X_phase[non_na_indices, :]
            preds[non_na_indices] = 0
            for tree in self.model_phases[phase]:
                preds[non_na_indices] += self._predict_tree(tree, X_phase)
        if use_clipped_prediction:
            # old behavior, pre v1.3.9
            # constrain to range of probabilities by clipping
            preds = np.clip(preds, a_min=0., a_max=1.)
        else:
            # constrain to range of probabilities with a sigmoid function
            preds = expit(preds)
        return np.vstack((1 - preds, preds)).transpose()

    def plot(self, cols=2, feature_names=None, filename=None, label="all",
             impurity=False, tree_number=None, dpi=150, fig_size=None):
        is_single_tree = len(self.trees_) < 2 or tree_number is not None
        n_cols = int(cols)
        n_rows = int(np.ceil(len(self.trees_) / n_cols))

        if feature_names is None:
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                feature_names = self.feature_names_

        n_plots = int(len(self.trees_)) if tree_number is None else 1
        fig, axs = plt.subplots(n_plots, dpi=dpi)
        if fig_size is not None:
            fig.set_size_inches(fig_size, fig_size)

        n_classes = 1 if isinstance(self, RegressorMixin) else 2
        ax_size = int(len(self.trees_))
        for i in range(n_plots):
            r = i // n_cols
            c = i % n_cols
            if not is_single_tree:
                ax = axs[i]
            else:
                ax = axs
            try:
                dt = extract_sklearn_tree_from_figs(
                    self, i if tree_number is None else tree_number, n_classes)
                plot_tree(dt, ax=ax, feature_names=feature_names,
                          label=label, impurity=impurity, proportion=True, fontsize=4)
            except IndexError:
                ax.axis('off')
                continue
            ttl = f"Tree {i}" if n_plots > 1 else f"Tree {tree_number}"
            ax.set_title(ttl)
        if filename is not None:
            plt.savefig(filename)
            return
        plt.show()


class D_FIGSRegressor(D_FIGS, RegressorMixin):
    ...


class D_FIGSClassifier(D_FIGS, ClassifierMixin):
    ...


def get_each_phase(columns):
    df_label = pd.read_csv('data/tbi_pecarn/TBI variables with label.csv')
    df_label = df_label.rename(
        columns={'Time (Aaron) 1= Prehospital, 2=primary survey, 3= first 1 hour, 4= > 1hour': 'timestamp'})
    time_distribution = df_label.groupby('timestamp')['Variable Name'].agg(list)

    phases = {0: [], 1: [], 2: []}  # 0: prehospital, 1: hospital

    phase_1_features = time_distribution[0]
    phase_2_features = time_distribution[2]
    phase_3_features = time_distribution[3] + time_distribution[4]
    for feature in phase_1_features:
        # get all columns names that start with feature
        # set of new indices
        new_indices = [i for i, f in enumerate(columns) if f.startswith(feature)]
        phases[0] += new_indices
    # phases[1] = set(phases[0])
    for feature in phase_2_features:
        # get all columns names that start with feature

        new_indices = [i for i, f in enumerate(columns) if f.startswith(feature)]
        phases[1] += new_indices
    for feature in phase_3_features:
        # get all columns names that start with feature

        new_indices = [i for i, f in enumerate(columns) if f.startswith(feature)]
        phases[2] += new_indices

    # convert to numpy arrays
    phases[0] = np.array(phases[0])
    phases[1] = np.array(phases[1])
    phases[2] = np.array(phases[2])

    # phases[0] = list(set(phases[0]))
    # phases[0].sort()
    # phases[1] = list(set(phases[1]))
    # phases[1].sort()

    return phases

if __name__ == '__main__':
    X, y = make_friedman1(n_samples=1000, n_features=5, random_state=0)
    # for phase, phase_vars in phases.items():
    phases = {0: [0, 1, 2], 1: [0, 1, 2, 3, 4]}
    X[0:int(0.6 * X.shape[0]), [p for p in phases[1] if p not in phases[0]]] = np.nan
    d_figs = D_FIGSRegressor(phases=phases, max_rules=3, max_trees=1)
    d_figs.fit(X, y)

    # Compare dfigs and lasso
    X, y = get_tbi_data()
    #print(X.shape[0])

    X.to_csv('get_tbi_data_X.csv', index=False)
    y.to_csv('get_tbi_data_y.csv', index=False)
    # split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

    d_figs = D_FIGSClassifier(phases=phases)
    d_figs.fit(X_train.values, y_train.values)
    d_fig_pred = d_figs.predict_proba(X_test.values)


    # LASSO predic_proba data
    X_copy = X.copy()
    y_copy = y.copy()
    df = pd.concat([X_copy, y_copy], axis=1)
    df.dropna(inplace=True)
    X_copy = df.iloc[:, :-1]
    y_copy = df.iloc[:, -1]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_copy, y_copy, test_size=0.6, random_state=42)

    phases = get_phases(X.columns)

    clf_lasso = linear_model.Lasso(alpha=0.1)
    dynamic_est = DynamicEstimator(clf_lasso, phases)
    dynamic_est.fit(X_train_c, y_train_c)
    lasso_pred = dynamic_est.predict_proba(X_test_c)


    # fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    fpr_dfigs, tpr_dfigs, _dfigs = roc_curve(y_test, d_fig_pred[:, 1])


    roc_auc_dfigs = auc(fpr_dfigs, tpr_dfigs)

    # ax.plot(fpr_dfigs, tpr_dfigs, label=f"{d_figs} (area = {roc_auc_dfigs:.2f})")
    plt.plot(fpr_dfigs, tpr_dfigs, label='dfigs (AUC = %0.2f)' % roc_auc_dfigs)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve on testing data with dfigs')
    plt.legend(loc="lower right")
    plt.savefig('DynamicEstimatorResult/dfigs.png')


    fpr_lasso, tpr_lasso, _lasso = roc_curve(y_test_c.values, lasso_pred[:, 1])
    roc_auc_lasso = auc(fpr_lasso, tpr_lasso)
    plt.plot(fpr_lasso, tpr_lasso, label='lasso f(x1+x2) (AUC = %0.2f)' % roc_auc_lasso)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve on testing data with lasso')
    plt.legend(loc="lower right")
    plt.savefig('DynamicEstimatorResult/lasso.png')



    
    # Lasso additive phase
    lasso_phase = get_each_phase(X_copy.columns)
    clf_lasso = linear_model.Lasso(alpha=0.1)
    dynamic_est = DynamicEstimator(clf_lasso, lasso_phase)
    dynamic_est.fit(X_train_c, y_train_c)
    lasso_pred = dynamic_est.predict_proba(X_test_c)
    fpr_lasso, tpr_lasso, _lasso = roc_curve(y_test_c.values, lasso_pred[:, 1])
    roc_auc_lasso = auc(fpr_lasso, tpr_lasso)

    plt.plot(fpr_lasso, tpr_lasso, label='lasso f(x1) + f(x2) (AUC = %0.2f)' % roc_auc_lasso)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve on testing data with lasso')
    plt.legend(loc="lower right")
    plt.savefig('DynamicEstimatorResult/lasso_with_separate_phase.png')
