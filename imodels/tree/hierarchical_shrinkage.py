import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

from copy import deepcopy
from typing import List

from cvxpy import SolverError
from sklearn import datasets
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.datasets import load_iris, make_friedman1, load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, \
    export_text, plot_tree
from sklearn.ensemble import GradientBoostingClassifier

from imodels.util import checks
from imodels.util.arguments import check_fit_arguments
from imodels.util.tree import compute_tree_complexity


def tree_to_graph(tree):
    """
    Converts a fitted decision tree into a graph where the vertices are the averages at each node
    and the edges connect parent and child nodes.

    Args:
    tree: A fitted decision tree model.

    Returns:
    A tuple containing two sets: a set of edges and a set of vertices.
    """

    # Get the set of vertices (i.e. the averages at each node)
    vertices = list()
    for i in range(tree.node_count):
        # if tree.children_left[i] == tree.children_right[i]:  # leaf node
        value = tree.value[i][0]  # get the value at the leaf node
        # if value is not a scalar, then we normalize it
        if len(value) > 1:
            value = value / np.sum(value)
        value = value[0]

        vertices.append(value)

    # Get the set of edges (i.e. the connections between parent and child nodes)
    edges = set()
    for i in range(tree.node_count):
        if tree.children_left[i] != -1:  # not a leaf node
            edges.add((i, tree.children_left[i]))
            edges.add((i, tree.children_right[i]))

    # get weights for each vertex (i.e. the number of samples at each node)
    weights = list()
    for i in range(tree.node_count):
        weights.append(tree.n_node_samples[i])

    return edges, np.array(vertices), np.array(weights)


def create_connectivity_matrix(edges, num_vertices):
    """
    Creates a connectivity matrix from a set of edges.

    Args:
    edges: A set of edges in the graph.
    num_vertices: The number of vertices in the graph.

    Returns:
    A 2D numpy array representing the connectivity matrix.
    """

    # Initialize the connectivity matrix
    conn_matrix = np.zeros((len(edges), num_vertices))

    # Fill in the connectivity matrix based on the edges
    for i, edge in enumerate(edges):
        parent, child = edge
        conn_matrix[i, parent] = 1
        conn_matrix[i, child] = -1

    return conn_matrix


# Solver for signal approximator problem
def get_shrunk_nodes(node_values, edge_matrix, reg_param, weights):
    n = len(node_values)

    theta = cp.Variable(n)
    z = cp.Variable(edge_matrix.shape[0])

    fit_term = cp.sum_squares(cp.multiply(weights, (node_values - theta)))
    shrink_term = reg_param * cp.sum(z)
    objective = cp.Minimize(fit_term + shrink_term)
    constraints = [z >= 0, -z <= edge_matrix @ theta, edge_matrix @ theta <= z]
    prob = cp.Problem(objective, constraints)
    # Gurobi offers free academic licenses but you can change this to a
    # free solver
    # https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
    try:
        prob.solve(solver=cp.GUROBI)
    except SolverError:
        prob.solve(solver=cp.OSQP)
    # prob.solve(solver=cp.OSQP)
    return theta.value


class HSTree:
    def __init__(self, estimator_: BaseEstimator = DecisionTreeClassifier(max_leaf_nodes=20),
                 reg_param: float = 1, shrinkage_scheme_: str = 'ridge'):
        """HSTree (Tree with hierarchical shrinkage applied).
        Hierarchical shrinkage is an extremely fast post-hoc regularization method which works on any decision tree (or tree-based ensemble, such as Random Forest).
        It does not modify the tree structure, and instead regularizes the tree by shrinking the prediction over each node towards the sample means of its ancestors (using a single regularization parameter).
        Experiments over a wide variety of datasets show that hierarchical shrinkage substantially increases the predictive performance of individual decision trees and decision-tree ensembles.
        https://arxiv.org/abs/2202.00858

        Params
        ------
        estimator_: sklearn tree or tree ensemble model (e.g. RandomForest or GradientBoosting)
            Defaults to CART Classification Tree with 20 max leaf nodes
            Note: this estimator will be directly modified

        reg_param: float
            Higher is more regularization (can be arbitrarily large, should not be < 0)
        
        shrinkage_scheme: str
            Experimental: Used to experiment with different forms of shrinkage. options are: 
                (i) node_based shrinks based on number of samples in parent node
                (ii) leaf_based only shrinks leaf nodes based on number of leaf samples 
                (iii) constant shrinks every node by a constant lambda
        """
        super().__init__()
        self.reg_param = reg_param
        self.estimator_ = estimator_
        self.shrinkage_scheme_ = shrinkage_scheme_
        if checks.check_is_fitted(self.estimator_):
            self._shrink()

    def get_params(self, deep=True):
        if deep:
            return deepcopy({'reg_param': self.reg_param, 'estimator_': self.estimator_,
                             'shrinkage_scheme_': self.shrinkage_scheme_})
        return {'reg_param': self.reg_param, 'estimator_': self.estimator_,
                'shrinkage_scheme_': self.shrinkage_scheme_}

    def fit(self, X, y, sample_weight=None, *args, **kwargs):
        # remove feature_names if it exists (note: only works as keyword-arg)
        feature_names = kwargs.pop('feature_names', None)  # None returned if not passed
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        self.estimator_ = self.estimator_.fit(X, y, *args, sample_weight=sample_weight, **kwargs)
        self._shrink()

        # compute complexity
        if hasattr(self.estimator_, 'tree_'):
            self.complexity_ = compute_tree_complexity(self.estimator_.tree_)
        elif hasattr(self.estimator_, 'estimators_'):
            self.complexity_ = 0
            for i in range(len(self.estimator_.estimators_)):
                t = deepcopy(self.estimator_.estimators_[i])
                if isinstance(t, np.ndarray):
                    assert t.size == 1, 'multiple trees stored under tree_?'
                    t = t[0]
                self.complexity_ += compute_tree_complexity(t.tree_)
        return self

    def _shrink_tree(self, tree, reg_param, i=0, parent_val=None, parent_num=None, cum_sum=0):
        if self.shrinkage_scheme_ == 'ridge':
            return self._shrink_tree_ridge(tree, reg_param, i, parent_val, parent_num, cum_sum)
        elif self.shrinkage_scheme_ == 'tv':
            return self._shrink_tree_tv(tree, reg_param, i, parent_val, parent_num, cum_sum)
        else:
            raise ValueError('shrinkage_scheme_ must be one of: ridge, tv')

    def _shrink_tree_ridge(self, tree, reg_param, i=0, parent_val=None, parent_num=None, cum_sum=0):
        """Shrink the tree
        """
        if reg_param is None:
            reg_param = 1.0
        left = tree.children_left[i]
        right = tree.children_right[i]
        is_leaf = left == right
        n_samples = tree.weighted_n_node_samples[i]
        if isinstance(self, RegressorMixin) or isinstance(self.estimator_, GradientBoostingClassifier):
            val = deepcopy(tree.value[i, :, :])
        else:  # If classification, normalize to probability vector
            val = tree.value[i, :, :] / n_samples

        # Step 1: Update cum_sum
        # if root
        if parent_val is None and parent_num is None:
            cum_sum = val

        # if has parent
        else:
            if self.shrinkage_scheme_ == 'ridge':
                val_new = (val - parent_val) / (1 + reg_param / parent_num)
            elif self.shrinkage_scheme_ == 'constant':
                val_new = (val - parent_val) / (1 + reg_param)
            else:  # leaf_based
                val_new = 0
            cum_sum += val_new

        # Step 2: Update node values
        if self.shrinkage_scheme_ == 'ridge' or self.shrinkage_scheme_ == 'constant':
            tree.value[i, :, :] = cum_sum
        else:  # leaf_based
            if is_leaf:  # update node values if leaf_based
                root_val = tree.value[0, :, :]
                tree.value[i, :, :] = root_val + (val - root_val) / (1 + reg_param / n_samples)
            else:
                tree.value[i, :, :] = val

                # Step 3: Recurse if not leaf
        if not is_leaf:
            self._shrink_tree_ridge(tree, reg_param, left,
                                    parent_val=val, parent_num=n_samples, cum_sum=deepcopy(cum_sum))
            self._shrink_tree_ridge(tree, reg_param, right,
                                    parent_val=val, parent_num=n_samples, cum_sum=deepcopy(cum_sum))

            # edit the non-leaf nodes for later visualization (doesn't effect predictions)

        return tree

    def _shrink_tree_tv(self, tree, reg_param, i=0, parent_val=None, parent_num=None, cum_sum=0):
        if reg_param is None:
            reg_param = 1.0
        # tree = copy.deepcopy(tree)
        edges, vertices, n_node = tree_to_graph(tree)

        weights = np.sqrt(n_node)  # / np.std(n_node)

        edge_matrix = create_connectivity_matrix(edges, len(vertices))

        shrunk_values = get_shrunk_nodes(node_values=vertices, edge_matrix=edge_matrix, reg_param=reg_param,
                                         weights=weights)
        for i, node in enumerate(tree.value):
            if type(self.estimator_) == DecisionTreeClassifier:
                tree.value[i] = np.array([shrunk_values[i] * n_node[i], (1 - shrunk_values[i]) * n_node[i]])
            else:
                tree.value[i] = shrunk_values[i]

        return tree

    def _shrink(self):
        if hasattr(self.estimator_, 'tree_'):
            self.estimator_.tree_ = self._shrink_tree(self.estimator_.tree_, self.reg_param)
        elif hasattr(self.estimator_, 'estimators_'):
            for t in self.estimator_.estimators_:
                if isinstance(t, np.ndarray):
                    assert t.size == 1, 'multiple trees stored under tree_?'
                    t = t[0]
                t.tree_ = self._shrink_tree(t.tree_, self.reg_param)

    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        if hasattr(self.estimator_, 'predict_proba'):
            return self.estimator_.predict_proba(X, *args, **kwargs)
        else:
            return NotImplemented

    def score(self, X, y, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(X, y, *args, **kwargs)
        else:
            return NotImplemented

    def __str__(self):
        s = '> ------------------------------\n'
        s += '> Decision Tree with Hierarchical Shrinkage\n'
        s += '> \tPrediction is made by looking at the value in the appropriate leaf of the tree\n'
        s += '> ------------------------------' + '\n'
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            return s + export_text(self.estimator_, feature_names=self.feature_names, show_weights=True)
        else:
            return s + export_text(self.estimator_, show_weights=True)

    def __repr__(self):
        # s = self.__class__.__name__
        # s += "("
        # s += "estimator_="
        # s += repr(self.estimator_)
        # s += ", "
        # s += "reg_param="
        # s += str(self.reg_param)
        # s += ", "
        # s += "shrinkage_scheme_="
        # s += self.shrinkage_scheme_
        # s += ")"
        # return s
        attr_list = ["estimator_", "reg_param", "shrinkage_scheme_"]
        s = self.__class__.__name__
        s += "("
        for attr in attr_list:
            s += attr + "=" + repr(getattr(self, attr)) + ", "
        s = s[:-2] + ")"
        return s


class HSTreeRegressor(HSTree, RegressorMixin):
    ...


class HSTreeClassifier(HSTree, ClassifierMixin):
    ...


# range from 0 to 100 each step ten time the previous one
reg_param_list = np.logspace(-6, 2, 100)


class HSTreeClassifierCV(HSTreeClassifier):
    def __init__(self, estimator_: BaseEstimator = None,
                 reg_param_list: List[float] = reg_param_list,
                 shrinkage_scheme_: str = 'ridge',
                 max_leaf_nodes: int = 20,
                 cv: int = 3, scoring=None, *args, **kwargs):
        """Cross-validation is used to select the best regularization parameter for hierarchical shrinkage.

         Params
        ------
        estimator_
            Sklearn estimator (already initialized).
            If no estimator_ is passed, sklearn decision tree is used

        max_rules
            If estimator is None, then max_leaf_nodes is passed to the default decision tree

        args, kwargs
            Note: args, kwargs are not used but left so that imodels-experiments can still pass redundant args.
        """
        if estimator_ is None:
            estimator_ = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
        super().__init__(estimator_, reg_param=None, shrinkage_scheme_=shrinkage_scheme_)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring
        self.shrinkage_scheme_ = shrinkage_scheme_
        # print('estimator', self.estimator_,
        #       'checks.check_is_fitted(estimator)', checks.check_is_fitted(self.estimator_))
        # if checks.check_is_fitted(self.estimator_):
        #     raise Warning('Passed an already fitted estimator,'
        #                   'but shrinking not applied until fit method is called.')

    def fit(self, X, y, *args, **kwargs):
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = HSTreeClassifier(deepcopy(self.estimator_), reg_param=reg_param,
                                   shrinkage_scheme_=self.shrinkage_scheme_)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        print("param selected:", self.reg_param, "shrinkage scheme:", self.shrinkage_scheme_)
        super().fit(X=X, y=y, *args, **kwargs)

    def __repr__(self):
        attr_list = ["estimator_", "reg_param_list", "shrinkage_scheme_",
                     "cv", "scoring"]
        s = self.__class__.__name__
        s += "("
        for attr in attr_list:
            s += attr + "=" + repr(getattr(self, attr)) + ", "
        s = s[:-2] + ")"
        return s


class HSTreeRegressorCV(HSTreeRegressor):
    def __init__(self, estimator_: BaseEstimator = None,
                 reg_param_list: List[float] = reg_param_list,
                 shrinkage_scheme_: str = 'ridge',
                 max_leaf_nodes: int = 20,
                 cv: int = 3, scoring=None, *args, **kwargs):
        """Cross-validation is used to select the best regularization parameter for hierarchical shrinkage.

         Params
        ------
        estimator_
            Sklearn estimator (already initialized).
            If no estimator_ is passed, sklearn decision tree is used

        max_rules
            If estimator is None, then max_leaf_nodes is passed to the default decision tree

        args, kwargs
            Note: args, kwargs are not used but left so that imodels-experiments can still pass redundant args.
        """
        if estimator_ is None:
            estimator_ = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
        super().__init__(estimator_, reg_param=None)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring
        self.shrinkage_scheme_ = shrinkage_scheme_
        # print('estimator', self.estimator_,
        #       'checks.check_is_fitted(estimator)', checks.check_is_fitted(self.estimator_))
        # if checks.check_is_fitted(self.estimator_):
        #     raise Warning('Passed an already fitted estimator,'
        #                   'but shrinking not applied until fit method is called.')

    def fit(self, X, y, *args, **kwargs):
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = HSTreeRegressor(deepcopy(self.estimator_), reg_param, shrinkage_scheme_=self.shrinkage_scheme_)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        print("param selected:", self.reg_param, "shrinkage scheme:", self.shrinkage_scheme_)
        super().fit(X=X, y=y, *args, **kwargs)

    def __repr__(self):
        attr_list = ["estimator_", "reg_param_list", "shrinkage_scheme_",
                     "cv", "scoring"]
        s = self.__class__.__name__
        s += "("
        for attr in attr_list:
            s += attr + "=" + repr(getattr(self, attr)) + ", "
        s = s[:-2] + ")"
        return s


def compare_time():
    # Load the iris dataset
    # X, y = make_friedman1(n_samples=1000)
    # # generate different regression targets
    # # y = X[:, 0] + np.sin(X[:, 1])
    # X = iris.data
    # y = iris.target

    # load a classification dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a decision tree on the training data
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Define a function to perform cost complexity pruning
    # Define a function to perform cost complexity pruning using cross-validation
    def cost_complexity_pruning_cv(reg, X_train, y_train):
        # Calculate the alpha values to test
        path = reg.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas

        # Train a series of regression trees with different alpha values
        regs = []
        for ccp_alpha in ccp_alphas:
            reg = DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alpha)
            reg.fit(X_train, y_train)
            regs.append(reg)

        # Evaluate the performance of each pruned tree using cross-validation
        cv_scores = []
        for reg in regs:
            scores = cross_val_score(reg, X_train, y_train, cv=5)
            cv_scores.append(np.mean(scores))

        # Find the alpha value that gives the best cross-validation score
        best_alpha = ccp_alphas[np.argmax(cv_scores)]
        print("Best alpha:", best_alpha)

        # Train a final regression tree with the best alpha value
        final_reg = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
        final_reg.fit(X_train, y_train)

        return final_reg

    s = time.time()
    # Perform cost complexity pruning on the decision tree
    pruned_clf = cost_complexity_pruning_cv(clf, X_train, y_train)
    total_time_ccp = time.time() - s

    s = time.time()
    hs_tv = HSTreeClassifierCV(deepcopy(clf), shrinkage_scheme_='tv')
    hs_tv.fit(X_train, y_train)
    total_time_tv = time.time() - s
    # get tv and ccp predictions on the test set and print the rmse
    # print("ccp:", np.sqrt(mean_squared_error(y_test, pruned_clf.predict(X_test))))
    # print("tv:", np.sqrt(mean_squared_error(y_test, hs_tv.predict(X_test))))
    # print("cart:", np.sqrt(mean_squared_error(y_test, clf.predict(X_test))))

    # print the auc for the test set for cart, ccp and tv
    print("ccp:", roc_auc_score(y_test, pruned_clf.predict_proba(X_test)[:, 1]))
    print("tv:", roc_auc_score(y_test, hs_tv.predict_proba(X_test)[:, 1]))
    print("cart:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    print("ccp:", total_time_ccp, "tv:", total_time_tv)
    # fig, ax = plt.subplots(1)
    # plot_tree(hs_tv.estimator_, ax=ax)
    # plt.show()


if __name__ == '__main__':
    compare_time()
    # np.random.seed(15)
    # # X, y = datasets.fetch_california_housing(return_X_y=True)  # regression
    # # X, y = datasets.load_breast_cancer(return_X_y=True)  # binary classification
    # X, y = datasets.load_diabetes(return_X_y=True)  # regression
    # # X = np.random.randn(500, 10)
    # # y = (X[:, 0] > 0).astype(float) + (X[:, 1] > 1).astype(float)
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.33, random_state=10
    # )
    # print('X.shape', X.shape)
    # print('ys', np.unique(y_train))
    #
    # # m = HSTree(estimator_=DecisionTreeClassifier(), reg_param=0.1)
    # # m = DecisionTreeClassifier(max_leaf_nodes = 20,random_state=1, max_features=None)
    # m = DecisionTreeRegressor(random_state=42, max_leaf_nodes=20)
    # # print('best alpha', m.reg_param)
    # m.fit(X_train, y_train)
    # # m.predict_proba(X_train)  # just run this
    # print('score', r2_score(y_test, m.predict(X_test)))
    # print('running again....')
    #
    # # x = DecisionTreeRegressor(random_state = 42, ccp_alpha = 0.3)
    # # x.fit(X_train,y_train)
    #
    # # m = HSTree(estimator_=DecisionTreeRegressor(random_state=42, max_features=None), reg_param=10)
    # # m = HSTree(estimator_=DecisionTreeClassifier(random_state=42, max_features=None), reg_param=0)
    # m = HSTreeClassifierCV(estimator_=DecisionTreeRegressor(max_leaf_nodes=10, random_state=1),
    #                        shrinkage_scheme_='node_based',
    #                        reg_param_list=[0.1, 1, 2, 5, 10, 25, 50, 100, 500])
    # # m = ShrunkTreeCV(estimator_=DecisionTreeClassifier())
    #
    # # m = HSTreeClassifier(estimator_ = GradientBoostingClassifier(random_state = 10),reg_param = 5)
    # m.fit(X_train, y_train)
    # print('best alpha', m.reg_param)
    # # m.predict_proba(X_train)  # just run this
    # # print('score', m.score(X_test, y_test))
    # print('score', r2_score(y_test, m.predict(X_test)))
