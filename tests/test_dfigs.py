import unittest

import numpy as np

from sklearn.datasets import make_friedman1
from sklearn.tree import DecisionTreeRegressor

from imodels.tree.figs import FIGSRegressor
from imodels.tree.dfigs import D_FIGSRegressor


def get_data(n, phases):
    X, y = make_friedman1(n_samples=n, n_features=5, random_state=0)
    # for phase, phase_vars in phases.items():

    # split to train and test
    from sklearn.model_selection import train_test_split
    # fit cart model to X and y
    cart = DecisionTreeRegressor(max_depth=4)
    cart.fit(X, y)
    # get prediction as new y
    y = cart.predict(X)
    X[0:int(0.6 * X.shape[0]), [p for p in phases[1] if p not in phases[0]]] = np.nan
    return X, y


class TestD_FIGS(unittest.TestCase):
    def test_phase(self):
        phases = {0: [0, 1, 2], 1: [0, 1, 2, 3, 4]}

        X, y = get_data(1000, phases)
        # fit phase 1 figs
        figs = FIGSRegressor(max_rules=3, max_trees=1)
        figs.fit(X[:, phases[0]], y)
        figs_preds = figs.predict(X[:, phases[0]])
        print(figs)
        print("------------ moving to d_figs --------------")
        # fit d_figs
        d_figs = D_FIGSRegressor(phases=phases, max_rules=3, max_trees=1)
        d_figs.fit(X, y)
        print(d_figs)
        d_figs_preds = d_figs.predict(X)
        # get indices for which at least one phase 2 variable is nan
        phase_1_indices = np.where(np.isnan(X[:, phases[1]]).any(axis=1))[0]
        assert np.allclose(figs_preds[phase_1_indices], d_figs_preds[phase_1_indices])
        # check that d-figs fits the data better with phase 2 variables
        phase_2_indices = np.where(np.invert(np.isnan(X[:, phases[1]]).any(axis=1)))[0]
        y_phase_2 = y[phase_2_indices]
        d_figs_preds_phase_2 = d_figs_preds[phase_2_indices]
        figs_preds_phase_2 = figs_preds[phase_2_indices]

        rmse_d_figs = np.sqrt(np.mean((y_phase_2 - d_figs_preds_phase_2) ** 2))
        rmse_figs = np.sqrt(np.mean((y_phase_2 - figs_preds_phase_2) ** 2))

        assert rmse_d_figs < rmse_figs


if __name__ == '__main__':
    unittest.main()
