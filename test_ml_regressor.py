import unittest
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR

from regressor_models import (LinearRegressionModel as LRM, 
                              LassoModel as LM, 
                              RidgeModel as RM,
                              ElasticNetModel as ENM,
                              SVRModel as SVRM)

from data_processing import DataProcessing as DP

class TestMlRegressor(unittest.TestCase):
    """Test cases for the regressor models."""

    def setUp(self):
        self.df = DP("Adv.csv")
        self.df.read_csv()
        self.X, self.y = self.df.prepare_for_ml(3)

    def test_linear_regression(self):
        model = LRM(self.X, self.y)
        self.assertIsInstance(model.define_model(), LinearRegression().__class__)  
        self.assertEqual(model.get_param_grid(), {})  

    def test_lasso_model(self):
        model = LM(self.X, self.y)
        self.assertIsInstance(model.define_model(), Lasso().__class__) 
        self.assertEqual(model.get_param_grid(), {'model__alpha': [0.001, 0.01, 0.1, 1, 10]})

    def test_ridge_model(self):
        model = RM(self.X, self.y)
        self.assertIsInstance(model.define_model(), Ridge().__class__)
        self.assertEqual(model.get_param_grid(), {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})

    def test_elastic_net_model(self):
        model = ENM(self.X, self.y)
        self.assertIsInstance(model.define_model(), ElasticNet().__class__) 
        self.assertEqual(model.get_param_grid(), {
            'model__l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
            'model__alpha': [0.01, 0.1, 1, 10, 100],
            'model__max_iter': [10_000]
        })

    def test_svr_model(self):
        model = SVRM(self.X, self.y)
        self.assertIsInstance(model.define_model(), SVR().__class__) 

        param_grid = model.get_param_grid()
        
        self.assertEqual(param_grid['model__kernel'], ['linear', 'poly', 'rbf', 'sigmoid'])
        self.assertEqual(param_grid['model__gamma'], ['scale', 'auto'])
        self.assertTrue(np.array_equal(param_grid['model__C'], np.logspace(0, 1, 10)))
        self.assertTrue(np.array_equal(param_grid['model__degree'], np.arange(1, 9)))

    def test_invalid_model(self):
        """Test invalid model definition."""
        with self.assertRaises(ValueError):
            model = LRM("Wrong", "wrong")
            