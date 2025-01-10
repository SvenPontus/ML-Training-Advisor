import unittest
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from classifier_models import (LogisticRegressionModel as LoRM, 
                               KNNModel as KNM, SVCModel as SVCM)
from data_processing import DataProcessing as DP

class TestMlClassifier(unittest.TestCase):
    """Test cases for the classifier models."""

    def setUp(self):
        """Initial setup before each test."""
        self.df = DP("multi_c_test.csv") # import the csv file
        self.df.read_csv()
        self.X, self.y = self.df.prepare_for_ml(4) 

    def test_logistic_regression_model(self):
        """Test Logistic Regression model definition and param grid."""
        model = LoRM(self.X, self.y)
        self.assertIsInstance(model.define_model(), 
                              LogisticRegression().__class__) 
        self.assertEqual(model.get_param_grid(), {
            'model__solver': ['liblinear', 'saga', 'lbfgs'],
            'model__C': [0.1, 1.0, 10.0]
        })

    def test_knn_model(self):
        """Test K-Nearest Neighbors model definition and param grid."""
        model = KNM(self.X, self.y)
        self.assertIsInstance(model.define_model(), 
                              KNeighborsClassifier().__class__)
        self.assertEqual(model.get_param_grid(), {
            'model__n_neighbors': list(range(1, 30))
        }) 
    def test_svc_model(self):
        """Test Support Vector Classifier model definition and param grid."""
        model = SVCM(self.X, self.y)
        self.assertIsInstance(model.define_model(), SVC().__class__) 

        param_grid = model.get_param_grid()
        self.assertEqual(param_grid['model__kernel'], 
                         ['linear', 'poly', 'rbf', 'sigmoid'])
        self.assertEqual(param_grid['model__gamma'], ['scale', 'auto'])
        self.assertTrue(np.array_equal(param_grid['model__C'], 
                                       np.logspace(0, 1, 10)))
        self.assertTrue(np.array_equal(param_grid['model__degree'], 
                                       np.arange(1, 9)))
        
    def test_invalid_model(self):
        """Test invalid model definition."""
        with self.assertRaises(ValueError):
            model = LoRM("Wrong", "wrong")
