import numpy as np

from sklearn.svm import SVR
from sklearn.linear_model import (LinearRegression, 
                                  Lasso, 
                                  Ridge,
                                  ElasticNet)

from ml_base_regression import MLBaseRegression

# Linear Regression Model
class LinearRegressionModel(MLBaseRegression):
    def define_model(self):
        """Defines the Linear Regression model."""
        return LinearRegression()

    def get_param_grid(self):
        """Returns an empty parameter grid for Linear Regression as it 
        doesn't require hyperparameter tuning."""
        return {}

# Lasso Regression Model
class LassoModel(MLBaseRegression):
    """Lasso regression model implementation."""
    
    def define_model(self):
        """Defines the Lasso model with the default configuration."""
        return Lasso()

    def get_param_grid(self):
        """Returns the hyperparameter grid for tuning the Lasso model."""
        return {'model__alpha': [0.001, 0.01, 0.1, 1, 10]}

# Ridge Regression Model
class RidgeModel(MLBaseRegression):
    """Ridge regression model implementation."""
    
    def define_model(self):
        """Defines the Ridge model with the default configuration."""
        return Ridge()

    def get_param_grid(self):
        """Returns the hyperparameter grid for tuning the Ridge model."""
        return {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# ElasticNet Regression Model
class ElasticNetModel(MLBaseRegression):
    """ElasticNet regression model implementation."""
    
    def define_model(self):
        """Defines the ElasticNet model with the default configuration."""
        return ElasticNet()

    def get_param_grid(self):
        """Returns the hyperparameter grid for tuning the ElasticNet model."""
        return {
            'model__l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
            'model__alpha': [0.01, 0.1, 1, 10, 100],  
            'model__max_iter': [10_000]
        }

# SVR Model
class SVRModel(MLBaseRegression):
    """Support Vector Regression (SVR) model implementation."""
    
    def define_model(self):
        """Defines the SVR model with the default configuration."""
        return SVR()

    def get_param_grid(self):
        """Returns the hyperparameter grid for tuning the SVR model."""
        return {
            'model__C': np.logspace(0, 1, 10),
            'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'model__degree': np.arange(1, 9),
            'model__gamma': ['scale', 'auto']
        }
