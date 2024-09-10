from sklearn.linear_model import Lasso

from ml_base_regression import MLBaseRegression

class LassoModel(MLBaseRegression):
    """Lasso regression model implementation."""
    
    def define_model(self):
        """Defines the Lasso model with the default configuration."""
        return Lasso()

    def get_param_grid(self):
        """Returns the hyperparameter grid for tuning the Lasso model."""
        return {'model__alpha': [0.001, 0.01, 0.1, 1, 10]}