from sklearn.linear_model import LinearRegression
from ml_base_regression import MLBaseRegression

class LinearRegressionModel(MLBaseRegression):
    def define_model(self):
        """Defines the Linear Regression model."""
        return LinearRegression()

    def get_param_grid(self):
        """Returns an empty parameter grid for Linear Regression as it doesn't require hyperparameter tuning."""
        return {}

        
