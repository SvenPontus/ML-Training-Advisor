from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_abc import MLBase

class Regressor(MLBase):
    """General regressor class that can handle multiple algorithms."""

    def __init__(self, X, y, model_type='linear', **kwargs):
        """Initialize the Regressor with a model type and optional hyperparameters."""
        super().__init__(X, y)
        self.model_type = model_type
        self.model_params = kwargs

    def define_model(self):
        """Dynamically define the regression model based on the selected type."""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'lasso':
            return Lasso(**self.model_params)
        elif self.model_type == 'ridge':
            return Ridge(**self.model_params)
        elif self.model_type == 'svr':
            return SVR(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def get_param_grid(self):
        """Define hyperparameters grid based on model type."""
        if self.model_type == 'lasso':
            return {'model__alpha': [0.001, 0.01, 0.1, 1, 10]}
        elif self.model_type == 'ridge':
            return {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        elif self.model_type == 'svr':
            return {
                'model__C': [0.1, 1, 10, 100],
                'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'model__gamma': ['scale', 'auto']
            }
        # Linear regression has no hyperparameters
        return {}

    def evaluate(self):
        """Evaluate the model using regression metrics."""
        self.predict()
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(self.y_test, self.y_pred)

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2 Score': r2
        }