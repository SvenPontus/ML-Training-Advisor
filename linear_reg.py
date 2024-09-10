from sklearn.linear_model import LinearRegression
from ml_abc import MLBase
import numpy as np

class LinearRegressionModel(MLBase):
    def define_model(self):
        """Defines the Linear Regression model."""
        return LinearRegression()

    def get_param_grid(self):
        """Returns an empty parameter grid for Linear Regression as it doesn't require hyperparameter tuning."""
        return {}

    def evaluate(self):
        """Evaluates the model using relevant metrics for regression."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)
        return f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2 Score: {r2}"
