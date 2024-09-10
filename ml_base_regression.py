from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np


class MLBaseRegression(ABC):
    """Abstract base class for ML pipelines - regression"""
    
    def __init__(self, X, y, test_size=0.33, random_state=101):
        """Initialize the MLBase with training and test data."""
        (self.X_train, self.X_test, 
         self.y_train, self.y_test) = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.model = None
        self.best_model_params = None
        self.y_pred = None
        self.scaler = StandardScaler()

    @abstractmethod
    def define_model(self):
        """Define the ML model and any hyperparameters. Must be implemented in subclasses."""
        pass

    def train(self):
        """Train the model using the best hyperparameters."""
        pipeline = Pipeline([('scaler', self.scaler), ('model', self.define_model())])
        param_grid = self.get_param_grid()
        grid_search = self.grid_search_pipeline(pipeline, param_grid)
        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search
        self.best_model_params = grid_search.best_params_

    @abstractmethod
    def get_param_grid(self):
        """Define the hyperparameter grid for tuning."""
        pass

    def grid_search_pipeline(self, pipeline, param_grid):
        """Run grid search for the best hyperparameters."""
        return GridSearchCV(pipeline, param_grid=param_grid, cv=10)

    def predict(self):
        """Make predictions using the trained model."""
        if self.model:
            self.y_pred = self.model.predict(self.X_test)
        else:
            raise ValueError("Model not trained")

    
    def evaluate(self):
        """Evaluates the model using relevant metrics for regression."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)
        return f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2 Score: {r2}"

    def get_best_params(self):
        """Return the best hyperparameters after training."""
        return self.best_model_params