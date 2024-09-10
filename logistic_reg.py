from sklearn.linear_model import LogisticRegression
from ml_abc import MLBase
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

class LogisticRegressionModel(MLBase):
    def define_model(self):
        """Defines the Logistic Regression model."""
        return LogisticRegression()

    def get_param_grid(self):
        """Returns the hyperparameter grid for Logistic Regression."""
        return {'model__solver': ['liblinear', 'saga', 'lbfgs'],
                'model__C': [0.1, 1.0, 10.0]}  # Use 'model__' prefix to match pipeline

    def train(self):
        """Train the model using the best hyperparameters."""
        pipeline = Pipeline([('scaler', self.scaler), ('model', self.define_model())])
        param_grid = self.get_param_grid()
        grid_search = self.grid_search_pipeline(pipeline, param_grid)
        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search
        self.best_model_params = grid_search.best_params_

    def evaluate(self):
        """Evaluates the model using classification metrics."""
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        accuracy = accuracy_score(self.y_test, self.y_pred)
        confusion = confusion_matrix(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        return f"Accuracy: {accuracy}\nConfusion Matrix:\n{confusion}\nClassification Report:\n{report}"
