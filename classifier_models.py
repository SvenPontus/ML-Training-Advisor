from ml_base_classification import MLBaseClassification

from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(MLBaseClassification):
    def define_model(self):
        """Defines the Logistic Regression model."""
        return LogisticRegression() 

    def get_param_grid(self):
        """Returns the hyperparameter grid for Logistic Regression."""
        return {'model__solver': ['liblinear', 'saga', 'lbfgs'],
                'model__C': [0.1, 1.0, 10.0]}  # Use 'model__' prefix to match pipeline



