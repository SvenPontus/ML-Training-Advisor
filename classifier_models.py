import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ml_base_classification import MLBaseClassification



# Logistic Regression Model
class LogisticRegressionModel(MLBaseClassification):
    """Logistic Regression model implementation."""

    def define_model(self):
        """Defines the Logistic Regression model."""
        return LogisticRegression() 

    def get_param_grid(self):
        """Returns the hyperparameter grid for Logistic Regression."""
        return {
            'model__solver': ['liblinear', 'saga', 'lbfgs'],
            'model__C': [0.1, 1.0, 10.0]
        }

# KNN Model
class KNNModel(MLBaseClassification):
    """K-Nearest Neighbors (KNN) model implementation."""
    
    def define_model(self):
        """Defines the KNN model."""
        return KNeighborsClassifier()

    def get_param_grid(self):
        """Returns the hyperparameter grid for tuning the KNN model."""
        return {
            'model__n_neighbors': list(range(1, 30))
        }

# SVC Model
class SVCModel(MLBaseClassification):
    """Support Vector Classifier (SVC) model implementation."""

    def define_model(self):
        """Defines the SVC model."""
        return SVC()

    def get_param_grid(self):
        """Returns the hyperparameter grid for tuning the SVC model."""
        return {
            'model__C': np.logspace(0, 1, 10),
            'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'model__degree': np.arange(1, 9),
            'model__gamma': ['scale', 'auto']
        }
