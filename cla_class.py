from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from ml_abc import MLBase

class Classifier(MLBase):
    """General classifier class that can handle multiple algorithms."""

    def __init__(self, X, y, model_type='logistic', **kwargs):
        """Initialize the classifier with a model type and optional hyperparameters."""
        super().__init__(X, y)
        self.model_type = model_type
        self.model_params = kwargs

    def define_model(self):
        """Dynamically define the classification model based on the selected type."""
        if self.model_type == 'logistic':
            return LogisticRegression(**self.model_params)
        elif self.model_type == 'knn':
            return KNeighborsClassifier(**self.model_params)
        elif self.model_type == 'svc':
            return SVC(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def get_param_grid(self):
        """Define hyperparameter grid based on model type."""
        if self.model_type == 'logistic':
            return {'model__solver': ['liblinear', 'saga', 'lbfgs']}
        elif self.model_type == 'knn':
            return {'model__n_neighbors': list(range(1, 31))}
        elif self.model_type == 'svc':
            return {
                'model__C': [0.1, 1, 10, 100],
                'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'model__gamma': ['scale', 'auto']
            }
        return {}

    def evaluate(self):
        """Evaluate the model using classification metrics."""
        self.predict()
        accuracy = accuracy_score(self.y_test, self.y_pred)
        confusion = confusion_matrix(self.y_test, self.y_pred)
        classification_rep = classification_report(self.y_test, self.y_pred)

        return {
            'Accuracy': accuracy,
            'Confusion Matrix': confusion,
            'Classification Report': classification_rep
        }
