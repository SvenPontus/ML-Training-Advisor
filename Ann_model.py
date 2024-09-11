import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.multiclass import type_of_target # GOOD, Maybe fix the problem with the type_of_target
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class MyAnnClass:
    """Artificial Neural Network (ANN) class with a unified interface for classification and regression."""
    
    def __init__(self, X, y, hidden_layer_sizes=(100,), activation='relu', loss='mse',
                 optimizer='adam', batch_size=32, epochs=100, test_size=0.33, patience=10, verbose=0):
        """
        Initializes the MyAnnClass with given parameters for building an ANN.
        
        Parameters:
        X : np.array
            Features for the model.
        y : np.array
            Target for the model (either classification or regression).
        hidden_layer_sizes : tuple
            Tuple defining the number of units in each hidden layer. 
            Example: (100,) means one hidden layer with 100 units.
        activation : str
            Activation function for the hidden layers. Default is 'relu'.
        loss : str
            Loss function. 'mse' for regression, 'binary_crossentropy' or 'categorical_crossentropy' for classification.
        optimizer : str
            Optimizer used to minimize the loss. Default is 'adam'.
        batch_size : int
            Number of samples per gradient update.
        epochs : int
            Number of epochs to train the model.
        test_size : float
            Fraction of the dataset to be used for testing.
        patience : int
            Number of epochs with no improvement after which training will be stopped.
        verbose : int
            Verbosity mode (0, 1, or 2).
        """
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        
        self.scaler = MinMaxScaler()
        self.model = None
        self.y_pred = None

        self.r2_score = None

        # Scale the data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def define_model(self):
        """Defines and compiles the ANN model."""
        model = Sequential()

        # Input layer
        model.add(Dense(self.X_train_scaled.shape[1], activation=self.activation))

        # Hidden layers
        for units in self.hidden_layer_sizes:
            if units > 0:
                model.add(Dense(units, activation=self.activation))
            elif units < 0:
                model.add(Dropout(abs(units)))
            else:
                raise ValueError("hidden_layer_sizes must be a tuple of positive integers or floats.")

        # Output layer
        if self.loss == 'binary_crossentropy':
            model.add(Dense(1, activation='sigmoid'))  # Binary classification
        elif self.loss == 'categorical_crossentropy':
            model.add(Dense(self.y_train.shape[1], activation='softmax'))  # Multi-class classification
        else:
            model.add(Dense(1))  # Regression (default linear activation)

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'] if 'crossentropy' in self.loss else ['mse'])
        return model

    def train(self):
        """Trains the ANN model."""
        early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=self.verbose)

        self.model = self.define_model()
        self.model.fit(self.X_train_scaled, self.y_train, validation_data=(self.X_test_scaled, self.y_test),
                       epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=[early_stop])

    def predict(self):
        """Makes predictions on the test set."""
        if not self.model:
            raise ValueError("The model has not been trained yet.")
        
        self.y_pred = self.model.predict(self.X_test_scaled)
        
        # For classification, convert probabilities to class labels
        if self.loss == 'binary_crossentropy':
            self.y_pred = np.where(self.y_pred > 0.5, 1, 0)
        elif self.loss == 'categorical_crossentropy':
            self.y_pred = np.argmax(self.y_pred, axis=1)

    def evaluate(self):
        """Evaluates the model and returns performance metrics."""
        if self.y_pred is None or len(self.y_pred) == 0:
            self.predict()

        if 'crossentropy' in self.loss:
            accuracy = accuracy_score(self.y_test, self.y_pred)
            confusion = confusion_matrix(self.y_test, self.y_pred)
            report = classification_report(self.y_test, self.y_pred)
            return f"Accuracy: {accuracy}\nConfusion Matrix:\n{confusion}\nClassification Report:\n{report}"
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(self.y_test, self.y_pred)
            self.r2_score = r2_score(self.y_test, self.y_pred)
            return f"Mean Squared Error: {mse}\nR2 Score: {self.r2_score}"

    def get_best_params(self):
        """Not applicable for ANN, but provided for compatibility."""
        return "No hyperparameter tuning used for ANN."

