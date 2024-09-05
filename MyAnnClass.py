import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, \
                            classification_report, confusion_matrix, \
                            ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import joblib
import datetime
# Local library imports
from validation_myann import Validation as Vald

class MyAnnClass():
    """
    This class is a an Artificial Neural Network (ANN) class that can 
    be used for both regression and classification problems.
    And it can be used for binary and multi classification.

    Methods:
        build_model(final_model=False)
            - This method build the ANN model. If final_model is True, 
            the model will be trained on the entire dataset.

        model_losses()
            - This method will return the model losses.

        model_loss_regression()
            - This method will plot the model loss and accuracy for a regression model.

        y_pred_method()
           - This method will predict the y values for the test set.

        model_predict(user_row:np.array)
           - This method requires user input as a numpy array and returns the predicted value.

        mae_value()
           - This method will return the mean absolute error value.

        mse_value()
            - This method will return the mean squared error value.

        r2_value()
            - This method will return the r2 value.

        rmse_value()
            - This method will return the root mean squared error value.

        model_evaluate_classification()
           - This method will return the classification report and plot the confusion matrix for a 
             binary and multi classification model.

        scatter_plot()
            - This method will plot the real values in the target column against the predicted values.

        save_model(filename=None)
            - This method will save the model to a .h5 file and the scaler to a .pkl file.

        load_model(filename, load_scaler=False)
            - This method will load the model from a .h5 file and the scaler from a .pkl file.
          
    Properties:
        classes_
            - Returns the target label name.

        loss_
            - Returns the last recorded loss value from the training session.

        features_
            - A list of feature names from the input dataset.

        n_layers_
            - Reports the total number of layers in the model including the input and output layers.

        n_outputs_
            - Reports the number of outputs from the model, 
            which varies depending on the type of problem 
            (classification or regression).

        output_activation_
            - Returns the activation function used in the output layer.
    """
    def __init__(self, data_set:str, 
                 target:str,
                 patience:int = None,  
                 hidden_layer_sizes:tuple=(100,), 
                 activation:str = 'relu',
                 loss:str = 'mse', 
                 optimizer:str = 'adam', 
                 batch_size:int = 32, 
                 epochs:int = 1, 
                 monitor:str = 'val_loss', 
                 mode:str = 'auto', 
                 verbose:int = 1,
                 #use_multiprocessing:bool = False
                 ) -> None:
        
        """
        Parameters: data_set: str
                    this data is loaded and converted into a DataFrame.
                    It must be a csv file.

                    target: str
                    target is the target column in the DataFrame for supervised learning.

                    hidden_layer_sizes: tuple
                    hidden_layer_sizes is the number of hidden layers in the model. Default is (100,)
                    to get dropout layer, add a negative value in the tuple. 
                    Example: (-0.2, 100) -0.2 is dropout layer

                    activation: str
                    activation is the activation function in the model. Default is 'relu'
                    'relu', 'sigmoid', 'softmax' or 'tanh'

                    loss: str
                    loss is the loss function in the model. Default is 'mse'
                    other choice: 'binary_crossentropy' or 'categorical_crossentropy'

                    optimizer: str
                    optimizer is the optimizer in the model. Default is 'adam'

                    batch_size: int
                    batch_size is the batch size in the model. Default is 32

                    epochs: int
                    epochs is the number of epochs in the model. Default is 1

                    monitor: str
                    monitor is the monitor in the model. Default is 'val_loss'

                    patience: int
                    patience is the patience in the model. Default is 1

                    mode: str
                    mode is the mode in the model. Default is 'auto'

                    verbose: int
                    verbose is the verbose in the model. Default is 1

                    use_multiprocessing: bool
                    use_multiprocessing is the use_multiprocessing in the model. Default is False

                    Returns: None

        """

        # dataset
        # validate that data_set path and file type is a csv file
        # the data_set must be cleaned and ready for training
        if not Vald.validate_file_path(data_set):
            raise ValueError("File path must be a str and ending with '.csv'.")
        self.data_set = pd.read_csv(data_set)
        # target, validate that target is a string and only contain letters
        # target variable is the target column in the DataFrame for supervised learning
        if not Vald.validate_target(target):
            raise ValueError("target must be a string and only contain letters.") 
        self.target = target
        # For multi target classification, if not, its None value
        self.multi_target = None  
        # hidden_layer_sizes, validate that hidden_layer_sizes is a tuple and only contain int or float values
        # hidden_layer is the layers after the inut layer and before the output layer
        if not Vald.validate_tuple_int(hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must be a tuple and only contain int or float values.")
        self.hidden_layer_sizes = hidden_layer_sizes
        # activation, Validate that the input is a string and only contains the activation functions  
        # 'relu', 'sigmoid', 'softmax' or 'tanh'
        if not Vald.validate_activation(activation):
            raise ValueError("activation must be 'relu', 'sigmoid', 'softmax' or 'tanh'.")
        self.activation = activation
        # loss, Validate that the input is a string and contains 'mse','binary_crossentropy' or 'categorical_crossentropy'
        # This choice help if that is regressor, binary classifier or multi classifier 
        if not Vald.validate_loss(loss):
            raise ValueError("loss must be 'mse','binary_crossentropy' or 'categorical_crossentropy'.")
        self.loss = loss
        # optimizer
        # Validate that the input is a string and contains 'adam','sgd' or 'rmsprop'
        # default is 'adam'
        if not Vald.validate_optimizer(optimizer):
            raise ValueError("optimizer must be 'adam','sgd' or 'rmsprop'.")
        self.optimizer = optimizer
        # batch_size
        if not Vald.validate_int(batch_size):
            raise ValueError("batch_size must be an integer.")
        self.batch_size = batch_size
        # epochs 
        # how many times the model will be trained on the entire dataset
        if not Vald.validate_int(epochs):
            raise ValueError("epochs must be an integer.")
        self.epochs = epochs
        # monitor
        # val_loss for regressor and accuracy för classification
        if not Vald.validate_monitor(monitor):
            raise ValueError("monitor must be 'val_loss' or 'accuracy'.")
        self.monitor = monitor
        # patience
        if not Vald.validate_patience(patience):
            raise ValueError("patience must be an integer or None value")
        # this variabel is for early stopping
        if patience is None:
            self.patience = epochs  # Example of assigning epochs if patience is None
        else:
            self.patience = patience
        # mode
        if not Vald.validate_mode(mode):
            raise ValueError("mode must be 'auto', 'min' or 'max'.")
        self.mode = mode
        # verbose   
        if not Vald.validate_verbose(verbose):
        # verbose number can only be 0, 1 or 2, if 0 no output
            raise ValueError("verbose must be 0, 1 or 2.")
        self.verbose = verbose 

        #if not Vald.validate_bool(use_multiprocessing):
        #    raise ValueError("use_multiprocessing must be a boolean.")
        #self.use_multiprocessing = use_multiprocessing
        # Cant get this to work, so I have commented it out
        
        #           *Other Variables*
        self.scaler = MinMaxScaler()
        # If loss is categorical_crossentropy is a multi target classification, 
        # if not its a binary classification or regressor
        if loss == 'categorical_crossentropy':
            # Multi target classification
            # if multi classification, the target column is converted to dummies
            self.multi_target = pd.get_dummies(self.data_set[self.target]).astype('int8')
            self.y = self.multi_target.values
            self.X = self.data_set.drop([self.target], axis=1).values
        else:
        # if not multi classification, not dummies   
            self.X = self.data_set.drop(self.target, axis=1).values
            self.y = self.data_set[self.target].values
        # Train split
        test_size = 0.30
        # if more than 1000 rows, test_size is 0.20 if not 0.30
        if len(data_set) > 1000:
            test_size = 0.20
        (self.X_train, 
         self.X_test, 
         self.y_train, 
         self.y_test) = train_test_split(self.X, 
                                         self.y, 
                                         test_size=test_size, 
                                         random_state=101)
        # Scaling
        self.scaled_X_train = self.scaler.fit_transform(self.X_train)
        self.scaled_X_test = self.scaler.transform(self.X_test)
        self.losses = None
        self.model = None
        self.y_pred = None
        self.mse = None

    # Get out classes atributes from object
    @property
    def classes_(self):
    # Only target Label name, not names in label
        return self.target
    
    # Get out last loss value, Can only use after model_losses method
    @property
    def loss_(self):
        return self.losses['loss'].iloc[-1]
    
    # Get out feature atributes from object
    @property
    def features_(self):
        x = self.data_set.drop([self.target], axis=1)
        return x.columns.tolist()
    
    # Get out how many layers the model have
    @property
    def n_layers_(self):
        return len(self.hidden_layer_sizes) + 2
    
    # Get out numbers of outputs
    @property
    def n_outputs_(self):
        if self.loss == 'categorical_crossentropy':
            return self.multi_target.shape[1]
        else:
            return 1
    
    # Get out the output activation function
    @property
    def output_activation_(self):
        return self.model.layers[-1].activation.__name__
        
    def build_model(self, final_model=False):
        model = Sequential()
        # if final_model is True, the model will be trained on the entire dataset
        if final_model == True:
            scaled_X = self.scaler.fit_transform(self.X)
            model.add(Dense(units = scaled_X.shape[1], activation=self.activation))
        else:
        # if final_model False, the model will be trained on the train set
            model.add(Dense(units = self.scaled_X_train.shape[1], activation=self.activation))
        # hidden layers
        for units in self.hidden_layer_sizes:
        # if units is a positive integer add a Dense layer
            if units > 0:
                model.add(Dense(units, activation=self.activation))
        # if units is a negative integer add a Dropout layer
            elif units < 0:
                model.add(Dropout(abs(units)))
            else:
                raise ValueError("hidden_layer_sizes must be a tuple of integers and floats, lowest value -1 ")
            
        # depend on loss, its chooce the output layer
        if self.loss == 'binary_crossentropy':
        # Binary classification
            model.add(Dense(1, activation='sigmoid'))
        # Compile model 
            model.compile(optimizer = self.optimizer, loss = self.loss,  metrics = ['accuracy'])
        elif self.loss == 'categorical_crossentropy':
        # Multi target classification
            model.add(Dense(self.multi_target.shape[1], activation='softmax'))
            model.compile(optimizer = self.optimizer, loss = self.loss,  metrics = ['accuracy'])
        # regressor
        else:  
            model.add(Dense(1))  # Linear activation for regression output   
            model.compile(optimizer = self.optimizer, loss = self.loss,  metrics = ['mse'])
        
        # earlystopping
        early_stop = EarlyStopping(monitor=self.monitor, mode=self.mode, 
                                   verbose=self.verbose, patience=self.patience)
        if final_model == True:
            model.fit(scaled_X, self.y, epochs=self.epochs,
              batch_size=self.batch_size, validation_data=(scaled_X, self.y),
              verbose=self.verbose, 
              callbacks=[early_stop],
              #use_multiprocessing=self.use_multiprocessing
              )
        else:
            model.fit(self.scaled_X_train, self.y_train, epochs=self.epochs,
                batch_size=self.batch_size, validation_data=(self.scaled_X_test, self.y_test),
                verbose=self.verbose, 
                callbacks=[early_stop],
                #use_multiprocessing=self.use_multiprocessing
                )
            

        self.model = model

    ###Methods###
    # model_loss
    def model_losses(self):
        losses = pd.DataFrame(self.model.history.history)
        self.losses = losses

     # loss against epochs
    def model_loss_regression(self):
        """ Plot the model's loss and Mean Squared Error (MSE) for a regression model. """    
        if 'loss' in self.losses.columns and 'val_loss' in self.losses.columns:
            self.losses[['loss', 'val_loss']].plot()
            plt.title('Loss over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(['Training Loss', 'Validation Loss'])
            plt.show()
        else:
            print("[i] Error! Only regressor")

    # Predict scaled_X_test
    def y_pred_method(self):
        self.y_pred = self.model.predict(self.scaled_X_test)

    # predict user row
#    def model_predict(self, user_row:np.array, ):
#        """ This method requires user input as a numpy array and returns the predicted value. """
#        return self.model.predict(user_row)

    # predict user row, multi, binary or regressor
    def model_predict(self, user_row: np.array):
        """
        Predict the target value for input data.

        Parameters:
        user_row (np.array): The input data for prediction.

        Returns:
        np.array: The predicted target value(s).
        """
        predictions = self.model.predict(user_row)

        if self.loss == 'categorical_crossentropy':
        # For multi-class classification, return the class with the highest probability
            return np.argmax(predictions, axis=-1)
        elif self.loss == 'binary_crossentropy':
        # For binary classification, return 1 if probability > 0.5, else 0
            return np.where(predictions > 0.5, 1, 0)
        else:
        # For regression, return the predicted value
            return predictions
             
    # model evaluate regressor
    def mae_value(self):
        """ return mae value """
        return mean_absolute_error(self.y_test, self.y_pred)
       
    def mse_value(self):
        """ return mse value """
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        return self.mse
    
    def r2_value(self):
        """ return r2 value """
        return r2_score(self.y_test, self.y_pred)
           
    def rmse_value(self):
        """ return rmse value """
        return np.sqrt(self.mse)
    # classification report and confusion matrix
    def model_evaluate_classification(self):   
        """ This method will return the classification report and plot the 
        confusion matrix for a binary and multi classification model 
        """
        # Check if multi or binary
        # multi            
        if self.loss == 'categorical_crossentropy':
            y_pred = np.argmax(self.y_pred, axis=-1)
            y_test = np.argmax(self.y_test, axis=-1)
        else:
        # binary
            y_pred = np.where(self.y_pred > 0.5, 1, 0)
            y_test = self.y_test
        # classification report    
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        # confusion matrix 
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nDisplaying Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(6,6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        plt.show()

    # scatterplot, predict value. 
    def scatter_plot(self):
        """ This method will plot the predicted values """
        if self.y_test is None or self.y_pred is None:
            raise ValueError("You dont have information in y_test or y_pred")
        indices = np.arange(len(self.y_test))  
        plt.scatter(indices, self.y_test, color='blue', alpha=0.5, label='Actual Y')  
        plt.scatter(indices, self.y_pred, color='orange', alpha=0.5, label='Predicted Y') # pred values orange     
        plt.xlabel('Index')  
        plt.ylabel('Y Values') 
        plt.title('Comparison of Actual Y and Predicted Y')  
        plt.legend()  
        plt.grid(True)  
        plt.show() 

    # plot residual error plot
    def plot_residual_error(self):
        """ This method will plot the residual error """
        residuals = self.y_test - self.y_pred.reshape(-1)
        plt.scatter(self.y_pred, residuals, color='blue', alpha=0.5)  
        plt.xlabel('Predicted Y')  
        plt.ylabel('Residuals')  
        plt.title('Residual Error')  
        plt.axhline(y=0, color='red', linewidth=2)  
        plt.grid(True)  
        plt.show()

    # save_orginal keras model
    # I have choose keras model because .h5 not work for me
    def save_model(self, filename=None):
        """
            Save the model to a keras file and the scaler to a .pkl file.
            
            Parameters:
            filename (str): The path to the keras file where the model will be saved. Default is None.

        """
        if filename is None:
            # if filename is None, the model will be saved with the current date
            today = datetime.datetime.today().strftime('%Y-%m-%d')
            filename = f'model_{today}.keras'
        if not Vald.validate_model_name(filename):
            raise ValueError("filename must be a string and ending with '.keras'.")
        self.build_model(final_model=True)  
        self.model.save(filename)
        # save scaler package
        scaler_filename = filename.replace('.keras', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_filename)
        
    # load keras model
    @staticmethod
    def load_model(filename, load_scaler=False):
        """
            Load keras file -> model

            Parameters:
            filename: The path to the keras file containing the saved model.

            load_scaler(bool): If True, the scaler will be loaded. Default is False.

            Returns:
            model (tensorflow.keras.Model): The loaded Keras model.

            scaler MinMaxScaler. Only if load_scaler is True.
        """
        if not Vald.validate_model_name(filename):
            raise ValueError("filename must be a string and ending with 'keras'.")       
        try:
            model = load_model(filename)
        except:
            raise ValueError("Could not load model. Please check the file path and try again.")
        # if load_scaler is True, the scaler will be loaded
        if load_scaler:
            scaler_filename = filename.replace('.keras', '_scaler.pkl')
            try:
                scaler = joblib.load(scaler_filename)
            except:
                raise ValueError("Could not load scaler. Please check the file path and try again.")

        return model, scaler if load_scaler else model


