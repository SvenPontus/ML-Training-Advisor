import re

class Validation:
    """ Class for validation of input values from user in MyAnnClass """

    @staticmethod
    def validate_model_name(value:str):
        """ Validate that the input is a string and only contains the following model names """
        # Validate input from user to be a string and correct keras filename
        return isinstance(value, str) and value.endswith(".keras")

    @staticmethod
    def validate_file_path(file_path):
        """ Validate that the input is a string and only contains the following .csv file path """
        # absolute path to the file
        return isinstance(file_path, str) and file_path.endswith('.csv')
    
    @staticmethod
    def validate_target(value:str):
        """ Validate that the input is a string and only contains the following target values """
        pattern = r"^[0-9a-zA-Z_/\(\)\s]*$"
        return isinstance(value, str) and re.match(pattern, value) is not None
    
    @staticmethod
    def validate_patience(value:int):
        """ Validate that the input is an integer and only contains positive values """
        return value is None or isinstance(value, int)
    
    @staticmethod
    def validate_str_alpha(value:str):
        """ Validate that the input is a string and only contains alphabetic characters """
        return isinstance(value,str) and value.isalpha()
    
    # vill inte -1 och är det float så inte positivt
    @staticmethod 
    def validate_tuple_int(value:tuple):
        """ Validate that the input is a tuple and only contains int or float values """
        return isinstance(value, tuple) and all(isinstance(x, (int, float)) for x in value)
    
    @staticmethod
    def validate_bool(value:bool):
        """ Validate that the input is a boolean"""
        return isinstance(value, bool)
    
    @staticmethod
    def validate_activation(value:str):
        """ Validate that the input is a string and only contains the following activation functions """
        return value in ('relu', 'sigmoid', 'softmax', 'tanh')
    
    @staticmethod
    def validate_loss(value:str):
        """ Validate that the input is a string and only contains the following loss functions """
        return isinstance(value,str) and value in ('mse','binary_crossentropy','categorical_crossentropy')
    
    @staticmethod
    def validate_optimizer(value:str):
        """ Validate that the input is a string and only contains the following optimizers """
        return value in ('adam','sgd','rmsprop')
    
    @staticmethod
    def validate_monitor(value:str):
        """ Validate that the input is a string and only contains the following monitors """
        return value in ('val_loss','accuracy')
    
    @staticmethod
    def validate_int(value:int):
        return isinstance(int(value), int)
    
    @staticmethod
    def validate_mode(value:str):
        """ Validate that the input is a string and only contains the following modes """
        return value in ('auto', 'min', 'max')
    
    @staticmethod
    def validate_verbose(value:int):
        """ Validate that the input is an integer and only contains the following verbose values """
        return value in (0,1,2)
        
    
    @staticmethod
    def validate_str_numeric(value:str):
        return isinstance(value,str) and value.strip().isnumeric()

    @staticmethod
    def validate_str(value:str):
        return isinstance(value, str)

    





