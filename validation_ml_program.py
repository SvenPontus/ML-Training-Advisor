import pandas as pd

class Validation:
    """My Validation class"""
    @staticmethod
    def validate_user_input_r_or_c(value: str):
        """Validate if the user input is 'r' or 'c'."""    
        if value == "r" or value == "c":
            return True
        else:
            raise ValueError("You have to choose 'r' or 'c'")
    
    @staticmethod
    def control_csv(value):   
        """Validate if the user input is a csv file."""
        if value.endswith(".csv"):
            return value
        else:
            raise ValueError("You have to choose a csv file")
    """  
    @staticmethod
    def control_df(value):
        # Validate if the user input is a dataframe.
        if isinstance(value, pd.DataFrame):
            return value
        else:
            raise ValueError("You must provide a valid pandas DataFrame")
    """ 

    @staticmethod
    def validate_yes_or_no(value:str):
        return value in ["yes", "y", "no", "n"]
        
    @staticmethod
    def validate_str(value:str):
        return isinstance(value, str)

    @staticmethod
    def validate_int(value:int):
        return isinstance(int(value), int)
    
              
#  -  -  -  -  -  -  -  User input validation  -  -  -  -  -  -  -  -  #
    
    @staticmethod
    def read_in_str_value(validation_function, message:str):
        while True:
            user_input = input(message)
            if validation_function(user_input):
                return user_input
    
    @staticmethod
    def read_in_int_value(validation_function, message:str):
        while True:
            user_input = (input(message))
            if user_input.isnumeric() and validation_function(int(user_input)):
                return int(user_input)

#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  #




