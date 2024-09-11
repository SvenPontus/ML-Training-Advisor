from classifier_models import *
from regressor_models import *
from data_processing import DataProcessing as DP
from validationbackend import Validation as Vald

class RunUI():

    def __init__(self):
        self.r_or_c = None
        self.csv_file = None # Object from DP
        self.df = None
        self.target = None

    def get_r_or_c_input(self):
        """get user r or c """  
        print(
            "\nFind the best regressor or best classifier model for your data!\n"
            "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n"
            "Is it a regressor (r) or classifier (c) model you need for your data? "
        )
        while True:
            user_input = input("Please choose 'r' for regressor or 'c' for classifier: ")
            
            if user_input in ['r', 'c']:
                self.r_or_c = user_input
                break
            else:
                print("Invalid input. Only (r) or (c) Please try again.")
            
    def read_in_csv(self):
        """read in csv file"""
        while True:
            try:
                csv_name = input("Upload your csv file, dont forget .csv : ")
                self.csv_file = DP(Vald.control_csv(csv_name))
                self.df = self.csv_file.read_csv()
                break
            except Exception as e:
                print(e)

    def read_in_dependent_target(self):
        """read in dependent target"""

        for nr, columns in enumerate(self.df.columns):
            print(f"{nr} - {columns}")

        while True:
            try:
                self.target = int(input("What is the dependent target? "))
                if self.check_continuing_or_Categorical():
                    break
            except ValueError:
                print("Invalid input. Please enter a number.")  
    
    def check_continuing_or_Categorical(self):
        return_bool = DP.control_reg_or_cat(self.target, self.r_or_c, self.df)
        try:
            if return_bool == True:
                if self.r_or_c == "r":
                    print("It is a continuous value.")
                    return True
                elif self.r_or_c == "c":
                    print("It is a categorical value.")
                    return True
            else:
                print("Wrong target for this data type. Please try again.")
        except Exception as e:
            print(e)
    

    def check_if_ready_for_ml(self):
        messege_or_ready = self.csv_file.check_data_for_ml(self.target)
        if messege_or_ready == True:
            print("Data is ready for machine learning.")
        else:
            print(f"You need to fix this->{self.csv_file.messages} \n"
                  f"and run the program again.")
                 

            
    def run(self):
        self.get_r_or_c_input()
        self.read_in_csv()
        self.read_in_dependent_target()
        self.check_if_ready_for_ml()

