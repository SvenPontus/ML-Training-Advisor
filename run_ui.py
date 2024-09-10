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
        print(f"{self.csv_file.basic_info()}\n")
        for nr, columns in enumerate(self.df.columns):
              print(f"{nr} - {columns}")

        while True:
            try:
                self.target = int(input("What is the dependent target? "))
                break
            except Exception as e:
                print(e)
    
    def check_continuing_or_Categorical(self):
        a = DP.control_reg_or_cat(self.target, self.r_or_c, self.df)
        print(a) # Fixa med true och false snyggare i return, sen skapa
        # sen gör utfallen med print här i functionen
            

    def run(self):
        self.get_r_or_c_input()
        self.read_in_csv()
        self.read_in_dependent_target()
        self.check_continuing_or_Categorical()
