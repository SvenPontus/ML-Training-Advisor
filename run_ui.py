from classifier_models import (LogisticRegressionModel as LoRM, 
                               KNNModel as KNNM, 
                               SVCModel as SVCM)
from regressor_models import (LinearRegressionModel as LRM, 
                              LassoModel as LM, 
                              RidgeModel as RM,
                              ElasticNetModel as ENM,
                              SVRModel as SVRM
                              )
from ann_model import MyAnnClass as ANN
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
            print("Data is ready for machine learning.\n"
                  "\nPress enter or any key to continue.\n"
                  "And wait for results.")
            return True
        else:
            print(f"You need to fix this->{self.csv_file.messages} \n"
                  f"and run the program again.")
                 
    def start_ml(self):
        self.df
        if self.r_or_c == "r":            
            X, y = self.csv_file.prepare_for_ml(self.target)
            r2_LRM = self.auto_use_model(LRM, X, y)
            r2_LM = self.auto_use_model(LM, X, y)
            r2_RM = self.auto_use_model(RM, X, y)
            r2_ENM = self.auto_use_model(ENM, X, y)
            r2_SVRM = self.auto_use_model(SVRM, X, y)
            r2_ANN = self.auto_use_model(ANN, X, y)
            best_model, best_r2 = self.calculate_best_r2_score(r2_LRM, r2_LM, r2_RM, r2_ENM, r2_SVRM, r2_ANN)
            print(f"Best model is: {best_model} with the best r2 score: {best_r2}.")

        elif self.r_or_c == "c":
            X, y = self.csv_file.prepare_for_ml(self.target)
            accuracy_LoRM = self.auto_use_model(LoRM, X, y)
            accuracy_KNNM = self.auto_use_model(KNNM, X, y)
            accuracy_SVCM = self.auto_use_model(SVCM, X, y)
            #accuracy_ANN = self.auto_use_model(ANN(loss='categorical_crossentropy'), X, y) # Easy way, Limit with time
            best_model, best_accuracy = self.calculate_best_accuracy_score(accuracy_LoRM, accuracy_KNNM, accuracy_SVCM) # accuracy_ANN
            print(f"Best model is: {best_model} with the best accuracy score: {best_accuracy}.")
               
    def auto_use_model(self, model, X, y):
        model = model(X, y)
        model.train()
        model.predict()
        model.evaluate()
        if self.r_or_c == "r": 
            return model.r2_score
        elif self.r_or_c == "c":
            return model.accuracy
    
    def calculate_best_r2_score(self, LRM, LM, RM, ENM, SVRM, ANN):
        models = {
            'LRM': LRM,
            'LM': LM,
            'RM': RM,
            'ENM': ENM,
            'SVRM': SVRM,
            'ANN': ANN
        }
        best_model = max(models, key=models.get)
        return best_model, models[best_model]
    
    def calculate_best_accuracy_score(self, LoRM, KNNM, SVCM): # ANN
        models = {
            'LoRM': LoRM,
            'KNNM': KNNM,
            'SVCM': SVCM,
            #'ANN': ANN
        }
        best_model = max(models, key=models.get)
        return best_model, models[best_model]
    
    def run(self):
        self.get_r_or_c_input()
        self.read_in_csv()
        self.read_in_dependent_target()
        if self.check_if_ready_for_ml():
            self.start_ml()
        print(" ASK TO DUMP THE BEST MODEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

