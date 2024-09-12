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
        self.csv_file = None  # Object from DP
        self.df = None
        self.target = None
        self.right_file_best_model = None
        self.best_model_name = None

    def get_r_or_c_input(self):
        """get user r or c """
        print(
            "\nFind the best regressor or best "
            "classifier model for your data!\n"
            "- - - - - - - - - - - - - - - - - "
            "- - - - - - - - - - - - - - -\n"
            "Is it a regressor (r) or classifier "
            "(c) model you need for your data? "
            )
        while True:
            user_input = input(
                "Please choose 'r' for regressor or 'c' for classifier: ")

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
            except ValueError as e:
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
            if return_bool:
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
        if messege_or_ready:
            print("Data is ready for machine learning.\n"
                  "\nPress enter or any key to continue.\n"
                  "And wait for results.")
            input("")
            return True
        else:
            print(f"You need to fix this->{self.csv_file.messages} \n"
                  f"and run the program again.")

    def start_ml(self):
        if self.r_or_c == "r":
            X, y = self.csv_file.prepare_for_ml(self.target)
            r2_LRM, self.model_LRM, str_LRM = self.auto_use_model(
                LRM, X, y, "Linear Regression Model"
            )
            r2_LM, self.model_LM, str_LM = self.auto_use_model(
                LM, X, y, "Lasso Regression Model"
            )
            r2_RM, self.model_RM, str_RM = self.auto_use_model(
                RM, X, y, "Ridge Regression Model"
            )
            r2_ENM, self.model_ENM, str_ENM = self.auto_use_model(
                ENM, X, y, "ElasticNet Regression Model"
            )
            r2_SVRM, self.model_SVRM, str_SVRM = self.auto_use_model(
                SVRM, X, y, "SVR Model"
            )
            r2_ANN, self.model_ANN, str_ANN = self.auto_use_model(
                ANN, X, y, "Artificial Neural Network (ANN)"
            )
            best_model_name, best_r2 = self.calculate_best_r2_score(
                r2_LRM, r2_LM, r2_RM, r2_ENM, r2_SVRM, r2_ANN
            )
            self.best_model_name = best_model_name
            model_list = [
                (str_LRM, self.model_LRM),
                (str_LM, self.model_LM),
                (str_RM, self.model_RM),
                (str_ENM, self.model_ENM),
                (str_SVRM, self.model_SVRM),
                (str_ANN, self.model_ANN)
            ]

            for model_str, model_instance in model_list:
                if best_model_name in model_str:
                    self.right_file_best_model = model_instance
                    break
            print(f"\nBest model is: {best_model_name} "
                  f"with the best R2 score: {best_r2}.")

        elif self.r_or_c == "c":
            X, y = self.csv_file.prepare_for_ml(self.target)
            accuracy_LoRM, model_LoRM, str_LoRM = self.auto_use_model(
                LoRM, X, y, "Logistic Regression Model"
            )
            accuracy_KNNM, model_KNNM, str_KNNM = self.auto_use_model(
                KNNM, X, y, "K-Nearest Neighbors (KNN)"
            )
            accuracy_SVCM, model_SVCM, str_SVCM = self.auto_use_model(
                SVCM, X, y, "Support Vector Classifier (SVC)"
            )
            # The time is limited...
            # accuracy_ANN = self.auto_use_model(ANN(
            # loss='categorical_crossentropy'), X, y) .
            best_model_name, best_accuracy = self.calculate_best_accuracy_score(
                accuracy_LoRM, accuracy_KNNM, accuracy_SVCM)  # accuracy_ANN
            self.best_model = best_model_name
            model_list = [
                (str_LoRM, model_LoRM),
                (str_KNNM, model_KNNM),
                (str_SVCM, model_SVCM)
            ]

            for model_str, model_instance in model_list:
                if best_model_name in model_str:
                    self.right_file_best_model = model_instance
                    break
            print(f"\nBest model is: {best_model_name} "
                  f"with the best accuracy score: {best_accuracy}.")

    def auto_use_model(self, model, X, y, model_name):
        model = model(X, y)
        model.train()
        model.predict()
        evaluation = model.evaluate()
        print(f"\nModel: {model_name}\n{evaluation}")
        if self.r_or_c == "r":
            return model.r2_score, model, model_name
        elif self.r_or_c == "c":
            return model.accuracy, model, model_name

    def calculate_best_r2_score(self, LRM, LM, RM, ENM, SVRM, ANN):
        models = {
            'Linear Regression Model': LRM,
            'Lasso Regression Model': LM,
            'Ridge Regression Model': RM,
            'ElasticNet Regression Model': ENM,
            'SVR Model': SVRM,
            'Artificial Neural Network (ANN)': ANN
        }
        best_model = max(models, key=models.get)
        return best_model, models[best_model]

    def calculate_best_accuracy_score(self, LoRM, KNNM, SVCM):  # ANN
        models = {
            'Logistic Regression Model': LoRM,
            'K-Nearest Neighbors (KNN)': KNNM,
            'Support Vector Classifier (SVC)': SVCM,
            # 'ANN': ANN
        }
        best_model = max(models, key=models.get)
        return best_model, models[best_model]

    def dump_best_model(self):
        while True:
            user_input = input(f"\nDo you want to save the best model "
                               f"({self.best_model_name})? (y/n): ")
            if user_input.lower() == 'y':
                filename = input(
                    "Enter the filename to save the model: "
                ) + ".h5"

                try:
                    self.right_file_best_model.dump_model(filename)
                    print(
                        f"Model trained on full dataset and saved to "
                        f"{filename}")
                    break
                except Exception as e:
                    print(f"Error saving model: {e}")
            elif user_input.lower() == 'n':
                print("The model will not be saved.")
                break
            else:
                print(
                    "Invalid input. Please enter 'y' "
                    f"for yes or 'n' for no.")

    def run(self):
        self.get_r_or_c_input()
        self.read_in_csv()
        self.read_in_dependent_target()
        if self.check_if_ready_for_ml():
            self.start_ml()
            self.dump_best_model()
