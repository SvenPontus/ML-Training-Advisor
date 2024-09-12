import unittest
from unittest.mock import patch, ANY

from regressor_models import (LinearRegressionModel as LRM, 
                              LassoModel as LM, 
                              RidgeModel as RM,
                              ElasticNetModel as ENM,
                              SVRModel as SVRM)
from classifier_models import (LogisticRegressionModel as LoRM,
                               KNNModel as KNNM,
                               SVCModel as SVCM)

from ann_model import MyAnnClass as ANN

from data_processing import DataProcessing as DP
from run_ui import RunUI

class TestRunUI(unittest.TestCase):

    def setUp(self):
        self.run_ui = RunUI()
        # setup for the test
        self.run_ui.csv_file = DP("Adv.csv") 
        self.run_ui.r_or_c = "r"  # Simulate regressor mode
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        self.X, self.y = self.run_ui.csv_file.prepare_for_ml(3) 

    def tearDown(self) -> None:
        # clean
        self.run_ui = None

    # 1
    def test_get_r_or_c_input(self):
        with patch("builtins.input", side_effect=["r"]):
            self.run_ui.get_r_or_c_input()
            self.assertEqual(self.run_ui.r_or_c, "r")
    
    def test_invalid_get_r_or_c_input(self):
        with patch("builtins.input", side_effect=["x","c"]):
            with patch("builtins.print") as mock_print:
                self.run_ui.get_r_or_c_input()
                mock_print.assert_called_with(
                    "Invalid input. Only (r) or (c) Please try again.")

    # 2
    def test_read_in_csv(self):
        with patch("builtins.input", return_value="Adv.csv"):
            with patch(
                "data_processing.DataProcessing.read_csv"
                ) as mock_read_csv:
                self.run_ui.read_in_csv()
                mock_read_csv.assert_called_once()
    
    """
    # Problem..
    def test_invalid_read_in_csv(self):
        with patch("builtins.input", side_effect=["Adv","Adv.csv"]):
            with patch("builtins.print") as mock_print:
                self.run_ui.read_in_csv()
                mock_print.assert_called_with("You must choose an existing CSV file")
    """
    # 3
    def test_read_in_dependent_target(self):
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        with patch("builtins.input", side_effect=["3"]):
            with patch("builtins.print") as mock_print:
                self.run_ui.read_in_dependent_target()
                mock_print.assert_called_with("It is a continuous value.")


    def test_invalid_read_in_dependent_target(self):
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        with patch("builtins.input", side_effect=["x", "1"]):  
            with patch("builtins.print") as mock_print:
                self.run_ui.read_in_dependent_target()
                mock_print.assert_any_call(
                    "Invalid input. Please enter a number.")
                

    # 4
    def test_check_if_ready_for_ml(self):
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        self.run_ui.target = 3
        with patch(
            "data_processing.DataProcessing.check_data_for_ml", 
            return_value=True):
            with patch("builtins.print") as mock_print:
                with patch("builtins.input", return_value=""):  
                    result = self.run_ui.check_if_ready_for_ml()
                    mock_print.assert_called_with(
                        "Data is ready for machine learning.\n"
                    "\nPress enter or any key to continue.\n"
                    "And wait for results.")
                    self.assertTrue(result)
    # If not ready for ML
    def test_invalid_check_if_ready_for_ml(self):
        self.run_ui.csv_file = DP("multi_c_test.csv")
        self.run_ui.r_or_c = "r"
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        self.run_ui.target = 1
        with patch("data_processing.DataProcessing.check_data_for_ml", 
                   return_value=False):
            with patch("builtins.print") as mock_print:
                with patch("builtins.input", return_value=""):
                    self.run_ui.check_if_ready_for_ml()
                    mock_print.assert_called_with(
                        f"You need to fix this->{self.run_ui.csv_file.messages} \n"
                                                f"and run the program again.")
                                    
    # 4c
    # If model instance is correct
    def test_model_regressors(self):
        self.run_ui.target = 3 
        self.run_ui.start_ml()
        # Check if the selected model is an instance of the correct class
        self.assertIsInstance(self.run_ui.model_LRM, LRM)  
        self.assertIsInstance(self.run_ui.model_LM, LM)    
        self.assertIsInstance(self.run_ui.model_RM, RM)    
        self.assertIsInstance(self.run_ui.model_ENM, ENM) 
        self.assertIsInstance(self.run_ui.model_SVRM, SVRM)  
        self.assertIsInstance(self.run_ui.model_ANN, ANN)  



    # Test evaluate prints, and test best param print
    def test_print_report_regressors(self):
        """Test if the report is printed correctly"""
        with patch("builtins.print") as mock_print:
            self.run_ui.auto_use_model(LRM, self.X, self.y, 
                                       "Linear Regression Model")
            mock_print.assert_called_with(
                "\nModel: Linear Regression Model\n"
                "The best paramgrid: {}\n"
                "MAE: 1.237, MSE: 2.348, RMSE: 1.532, R2 Score: 0.923"
            )

    def test_print_report_lasso(self):
        """Test if the Lasso regression report is printed correctly"""
        with patch("builtins.print") as mock_print:
            self.run_ui.auto_use_model(LM, self.X, self.y, 
                                       "Lasso Regression Model")
            mock_print.assert_called_with(
                "\nModel: Lasso Regression Model\n"
                "The best paramgrid: {'model__alpha': 0.1}\n"
                "MAE: 1.256, MSE: 2.463, RMSE: 1.569, R2 Score: 0.919"
            )

    def test_print_report_ridge(self):
        """Test if the Ridge regression report is printed correctly"""
        with patch("builtins.print") as mock_print:
            self.run_ui.auto_use_model(RM, self.X, self.y, 
                                       "Ridge Regression Model")
            mock_print.assert_called_with(
                "\nModel: Ridge Regression Model\n"
                "The best paramgrid: {'model__alpha': 1}\n"
                "MAE: 1.241, MSE: 2.373, RMSE: 1.541, R2 Score: 0.922"
            )

    def test_print_report_elastic_net(self):
        """Test if the ElasticNet regression report is printed correctly"""
        with patch("builtins.print") as mock_print:
            self.run_ui.auto_use_model(ENM, self.X, self.y, 
                                       "ElasticNet Regression Model")
            mock_print.assert_called_with(
                "\nModel: ElasticNet Regression Model\n"
                "The best paramgrid: {'model__alpha': 0.1, "
                "'model__l1_ratio': 1, 'model__max_iter': 10000}\n"
                "MAE: 1.256, MSE: 2.463, RMSE: 1.569, R2 Score: 0.919"
            )

    def test_print_report_svr(self):
        """Test if the SVR regression report is printed correctly"""
        with patch("builtins.print") as mock_print:
            self.run_ui.auto_use_model(SVRM, self.X, self.y, "SVR Model")
            mock_print.assert_called_with(
                "\nModel: SVR Model\n"
                "The best paramgrid: {'model__C': 10.0, 'model__degree': 1, "
                "'model__gamma': 'scale', 'model__kernel': 'rbf'}\n"
                "MAE: 0.526, MSE: 0.588, RMSE: 0.767, R2 Score: 0.981"
            )

    """
    Problem, different report
    def test_print_report_ann(self):
        #Test if the ANN report is printed correctly
        X, y = self.run_ui.csv_file.prepare_for_ml(3)
        with patch("builtins.print") as mock_print:
            self.run_ui.auto_use_model(ANN, X, y, "Artificial Neural Network (ANN)")
            
            # Check if the correct model name was printed
            mock_print.assert_any_call("\nModel: Artificial Neural Network (ANN)")
            
            # Check if the evaluation metrics were printed (separately)
            mock_print.assert_any_call("\nMean Squared Error: " + ANY)
            mock_print.assert_any_call("\nR2 Score: " + ANY)
            mock_print.assert_any_call("\nMean Absolute Error: " + ANY)
            mock_print.assert_any_call("\nRoot Mean Squared Error (RMSE): " + ANY)
    """
    # Classifier
    def test_print_report_LoRM(self):
        """Test if the LoRM Classifier report is printed correctly"""
        self.run_ui.csv_file = DP("multi_c_test.csv") 
        self.run_ui.r_or_c = "c"  
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        self.X, self.y = self.run_ui.csv_file.prepare_for_ml(4)
        with patch("builtins.print") as mock_print:
            self.run_ui.auto_use_model(LoRM, self.X, self.y, "LoRM Model")
            mock_print.assert_called_with(
                "\nModel: LoRM Model\n"
                "The best paramgrid: {'model__C': 10.0, 'model__solver':"
                 " 'saga'}\nAccuracy: 0.98\nConfusion Matrix:\n"
                 "[[15  0  0]\n [ 0 21  1]\n [ 0  0 13]]\n"
                 "Classification Report:\n              "
                 "precision    recall  f1-score   support\n\n"
                 "      setosa       1.00      1.00      1.00        15\n"
                 "  versicolor       1.00      0.95      0.98        22\n"
                 "   virginica       0.93      1.00      0.96        13\n"
                 "\n    accuracy                           0.98        50\n"
                 "   macro avg       0.98      0.98      0.98        50\n"
                 "weighted avg       0.98      0.98      0.98        50\n")
            
    def test_print_report_KNNM(self):
        """Test if the KNNM Classifier report is printed correctly"""
        self.run_ui.csv_file = DP("multi_c_test.csv") 
        self.run_ui.r_or_c = "c"  
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        self.X, self.y = self.run_ui.csv_file.prepare_for_ml(4)
        with patch("builtins.print") as mock_print:
            self.run_ui.auto_use_model(KNNM, self.X, self.y, 
                                       "K-Nearest Neighbors (KNN)")
            mock_print.assert_called_with(
                "\nModel: K-Nearest Neighbors (KNN)\nThe best paramgrid: "
                "{'model__n_neighbors': 9}\nAccuracy: 1.0\n"
                "Confusion Matrix:\n[[15  0  0]\n [ 0 22  0]\n"
                " [ 0  0 13]]\nClassification Report:\n       "
                "       precision    recall  f1-score   support\n\n"
                "      setosa       1.00      1.00      1.00        15\n"
                "  versicolor       1.00      1.00      1.00        22\n"
                "   virginica       1.00      1.00      1.00        13\n"
                "\n    accuracy                           1.00        50\n"
                "   macro avg       1.00      1.00      1.00        50\n"
                "weighted avg       1.00      1.00      1.00        50\n")
            
    
    def test_print_report_SVCM(self):
        """Test if the Support Vector Classifier (SVC) 
        report is printed correctly"""
        self.run_ui.csv_file = DP("multi_c_test.csv") 
        self.run_ui.r_or_c = "c"  
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        self.X, self.y = self.run_ui.csv_file.prepare_for_ml(4)
        with patch("builtins.print") as mock_print:
            self.run_ui.auto_use_model(SVCM, self.X, self.y, "Support Vector Classifier (SVC)")
            mock_print.assert_called_with(
                "\nModel: Support Vector Classifier (SVC)\n"
                "The best paramgrid: {'model__C': 1.6681005372000588,"
                " 'model__degree': 1, 'model__gamma': 'scale',"
                " 'model__kernel': 'rbf'}\nAccuracy: 0.98\nConfusion"
                " Matrix:\n[[15  0  0]\n [ 0 21  1]\n [ 0  0 13]]\n"
                "Classification Report:\n              precision    "
                "recall  f1-score   support\n\n      setosa       "
                "1.00      1.00      1.00        15\n  versicolor    "
                "   1.00      0.95      0.98        22\n   virginica "
                "      0.93      1.00      0.96        13\n\n    accuracy"
                "                           0.98        50\n   macro avg "
                "      0.98      0.98      0.98        50\nweighted avg  "
                "     0.98      0.98      0.98        50\n")
            
    # Dump model
            

        



        

    

               
                    
            