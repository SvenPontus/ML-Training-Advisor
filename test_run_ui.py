import unittest
from unittest.mock import patch, call

from data_processing import DataProcessing as DP
from run_ui import RunUI

class TestRunUI(unittest.TestCase):

    def setUp(self):
        self.run_ui = RunUI()
        # Maybe will see....
        #self.run_ui.csv_file = DP("Adv.csv") 
        #self.run_ui.r_or_c = "r"  # Simulate regressor mode
        #self.run_ui.df = self.run_ui.csv_file.read_csv()

    # 1
    def test_get_r_or_c_input(self):
        with patch("builtins.input", side_effect=["r"]):
            self.run_ui.get_r_or_c_input()
            self.assertEqual(self.run_ui.r_or_c, "r")
    
    def test_invalid_get_r_or_c_input(self):
        with patch("builtins.input", side_effect=["x","c"]):
            with patch("builtins.print") as mock_print:
                self.run_ui.get_r_or_c_input()
                mock_print.assert_called_with("Invalid input. Only (r) or (c) Please try again.")

    # 2
    def test_read_in_csv(self):
        with patch("builtins.input", return_value="Adv.csv"):
            with patch("data_processing.DataProcessing.read_csv") as mock_read_csv:
                self.run_ui.read_in_csv()
                mock_read_csv.assert_called_once()
    

    """    def test_invalid_read_in_csv(self):
        with patch("builtins.input", side_effect=["Adv","Adv.csv"]):
            with patch('data_processing.DataProcessing.read_csv', side_effect=Exception("You must choose an existing CSV file")) as mock_control_csv:
                with patch("builtins.print") as mock_print:
                    self.run_ui.read_in_csv()
                    mock_print.assert_called_with("You must choose an existing CSV file")
    """
    
    


    # 3
    def test_read_in_dependent_target(self):
        self.run_ui.csv_file = DP("Adv.csv")
        self.run_ui.r_or_c = "r"
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        with patch("builtins.input", side_effect=["3"]):
            with patch("builtins.print") as mock_print:
                self.run_ui.read_in_dependent_target()
                mock_print.assert_called_with("It is a continuous value.")


    def test_invalid_read_in_dependent_target(self):
        self.run_ui.csv_file = DP("Adv.csv")
        self.run_ui.r_or_c = "r"  # Simulate regressor mode
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        with patch("builtins.input", side_effect=["x", "1"]):  
            with patch("builtins.print") as mock_print:
                self.run_ui.read_in_dependent_target()
                mock_print.assert_any_call("Invalid input. Please enter a number.")
                

    # 4
    def test_check_if_ready_for_ml(self):
        self.run_ui.csv_file = DP("Adv.csv")
        self.run_ui.r_or_c = "r"  # Simulate regressor mode
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        self.run_ui.target = 3
        with patch("data_processing.DataProcessing.check_data_for_ml", return_value=True):
            with patch("builtins.print") as mock_print:
                self.run_ui.check_if_ready_for_ml()
                mock_print.assert_called_with("Data is ready for machine learning.\n"
                  "\nPress enter or any key to continue.\n"
                  "And wait for results.")

    def test_invalid_check_if_ready_for_ml(self):
        self.run_ui.csv_file = DP("multi_c_test.csv")
        self.run_ui.r_or_c = "r"
        self.run_ui.df = self.run_ui.csv_file.read_csv()
        self.run_ui.target = 1
        with patch("data_processing.DataProcessing.check_data_for_ml", return_value=False):
            with patch("builtins.print") as mock_print:
                self.run_ui.check_if_ready_for_ml()
                mock_print.assert_called_with(f"You need to fix this->{self.run_ui.csv_file.messages} \n"
                                              f"and run the program again.")
                
    # 4 C... Test here and ml files

                
                  
    


    

               
                    
            