import unittest
from unittest.mock import patch, call

from data_processing import DataProcessing as DP
from run_ui import RunUI

class TestRunUI(unittest.TestCase):

    def setUp(self):
        self.run_ui = RunUI()

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
                
                
                  
    


    

               
                    
            