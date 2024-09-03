import unittest
from unittest.mock import patch

from validation_ml_program import Validation as Vald
from terminal_ui import get_user_r_or_c_frontend, r_or_c_list, validate_r_or_c

# Backend test cases
class TestValidation(unittest.TestCase):

    # Test cases for validate_r_or_c
    def test_validate_user_input_r(self):
        """Test that 'r' is a valid input."""
        self.assertTrue(Vald.validate_user_input_r_or_c('r'))

    def test_validate_user_input_c(self):
        """Test that 'c' is a valid input."""
        self.assertTrue(Vald.validate_user_input_r_or_c('c'))

    def test_validate_user_input_invalid(self):
        """Test that an invalid input raises a ValueError."""
        with self.assertRaises(ValueError):
            Vald.validate_user_input_r_or_c('1') 

    def test_validate_user_input_empty(self):
        """Test that an empty input raises a ValueError."""
        with self.assertRaises(ValueError):
            Vald.validate_user_input_r_or_c('')
    
    # Test cases for read in the csv path backend

# Frontend test cases
class TestFrontend(unittest.TestCase):

    # Test cases for get_user_r_or_c_frontend
    @patch('builtins.input', side_effect=['x', 'c'])
    def test_run_app_invalid_then_valid_input(self, mock_input):
        """Test the program with one incorrect input followed by a correct input.
        And check if the correct input is stored in the list."""
        get_user_r_or_c_frontend()
        self.assertEqual(len(r_or_c_list), 1)
    
    @patch('builtins.input', side_effect=['a', '3', 'X', 'd','r'])
    def test_run_app_multiple_invalid_then_valid_input(self, mock_input):
        """Test the program with four incorrect inputs followed by a correct input."""
        r_or_c_list.clear()  # Clear the list before running the test
        get_user_r_or_c_frontend()
        self.assertEqual(len(r_or_c_list), 1)

    
    @patch('builtins.input', side_effect=['a', '_____', 'z', '*', '?'])
    def test_run_app_terminate_after_max_attempts(self, mock_input):
        """Test if the system terminates after exceeding the maximum number of attempts."""
        # clean list
        r_or_c_list.clear()

        with self.assertRaises(SystemExit) as cm:
            get_user_r_or_c_frontend()

        self.assertEqual(cm.exception.code, 1)

        self.assertEqual(len(r_or_c_list), 0)

    # Test cases for read in the csv path frontend