import unittest
from unittest.mock import patch

from validation_ml_program import Validation as Vald
from terminal_ui import (get_user_r_or_c_frontend, r_or_c_list,
                         csv_path_frontend)

# Backend Validation test cases
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
    
    # Task 2
    # Test cases for read in the csv path backend
    def test_validate_csv_path(self):
        """Test that a valid csv file path is accepted."""
        self.assertTrue(Vald.control_csv('data.csv'))
    
    def test_validate_csv_path_invalid(self):
        """Test that an invalid csv file path raises a ValueError."""
        with self.assertRaises(ValueError):
            Vald.control_csv('data.txt')
    
    def test_validate_csv_path_empty(self):
        """Test that an invalid csv file path raises a ValueError."""
        with self.assertRaises(ValueError):
            Vald.control_csv('')

# Frontend test cases terminal_ui
class TestFrontend(unittest.TestCase):

    def setUp(self):
        # Clear the list before each test
        r_or_c_list.clear()

    # Task 1
    # Test cases for get_user_r_or_c_frontend
    def test_run_app_invalid_then_valid_input(self):
        """Test the program with one incorrect input followed by a correct input.
        And check if the correct input is stored in the list."""
        with patch('builtins.input', side_effect=['x', 'c']):
            get_user_r_or_c_frontend()
        self.assertEqual(len(r_or_c_list), 1)
    



