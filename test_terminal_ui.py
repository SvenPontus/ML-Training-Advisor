from unittest import TestCase
from unittest.mock import patch
from io import StringIO
from terminal_ui import get_user_r_or_c

class TestTerminalUi(TestCase):

    # Test valid input "r" after 1st attempt
    @patch('builtins.input', side_effect=["r"])
    def test_get_user_r_or_c_first_attempt(self, mock_input):
        result = get_user_r_or_c()
        self.assertEqual(result, "r")

    # Test valid input "r" after 2 attempts (1st attempt wrong)
    @patch('builtins.input', side_effect=["x", "r"])
    def test_get_user_r_or_c_second_attempt(self, mock_input):
        result = get_user_r_or_c()
        self.assertEqual(result, "r")

    # Test valid input "r" after 3 attempts (1st and 2nd attempts wrong)
    @patch('builtins.input', side_effect=["x", "y", "r"])
    def test_get_user_r_or_c_third_attempt(self, mock_input):
        result = get_user_r_or_c()
        self.assertEqual(result, "r")

    # Test invalid input after 3 attempts, expecting ValueError
    @patch('builtins.input', side_effect=["x", "y", "z"])
    def test_get_user_r_or_c_after_three_wrong_attempts(self, mock_input):
        with self.assertRaises(ValueError):
            get_user_r_or_c()