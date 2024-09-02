from unittest import TestCase

from  validation_ml_program import Validation as Vald
from terminal_ui import get_user_r_or_c

class TestTerminalUi(TestCase):

    def test_get_user_r_or_c(self):
        expected = "r"
        self.assertEqual(expected, Vald.validate_r_or_c("r"))
