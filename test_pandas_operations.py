import unittest
from unittest.mock import patch, call
import pandas.testing as pdt
import pandas as pd

from pandas_operations import PandasOperations as PO

class TestPO(unittest.TestCase):

    def setUp(self):
        # Reset the DF_LIST before every test to ensure isolation
        PO.DF_LIST = []
    
    def tearDown(self):
        pass

    def test_init(self):
        # Test the __init__ method
        po = PO()
        self.assertIsNone(po.df)

    def test_read_csv_pandas(self):
        # Patcha pandas.read_csv so we can control what is returned
        with patch('pandas.read_csv') as mock_read_csv:
            # Create a mock DataFrame to return
            df = pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': [4, 5, 6],
                'col3': [7, 8, 9]
            })

            # Set up read_csv to return df
            mock_read_csv.return_value = df

            # Create a instance of your class (assuming it is named PO here)
            po = PO(df=None)

            # Call the method 
            result = po.read_csv_pandas()

            # Check that read_csv was called with the different combinations of encodings and delimiters
            expected_call = call(None, encoding='utf-8', delimiter=',')  
            mock_read_csv.assert_called_with(None, encoding='utf-8', delimiter=',')

            # Check that the returned DataFrame is the same as the one we created
            pdt.assert_frame_equal(result, df)

            # Check that the DataFrame was saved in the PandasOperations.DF_LIST
            self.assertIn(df, PO.DF_LIST)

    def test_read_csv_pandas_fails(self):
        # If all attempts to read the CSV fail
        with patch('pandas.read_csv', side_effect=Exception("Mocked failure")):
            po = PO(df=None)

            # Check that a ValueError is raised after all failed attempts
            with self.assertRaises(ValueError) as cm:
                po.read_csv_pandas()

            self.assertIn("Unable to read the file", str(cm.exception))


    def test_read_csv_pandas_failure(self):
        # Test the scenario where reading the CSV fails for all encodings and delimiters
        po = PO(df="INVALID CONTENT")

        with self.assertRaises(ValueError) as context:
            po.read_csv_pandas()

        # Verify that the ValueError contains appropriate error messages
        self.assertIn("Unable to read the file with the provided encodings and delimiters", str(context.exception))

    # Test control_re_or_cat
    def test_control_reg_or_cat(self):
        # Mock DataFrame
        mock_df_r = pd.DataFrame({
            "sales": [100, 200, 300],
            "1": [1, 2, 3],
            "2": [4, 5, 6],
            "3": [4, 5, 6],
        })

        mock_df_c = pd.DataFrame({
            "sales": ["high", "low", "medium"],
            "1": [1, 2, 3],
            "2": [4, 5, 6],
            "3": [4, 5, 6],
        })

        # Make sure DF_LIST is reset before the test
        PO.DF_LIST = [mock_df_r]

        # Create instance of PO without any arguments
        po_instance = PO()  

        # Test for continuous value (regression)
        result_reg = po_instance.control_reg_or_cat(1, "r")
        expected_reg = "It is a continuous value."
        self.assertEqual(result_reg, expected_reg)

        PO.DF_LIST = []
        PO.DF_LIST = [mock_df_c]

        # Test for categorical value (classification)
        result_cat = po_instance.control_reg_or_cat(0, "c")
        expected_cat = "It is a categorical value."
        self.assertEqual(result_cat, expected_cat)

        PO.DF_LIST = []
        PO.DF_LIST = [mock_df_r]
        
        # Test for incorrect type (neither "r" nor "c")
        result_invalid_type = po_instance.control_reg_or_cat(0, "c")
        expected_invalid_type = "ERROR! It is not a categorical value"
        self.assertEqual(result_invalid_type, expected_invalid_type)

        # TEST MORE...


        
