import unittest
from unittest.mock import patch
import pandas as pd
from io import StringIO

from pandas_operations import PandasOperations as PO

class TestPO(unittest.TestCase):

    def test_init(self):
        # Test the __init__ method
        po = PO()
        self.assertIsNone(po.df)

    @patch('pandas.read_csv')
    def test_read_csv_pandas(self, mock_read_csv):
        # Create a mock DataFrame that is returned by pd.read_csv
        mock_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        
        # Setup so that mock_read_csv returns mock_df
        mock_read_csv.return_value = mock_df

        # Simulate a CSV file via StringIO
        csv_content = StringIO("col1,col2,col3\n1,4,7\n2,5,8\n3,6,9")
        
        # Instantiate the PO class with the simulated CSV content
        po = PO(df=csv_content)
        
        # Test the read_csv_pandas method
        df = po.read_csv_pandas()

        # Verify that the DataFrame is returned correctly
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)  # Check that the number of rows is correct
        self.assertEqual(list(df.columns), ['col1', 'col2', 'col3'])  # Check that the column names are correct

        # Check that mock_read_csv was called with the expected parameters
        mock_read_csv.assert_called()  # Verifies that pandas.read_csv was indeed called

        # Check that the DataFrame is added to the DF_LIST
        self.assertIn(df, PO.DF_LIST)

        # Check that mock_read_csv was called with the correct arguments
        mock_read_csv.assert_called_with(csv_content, encoding='utf-8', delimiter=',')

    def test_read_csv_pandas_failure(self):
        # Test the scenario where reading the CSV fails for all encodings and delimiters
        po = PO(df=StringIO("INVALID CONTENT"))

        with self.assertRaises(ValueError) as context:
            po.read_csv_pandas()

        # Verify that the ValueError contains appropriate error messages
        self.assertIn("Unable to read the file with the provided encodings and delimiters", str(context.exception))