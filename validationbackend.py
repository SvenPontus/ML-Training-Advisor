import os
import csv

class Validation:

    @staticmethod
    def control_csv(value):
        """Validate if the user input is a CSV file, if the file 
        exists, and if it has a valid CSV structure."""
        if value.endswith(".csv") and os.path.isfile(value):
            try:
                with open(value, 'r') as file:
                    # Check if it has a valid CSV structure
                    csv.Sniffer().sniff(file.read(1024))
                return value
            except (csv.Error, OSError) as e:
                raise ValueError(
                    f"The file exists but is not a valid CSV file: {e}")
        else:
            raise ValueError("You must choose an existing CSV file")

    @staticmethod
    def validate_alpha(value: str):
        """Validate if the user input is alphabetic"""
        if value.isalpha():
            return True
    