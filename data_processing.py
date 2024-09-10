import pandas as pd
import numpy as np

class DataProcessing:
    """Class for handling CSV data and preparing it for machine learning models."""
    
    def __init__(self, filepath=None):
        self.df = None
        self.filepath = filepath

    def read_csv(self):
        """Reads a CSV file and handles different encodings and delimiters."""
        encodings = ['utf-8', 'ISO-8859-1', 'latin1']
        delimiters = [',', ';', '|']
        error_messages = []

        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    self.df = pd.read_csv(self.filepath, encoding=encoding, delimiter=delimiter)
                    if not isinstance(self.df, pd.DataFrame) or self.df.empty:
                        raise ValueError("Invalid or empty DataFrame")
                    return self.df
                except Exception as e:
                    error_messages.append(f"Failed with encoding {encoding} and delimiter '{delimiter}': {e}")

        raise ValueError(f"Unable to read file. Errors: {error_messages}")

    def basic_info(self):
        """Provides basic information about the dataframe."""
        if self.df is not None:
            non_numeric_columns = [col for col in self.df.columns if not pd.api.types.is_numeric_dtype(self.df[col])]
            return {
                "Rows": len(self.df),
                "Columns": len(self.df.columns),
                "Memory (KB)": (self.df.memory_usage().sum() / 1024).round(1),
                "Non-Numeric Columns": non_numeric_columns if non_numeric_columns else "None"
            }
        else:
            raise ValueError("DataFrame not initialized")

    def prepare_for_ml(self, target_column_index):
        """Prepares data by splitting into X (features) and y (target), processes non-numeric columns."""
        if self.df is not None:
            y = self.df.iloc[:, target_column_index]
            X = self.df.drop(self.df.columns[target_column_index], axis=1)
            
            # Handle categorical columns (non-numeric)
            non_numeric_columns = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
            for col in non_numeric_columns:
                unique_values = X[col].nunique()
                if unique_values == 2:
                    X = pd.get_dummies(X, columns=[col], drop_first=True)  # Binary dummies for two unique values
                else:
                    raise ValueError(f"Column {col} has more than 2 unique values and needs encoding.")
            
            return X, y
        else:
            raise ValueError("DataFrame not initialized")

    def check_data_for_ml(self, target_column_index):
        """Checks if the data is ready for machine learning."""
        if self.df is not None:
            X = self.df.drop(self.df.columns[target_column_index], axis=1)
            messages = []

            # Check for NaN values
            nan_columns = X.columns[X.isna().any()].tolist()
            if nan_columns:
                messages.append(f"Columns with NaN values: {nan_columns}")

            # Check non-numeric columns
            non_numeric_columns = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
            for col in non_numeric_columns:
                unique_values = X[col].nunique()
                if unique_values > 2:
                    messages.append(f"Column {col} has {unique_values} unique values, which may need encoding.")
                elif unique_values == 2:
                    messages.append(f"Column {col} is binary and can be converted to dummy variables.")

            return messages if messages else "Data is ready for machine learning."
        else:
            raise ValueError("DataFrame not initialized")
    
    @staticmethod
    def control_reg_or_cat(target_value,r_or_c, df):
        column_name = df.columns[target_value]  
        column_dtype = df[column_name].dtype 
        if r_or_c == "r":
            # np.number have every int and float, I hope.. not shure 
            if np.issubdtype(column_dtype, np.number):
                return f"It is a continuous value."
            else:
                return f"ERROR! It is not a continuous value"

        # Check for classification task
        elif r_or_c == "c":
            if not np.issubdtype(column_dtype, np.number):
                return f"It is a categorical value."
            else:
                return f"ERROR! It is not a categorical value"

    # DONT NEED? 
    def split_data(self, X, y, test_size=0.33, random_state=101):
        """Splits data into training and testing sets."""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

