import unittest
from fit_datapoints import DatasetCSV, DatasetWithSQLTable, \
    lsr_to_fit_functions
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect


class UnitTests(unittest.TestCase):

    '''
    "DatasetCSV" class tests
    '''
    # CSV file is read successfully and data is assigned to the df

    def test_read_ideal_csv_file(self):
        file_name = 'ideal.csv'
        dataset = DatasetCSV(file_name)
        self.assertEqual((400, 51), dataset.df.shape)

    def test_read_train_csv_file(self):
        file_name = 'train.csv'
        dataset = DatasetCSV(file_name)
        self.assertEqual((400, 5), dataset.df.shape)

    # DatasetCSV object is created with an invalid file_name argument
    # (not a string)
    def test_DatasetCSV_with_invalid_file_name_argument(self):
        file_name = 12345
        with self.assertRaises(TypeError):
            DatasetCSV(file_name)  # type: ignore

    # CSV file does not exist in the specified path
    def test_file_not_found(self):
        file_name = 'nonexistent.csv'
        with self.assertRaises(FileNotFoundError):
            DatasetCSV(file_name)

    '''
    "DataSetWithSQLTable" class tests
    '''
    # Creating an instance of the class with valid arguments should
    # not raise any exceptions.

    def test_DataSetWithSQLTable_with_valid_arguments(self):
        try:
            engine = create_engine('sqlite:///:memory:')
            DatasetWithSQLTable('train_data', 'train.csv', engine)
        except Exception as e:
            self.fail(f'Exception raised: {e}')

    # DatasetCSV object is created with an invalid argument
    def test_DataSetWithSQLTable_with_invalid_arguments(self):
        engine = create_engine('sqlite:///:memory:')
        with self.assertRaises(TypeError):
            DatasetWithSQLTable(345, 'train.csv', engine)  # type: ignore

    # The DataFrame should be successfully written to an SQLite database table.
    def test_write_to_sql(self):
        engine = create_engine('sqlite:///:memory:')
        table_name = 'train_data'
        DatasetWithSQLTable(table_name, 'train.csv', engine)
        self.assertTrue(inspect(engine).has_table(table_name))
        dataframe = pd.read_sql_table(table_name, con=engine)
        self.assertEqual(19.776287, dataframe.iloc[0, 4])

    '''
    "least_square_regression" unit tests
    '''

    # The function should return a pandas DataFrame with the same shape as the
    # second dataset.
    def test_return_same_shape(self):
        # Create sample dataframes
        df_i = pd.DataFrame({'x': [1, 2, 3], 'y1': [1, 2, 3], 'y2': [4, 5, 6],
                            'y3': [1, 5, 20]})
        df_n = pd.DataFrame({'x': [1, 2, 3], 'y1': [4, 5, 6],
                             'y2': [1.01, 2.01, 3.01]})

        # Call the function
        result = lsr_to_fit_functions(df_i, df_n)

        # Check if the shape is the same
        self.assertEqual(result.shape, df_n.shape)

    # The returned DataFrame should have the x values from the ideal dataset.
    def test_x_values_from_ideal_dataset(self):
        # Create sample dataframes
        df_i = pd.DataFrame({'x': [1, 2, 3], 'y1': [1, 2, 3], 'y2': [4, 5, 6],
                            'y3': [1, 5, 20]})
        df_n = pd.DataFrame({'x': [1, 2, 3], 'y1': [4, 5, 6],
                             'y2': [1.01, 2.01, 3.01]})

        # Call the function
        result = lsr_to_fit_functions(df_i, df_n)

        # Check if the x values are from the ideal dataset
        self.assertEqual(result['x'].tolist(), df_i['x'].tolist())

    # The returned DataFrame should have the y values replaced by the new
    # "more ideal" functions.
    def test_y_values_replaced(self):
        # Create sample dataframes
        df_i = pd.DataFrame({'x': [1, 2, 3], 'y1': [1, 2, 3], 'y2': [4, 5, 6],
                            'y3': [1, 5, 20]})
        df_n = pd.DataFrame({'x': [1, 2, 3], 'y1': [4, 5, 6],
                             'y2': [1.01, 2.01, 3.01]})

        # Call the function
        result = lsr_to_fit_functions(df_i, df_n)

        # Check if the y values are replaced by the new "more ideal" functions
        self.assertEqual(result['y1'].tolist(), df_i['y1'].tolist())
        self.assertEqual(result['y2'].tolist(), df_i['y2'].tolist())

    # The function should handle empty datasets.
    def test_empty_datasets(self):
        # Create empty dataframes
        df_i = pd.DataFrame(columns=['x', 'y1', 'y2'])
        df_n = pd.DataFrame(columns=['x', 'y1', 'y2'])

        # Call the function
        result = lsr_to_fit_functions(df_i, df_n)

        # Check if the result is an empty dataframe
        self.assertTrue(result.empty)

    # The function should handle arguments with wrong data types
    def test_wrong_dataset_type(self):
        # Create empty dataframes
        df_i = pd.DataFrame(columns=['x', 'y1', 'y2'])
        df_n = ['x', 'y1', 'y2']

        # Call the function
        with self.assertRaises(TypeError):
            lsr_to_fit_functions(df_i, df_n)  # type: ignore

    # The function should handle datasets with missing values.
    def test_datasets_with_missing_values(self):
        # Create sample dataframes with missing values
        df_i = pd.DataFrame(
            {'x': [1, 2, 3], 'y1': [1, 2, 3], 'y2': [4, 5, 6]})
        df_n = pd.DataFrame(
            {'x': [1, 2, 3], 'y1': [7, np.nan, 9], 'y2': [10, 11, np.nan]})

        # Call the function
        with self.assertRaises(TypeError):
            lsr_to_fit_functions(df_i, df_n)

    # The function should handle datasets with wrong value types.
    def test_datasets_with_wrong_value_types(self):
        # Create sample dataframes with missing values
        df_i = pd.DataFrame(
            {'x': [1, 2, 3], 'y1': [1, "bad value", 3], 'y2': [4, 5, np.nan]})
        df_n = pd.DataFrame(
            {'x': [1, 2, 3], 'y1': [7, np.nan, 9], 'y2': [10, 11, np.nan]})

        # Call the function
        with self.assertRaises(TypeError):
            lsr_to_fit_functions(df_i, df_n)


# dieses Skript im unittest-Kontext ausführen
if __name__ == '__main__':
    unittest.main()
