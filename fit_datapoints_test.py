import unittest
import fit_datapoints as tf
import pandas as pd
from sqlalchemy import create_engine, inspect


class UnitTests(unittest.TestCase):

    # CSV file is read successfully and data is assigned to the df
    def test_read_ideal_csv_file(self):
        file_name = 'ideal.csv'
        dataset = tf.DatasetCSV(file_name)
        self.assertEqual((400, 51), dataset.df.shape)

    def test_read_train_csv_file(self):
        file_name = 'train.csv'
        dataset = tf.DatasetCSV(file_name)
        self.assertEqual((400, 5), dataset.df.shape)

    # DatasetCSV object is created with an invalid file_name argument
    # (not a string)
    def test_DatasetCSV_with_invalid_file_name_argument(self):
        file_name = 12345
        with self.assertRaises(TypeError):
            tf.DatasetCSV(file_name)  # type: ignore

    # CSV file does not exist in the specified path
    def test_file_not_found(self):
        file_name = 'nonexistent.csv'
        with self.assertRaises(FileNotFoundError):
            tf.DatasetCSV(file_name)

    # Creating an instance of the class with valid arguments should
    # not raise any exceptions.
    def test_DataSetWithSQLTable_with_valid_arguments(self):
        try:
            engine = create_engine('sqlite:///:memory:')
            tf.DatasetWithSQLTable('train_data', 'train.csv', engine)
        except Exception as e:
            self.fail(f'Exception raised: {e}')

    # The DataFrame should be successfully written to an SQLite database table.
    def test_write_to_sql(self):
        engine = create_engine('sqlite:///:memory:')
        table_name = 'train_data'
        tf.DatasetWithSQLTable(table_name, 'train.csv', engine)
        self.assertTrue(inspect(engine).has_table(table_name))
        dataframe = pd.read_sql_table(table_name, con=engine)
        self.assertEqual(19.776287, dataframe.iloc[0, 4])

    # DatasetCSV object is created with an invalid argument
    def test_DataSetWithSQLTable_with_invalid_arguments(self):
        engine = create_engine('sqlite:///:memory:')
        with self.assertRaises(TypeError):
            tf.DatasetWithSQLTable(345, 'train.csv', engine)  # type: ignore


# dieses Skript im unittest-Kontext ausführen
if __name__ == '__main__':
    unittest.main()
