import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import sqlalchemy as db
from plot_data import plot_sql_data
from import_data import import_datasets_to_sqllite_table
import os


dir_path = "Python_Course_IU"
if os.path.exists(dir_path+"\\my_db.db"):
    os.remove(dir_path+"\\my_db.db")


def get_datasets_from_sql_database(path):
    # read data from sql tables into dataframes

    # Create a connection to the SQLite database
    engine = db.create_engine("sqlite:///"+path+"\\my_db.db", echo=True)
    connection = engine.connect()

    # Query the databases and store the results in a pandas dataframe
    train_df = pd.read_sql_query('SELECT * FROM train_data', connection)
    ideal_df = pd.read_sql_query('SELECT * FROM ideal_data', connection)

    # Close the database connection
    connection.close()

    return train_df, ideal_df


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def least_square_regression(df_noisy, df_ideal):
    # Apply least squares regression to each of the 4 functions in
    # the first (noisy) data set to find the best fitting function in the
    # second (ideal) data set.

    # Transforming the columns into Numpy-Arrays with a loop
    noisy_functions = []
    for column in df_noisy.columns:
        noisy_functions.append(df_noisy[column].to_numpy())
    noisy_functions.pop(0)  # delete x column

    ideal_functions = []
    for column in df_ideal.columns:
        ideal_functions.append(df_ideal[column].to_numpy())
    ideal_functions.pop(0)
    # Transform the array with 50 included functioons into a matrix
    ideal_matrix = np.matrix(ideal_functions)
    # Transponiere die Matrix
    transposed_matrix = ideal_matrix.T

    # Loop over the functions and find the best match
    for f in noisy_functions:
        # Define the function to be minimized
        # Transform the function array into a column
        column = np.matrix(f).T

        # Use numpy.linalg.lstsq to find the coefficients to each of
        # the 50 ideal functions
        coefficients = np.linalg.lstsq(
            transposed_matrix, column, rcond=None)[0]
        # print(coefficients)
        # get index and value of the function that has the coeff
        # which is nearest to zero
        nearest_to_zero_index, nearest_to_zero_coeff = find_nearest(
            coefficients, 0)
        # print(
        # f"Beste Funktion: f{nearest_to_zero_index+1}\n"
        # f"Coeff: {nearest_to_zero_coeff}")

    noise_free_function = noisy_functions
    return noise_free_function, noisy_functions, ideal_functions


if __name__ == '__main__':
    import_datasets_to_sqllite_table(dir_path)
    train_df, ideal_df = get_datasets_from_sql_database(dir_path)
    plot_sql_data(train_df, ideal_df)
    clean_funcs, noisy_funcs, ideal_funcs = least_square_regression(
        df_noisy=train_df, df_ideal=ideal_df)

# for unittests:
# Überprüfen der Länge der Arrays
# print("Arraylänge=", len(noisy_f))  # should be 4
# for x in range(len(noisy_f)):
#     print(x, "Spaltenlänge=", len(noisy_f[x]))  # should be 400
# print("Arraylänge=", len(ideal_f))  # should be 50
# for x in range(len(ideal_f)):
#     print(x, "Spaltenlänge=", len(ideal_f[x]))  # should be 400
# print("Arraylänge=", len(clean_f))  # should be 4
# for x in range(len(clean_f)):
#     print(x, "Spaltenlänge=", len(clean_f[x]))  # should be 400
