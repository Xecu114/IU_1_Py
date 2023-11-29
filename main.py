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


# function redundant !
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


# Define a function, that calculates the sum of the squared deviations
def residuals(p, y, x):
    return y - np.polyval(p, x)


def least_square_regression(df_ideal, df_noisy):
    # Apply least squares regression to each of the 4 functions in
    # the first (noisy) data set to find the best fitting function in the
    # second (ideal) data set.

    # Transforming the columns into Numpy-Arrays with the help of a loop
    noisy_functions = []
    for column in df_noisy.columns:
        noisy_functions.append(df_noisy[column].tolist())
    noisy_functions.pop(0)  # delete x column
    ideal_functions = []
    for column in df_ideal.columns:
        ideal_functions.append(df_ideal[column].tolist())
    ideal_functions.pop(0)

    # Initalize array to save the results
    result = np.zeros((50, 2), dtype=float)
    noise_free_functions = np.zeros((4, 2), dtype=int)

    # Loop over the functions and find the best match
    for j, f in enumerate(noisy_functions):
        # Iterate over the 50 ideal functions
        for i in range(50):
            #
            p = least_squares(residuals, np.ones(3), method='trf',
                              args=(ideal_functions[i], np.arange(400)),
                              verbose=0).x
            # Save the results into an array
            result[i, 0] = i+1
            result[i, 1] = np.sum(
                (np.polyval(p, np.arange(400)) - f) ** 2)

        # Sort the result array after the sum of squared deviations
        result = result[result[:, 1].argsort()]
        # print("result Array: ", result)

        # Save the best result into the new array
        noise_free_functions[j, 0] = j+1
        noise_free_functions[j, 1] = int(result[0, 0])

    return noise_free_functions, noisy_functions, ideal_functions


if __name__ == '__main__':
    import_datasets_to_sqllite_table(dir_path)
    train_df, ideal_df = get_datasets_from_sql_database(dir_path)
    plot_sql_data(train_df, ideal_df)
    clean_funcs_index, noisy_funcs, ideal_funcs = least_square_regression(
        df_ideal=ideal_df,
        df_noisy=train_df)
    print(clean_funcs_index)

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
