import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import sqlalchemy as db
from plot_data import plot_ideal_functions, plot_noisefree_functions
from import_data import import_datasets_to_sqllite_table, import_test_df
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
        if column != 'x':
            noisy_functions.append(df_noisy[column].tolist())
    ideal_functions = []
    for column in df_ideal.columns:
        if column != 'x':
            ideal_functions.append(df_ideal[column].tolist())

    # Initalize array to save the results
    result = np.zeros((50, 2), dtype=float)
    noise_free_functions = np.zeros((4, 2), dtype=int)

    # Loop over the functions and find the best match
    for j, f in enumerate(noisy_functions):
        # Iterate over the 50 ideal functions
        for i in range(50):
            # ...
            # trf and lm throw same result !
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

    return noise_free_functions


def find_best_fit_for_test_data(df_noisefree, df_test):
    # The drop() method removes the column named ‘x’ from the dataframe.
    # The axis=1 parameter specifies that we want to remove a column.
    # The inplace=True parameter specifies that we want to modify the
    # dataframe in place, rather than creating a new one.
    df_test.drop('x', axis=1, inplace=True)
    df_noisefree.drop('x', axis=1, inplace=True)

    result_df = pd.DataFrame()
    result_list = []
    for j in range(len(df_noisefree.columns)):

        temp = []
        for i in range(len(df_test)):
            if df_noisefree.iloc[:, j].apply(
                    lambda x: (x - df_test.iloc[i, 0])**2).min() < np.sqrt(2):
                temp.append(df_test.iloc[i, 0])
        result_list.append(temp)
        result_df = pd.DataFrame({df_noisefree.columns[j]: temp})
        print(f"n={len(temp)} bei Funktion {df_noisefree.columns[j]}")
        print(result_df)


if __name__ == '__main__':
    import_datasets_to_sqllite_table(dir_path)

    train_df, ideal_df = get_datasets_from_sql_database(dir_path)

    # plot_ideal_functions(ideal_df)

    noisefree_funcs_index = least_square_regression(
        ideal_df, train_df)  # noisefree_funcs_index has a (4, 2) shape

    # create df with the 4 new "ideal" functions instead of
    # the 4 noisy functions from the "train" dataset
    noisefree_df = pd.DataFrame(
        columns=['x'], index=range(400))
    noisefree_df['x'] = ideal_df.iloc[:, 0]
    for i in range(4):
        row_nr = noisefree_funcs_index[i, 1]
        noisefree_df['y'+str(row_nr)] = ideal_df.iloc[:, row_nr]
    # plot_noisefree_functions(noisefree_df, train_df)

    test_df = import_test_df(dir_path)
    find_best_fit_for_test_data(noisefree_df, test_df)

# for unittests:
# Überprüfen der Länge der Arrays
# print("Arraylänge=", len(noisy_f))  # should be 4
# for x in range(len(noisy_f)):
#     print(x, "Spaltenlänge=", len(noisy_f[x]))  # should be 400
# print("Arraylänge=", len(ideal_f))  # should be 50
# for x in range(len(ideal_f)):
#     print(x, "Spaltenlänge=", len(ideal_f[x]))  # should be 400
# print("Arraylänge=", len(noisefree_f))  # should be 4
# for x in range(len(noisefree_f)):
#     print(x, "Spaltenlänge=", len(noisefree_f[x]))  # should be 400
