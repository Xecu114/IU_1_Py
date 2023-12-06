import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import sqlalchemy as db
from plot_data import plot_ideal_functions, plot_noisefree_functions, \
    plot_testpoints_with_related_function
from import_data import import_csv_to_sqllite_table, \
    import_test_data_from_csv
import os
import logging

logging.basicConfig(level=logging.DEBUG)


def get_datasets_from_sql_database(path, db_name, db_table_name):
    """
    Retrieves a dataset from an SQL database and returns it
    as a pandas.DataFrames.

    Args:
        path (str): The path to the SQL database file.
        db_name (str): The name of the sqlite database.
        db_table_name (str): The name of the table.

    Returns:
        pandas.Dataframe: A Dataframe containing the data
            from the database table.

    Example Usage:
        train_df = get_datasets_from_sql_database(files_path,
                                            db_name, "train_data")
    """

    # Create a connection to the SQLite database
    engine = db.create_engine(f"sqlite:///{path}/{db_name}.db", echo=True)
    connection = engine.connect()
    # Query the databases and store the results in a pandas dataframe
    df = pd.read_sql_query(f"SELECT * FROM {db_table_name}", connection)
    # Close the database connection
    connection.close()
    return df


def least_square_regression(df_ideal, df_noisy):
    """
    Apply least squares regression to find the best fitting function in the
    ideal dataset for each function in the noisy dataset.

    Args:
        df_ideal (pandas DataFrame): The ideal dataset containing the
            x and y values of the functions.
        df_noisy (pandas DataFrame): The noisy dataset containing the
            x and y values of the functions.

    Returns:
        numpy array: An array containing the indices of the best fitting
            functions in the ideal dataset for each function in the noisy
            dataset. Array shape: (4, 2)
    """

    # Define a function, that calculates the sum of the squared deviations
    def residuals(p, y, x):
        return y - np.polyval(p, x)

    # Transforming the columns into Numpy-Arrays with the help of a loop
    noisy_functions = [df_noisy[column].tolist()
                       for column in df_noisy.columns if column != 'x']
    ideal_functions = [df_ideal[column].tolist()
                       for column in df_ideal.columns if column != 'x']

    # Initalize array to save the results
    result = np.zeros((50, 2), dtype=float)
    noise_free_functions = np.zeros((4, 2), dtype=int)

    # Loop over the functions and find the best match
    for j, f in enumerate(noisy_functions):
        # Iterate over the 50 ideal functions
        for i in range(50):
            p = least_squares(residuals, np.ones(3), method='trf',
                              args=(ideal_functions[i], np.arange(400)),
                              verbose=0).x
            # Note: method=... trf and lm throw same result

            # Save the results into an array
            result[i, 0] = i+1
            result[i, 1] = np.sum(
                (np.polyval(p, np.arange(400)) - f) ** 2)

        # Sort the result array after the sum of squared deviations
        result = result[result[:, 1].argsort()]

        # Save the best result into the new array
        noise_free_functions[j, 0] = j+1
        noise_free_functions[j, 1] = int(result[0, 0])

    return noise_free_functions


def find_best_fit_for_test_data(df_noisefree, df_test):
    """
    Finds the which test data points is nearby one of the four
    noise-free / ideal functions.

    Args:
        df_noisefree (pandas DataFrame): The DataFrame contains the four
            noise-free / ideal functions with their x and y values.
            shape: (400r x 5c)
        df_test (pandas DataFrame): The DataFrame contains the test points
            with their x and y values.
            shape: (100r x 2c)

    Returns:
        pandas.DataFrame: The DataFrame contains the test points that are
            nearby the four ideal functions. The DataFrame has the same x
            values as df_test and one column for each ideal function
            containing the points that fitted to this function. The columns
            are named after the ideal function with "_testpoints" appended
            to the column name.
            shape: (100r x 5c)
    """

    def is_point_nearby(function_df, testpoint):
        """
        Check if a given test point is nearby a specific function
        in the dataset (pandas.DataFrame).

        Args:
        function_df: A pandas.DataFrame containing the function
            with their x and y values.
        testpoint: The y-value of the test point.

        Returns:
        Boolean: Value indicating whether the test point is nearby the
            function.
        """

        # get the index of the x position in the 400 row DataFrames
        x_index = function_df.loc[function_df['x']
                                  == df_test.iloc[i, 0]].index[0]
        if i < 10:
            dist = function_df.iloc[x_index:(x_index+10), j].apply(
                lambda x: abs((x - testpoint))).min()
        else:
            dist = function_df.iloc[(x_index-10):(x_index+10), j].apply(
                lambda x: abs((x - testpoint))).min()
        temp_y = function_df.iloc[x_index, j]
        dist_min = ((temp_y*1.2)-temp_y)  # type: ignore
        if dist_min < 2:
            dist_min = 2
        return dist, (dist < dist_min)

    results_df = pd.DataFrame()
    results_df.loc[:, 'x'] = df_test.loc[:, 'x']
    fitted_testdp_df = df_test
    fitted_testdp_df['Delta Y (Deviation)'] = ''
    fitted_testdp_df['Nr. of the ideal function'] = ''

    for j in range(len(df_noisefree.columns)):
        if j > 0:
            c = str(df_noisefree.columns[j])+"_testpoints"
            for i in range(len(df_test)):
                delta_y, is_nearby = is_point_nearby(
                    df_noisefree, df_test.loc[i, 'y'])
                if is_nearby:
                    results_df.loc[i, c] = df_test.loc[i, 'y']

                    # store every matching function
                    if fitted_testdp_df.iloc[i, 3] == '':
                        fitted_testdp_df.iloc[i, 2] = \
                            str(round(delta_y, 3))
                        fitted_testdp_df.iloc[i, 3] = \
                            str(df_noisefree.columns[j])
                    else:
                        fitted_testdp_df.iloc[i, 2] = \
                            fitted_testdp_df.iloc[i, 2] + ', ' +\
                            str(round(delta_y, 3))
                        fitted_testdp_df.iloc[i, 3] = \
                            fitted_testdp_df.iloc[i, 3] + ', ' + \
                            str(df_noisefree.columns[j])

    fitted_testdp_df['Nr. of the ideal function'].replace(
        '', '-', inplace=True)
    fitted_testdp_df['Delta Y (Deviation)'].replace(
        '', '-', inplace=True)
    logging.debug(f"fitted_testdp_df:\n{fitted_testdp_df}")
    return results_df, fitted_testdp_df


if __name__ == '__main__':
    files_path = "Python_Course_IU"
    db_path = files_path + "/db"
    db_name = "my_db"
    if os.path.exists(f"{db_path}/{db_name}.db"):
        os.remove(f"{db_path}/{db_name}.db")
    elif not os.path.exists(db_path):
        os.makedirs(db_path)

    import_csv_to_sqllite_table(files_path, db_path)

    train_df = get_datasets_from_sql_database(
        files_path, db_name, "train_data")
    ideal_df = get_datasets_from_sql_database(
        files_path, db_name, "ideal_data")

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

    test_df = import_test_data_from_csv(files_path)
    functions_testdp_df, table3_df = find_best_fit_for_test_data(
        noisefree_df, test_df)
    logging.debug(f"df_test_cleaned:\n{functions_testdp_df}")
    # plot_testpoints_with_related_function(
    #     test_df, df_test_cleaned, noisefree_df)
    logging.debug("success")
