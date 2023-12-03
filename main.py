import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import sqlalchemy as db
from scipy.spatial.distance import euclidean
from plot_data import plot_ideal_functions, plot_noisefree_functions, \
    plot_testpoints_with_related_function
from import_data import import_datasets_to_sqllite_table, import_test_df
import os
import math

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

    def is_point_nearby(df1, df2):
        if i < 10:
            dist = df1.iloc[i:(i+10), j].apply(
                lambda x: abs((x - df2.loc[i, 'y']))).min()
        else:
            dist = df1.iloc[(i-10):(i+10), j].apply(
                lambda x: abs((x - df2.loc[i, 'y']))).min()
        # print("x:", df1.loc[i, 'y'])
        temp_y = df1.iloc[i, j]
        dist_min = ((temp_y*1.2)-temp_y)  # type: ignore
        if dist_min < 2:
            dist_min = 2
        if dist < dist_min:
            # print("YYY")
            return True
        else:
            # print("----------")
            # print("x:", df_noisefree.iloc[i, 0])
            # print("y_f", df1.iloc[i, j])
            # print("dist:", round(dist, 1))
            # print("y:", df2.loc[i, 'y'])
            return False

    """ transform the df with the noisefree functions to the same size and
    x values as the test data """
    df_noisefree_temp = pd.DataFrame()
    df_noisefree_temp.loc[:, 'x'] = df_test.loc[:, 'x']
    for j in range(1, 5):
        for i in range(len(df_test)):
            # get the index of the actual x position in the 400 row dataframes
            x_index = df_noisefree.loc[df_noisefree['x']
                                       == df_test.iloc[i, 0]].index[0]
            df_noisefree_temp.loc[i, str(
                df_noisefree.columns[j])] = df_noisefree.iloc[x_index, j]

    df_results_old = pd.DataFrame()
    df_results_old.loc[:, 'x'] = df_test.loc[:, 'x']

    df_noisefree_nox2 = df_noisefree.drop('x', axis=1, inplace=False)
    print(df_noisefree_nox2)
    # df_noisefree_nox = df_noisefree_temp.drop('x', axis=1, inplace=False)
    # df_test_nox = df_test.drop('x', axis=1, inplace=False)

    results_df = pd.DataFrame()
    results_df.loc[:, 'x'] = df_noisefree.loc[:, 'x']

    test_df_temp = pd.DataFrame()
    test_df_temp.loc[:, 'x'] = df_noisefree.loc[:, 'x']
    for i in range(len(df_test)):
        # get the index of the actual x position in the 400 row dataframes
        x_index = df_noisefree.loc[df_noisefree['x']
                                   == df_test.iloc[i, 0]].index[0]
        test_df_temp.loc[x_index, 'y'] = df_test.iloc[i, 1]
    test_df_temp_nox = test_df_temp.drop('x', axis=1, inplace=False)

    # for j in range(len(df_noisefree_nox.columns)):
    j = 0
    c = str(df_noisefree_nox2.columns[j])+"_testpoints"
    for i in range(len(test_df_temp_nox)):
        if not math.isnan(test_df_temp_nox.loc[i, 'y']):  # type: ignore
            results_df.loc[i, 'y'] = test_df_temp_nox.loc[i, 'y']
            if is_point_nearby(df_noisefree_nox2, test_df_temp_nox):
                results_df.loc[i, c] = test_df_temp_nox.loc[i, 'y']
    return results_df


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
    df_test_cleaned = find_best_fit_for_test_data(
        noisefree_df, test_df)
    print(df_test_cleaned)
    # plot_testpoints_with_related_function(
    #     test_df, df_test_cleaned, noisefree_df)
