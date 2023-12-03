import pandas as pd
import numpy as np


def find_best_fit_for_test_data(df_noisefree, df_test):

    def is_point_nearby():
        dist = abs(df_noisefree_nox.iloc[(i-3):(i+3), j].apply(
            lambda x: (x - df_test_nox.loc[i, 'y'])).min())
        print("----------")
        print("dist:", round(dist, 1))
        print("x:", df_test.iloc[i, 0])
        temp_y = df_noisefree_nox.iloc[i, j]
        print("< ...:", round((temp_y*np.sqrt(2))-temp_y, 1))
        if dist < ((temp_y*np.sqrt(2))-temp_y):
            print("YYY")
            return True

    """ transform the df with the noisefree functions to the same size and
    x values as the test data """
    # df_test_temp = pd.DataFrame()
    # df_test_temp.loc[:, 'x'] = df_noisefree.loc[:, 'x']
    # for i in range(len(df_test)):
    #     # get the index of the actual x position in the 400 row dataframes
    #     x_index = df_noisefree.loc[df_noisefree['x']
    #                              == df_test.iloc[i, 0]].index[0]
    #     df_test_temp.loc[x_index, 'y'] = df_test.iloc[i, 1]
    # # print(df_test_temp)
    """
    instead of making a new temp file, shrink the ideal function df
    """
    df_noisefree_temp = pd.DataFrame()
    df_noisefree_temp.loc[:, 'x'] = df_test.loc[:, 'x']
    for j in range(1, 5):
        for i in range(len(df_test)):
            # get the index of the actual x position in the 400 row dataframes
            x_index = df_noisefree.loc[df_noisefree['x']
                                       == df_test.iloc[i, 0]].index[0]
            df_noisefree_temp.loc[i, str(
                df_noisefree.columns[j])] = df_noisefree.iloc[x_index, j]

    print(df_noisefree_temp)

    df_results = pd.DataFrame()
    df_results.loc[:, 'x'] = df_test.loc[:, 'x']

    df_noisefree_nox = df_noisefree_temp.drop('x', axis=1, inplace=False)
    df_test_nox = df_test.drop('x', axis=1, inplace=False)
    # The drop() method removes the column named ‘x’ from the dataframe.
    # The axis=1 parameter specifies that we want to remove a column.
    # The inplace=True parameter specifies that we want to modify the
    # dataframe in place, rather than creating a new one.

    list_results = []

    # for j in range(len(df_noisefree_nox.columns)):
    j = 1
    c = str(df_noisefree_nox.columns[j])+"_testpoints"
    temp_list = []
    for i in range(len(df_test_nox)):
        temp_list = []
        # for i in range(len(df_test_nox)):
        #     df_results_old.loc[i, 'y'] = df_test_nox.loc[i, 'y']
        #     if is_point_nearby(df_noisefree_nox, df_test_nox):
        #         temp_list.append(df_test_nox.loc[i, 'y'])
        #         df_results_old.loc[i, c] = df_test_nox.loc[i, 'y']
        #         # print("i:", i)
        #         # print("testx:", df_test.iloc[i, 0])
        #         # print("testy:", df_test_nox.loc[i, 'y'])
        #         # print("noisefree_x:", df_noisefree_temp.iloc[i, 0])
        #         # print("noisefree_y:", df_noisefree_nox.iloc[i, j])
        #         # print("resultx:", df_results_old.iloc[i, 0])
        #         # print("resulty:", df_results_old.loc[i, c])
        # results_list.append(temp_list)
    # print(f"n={len(temp_list)} bei Funktion {df_noisefree_nox.columns[j]}")
    return df_results, list_results


# for unittests:
# Überprüfen der Länge der Arrays
# print("Arraylänge=", len(noisy_f))  # should be 4
# for x in range(len(noisy_f)):
#     print(x, "Spaltenlänge=", len(noisy_f[x]))  # should be 400
# print("Arraylänge=", len(ideal_f))  # should be 50
# for x in range(len(ideal_f)):
#     print(x, "Spaltenlänge=", len(ideal_f[x]))  # should be 400
