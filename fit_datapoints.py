import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import os
import logging
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral11, Bokeh5
from bokeh.models import Range1d

logging.basicConfig(level=logging.DEBUG)

files_path = 'Python_Course_IU'
my_db_path = files_path + '/db/my_db.db'
td_filename = 'test.csv'
engine = create_engine(f'sqlite:///{my_db_path}', echo=False)

# define superclass for datasets and define one subclass for...


class DatasetCSV():
    '''
    A class to represent a dataset imported from a CSV file.

    Attributes:
    - file_name (str): the name of the CSV file.
    - df (pd.DataFrame): the DataFrame containing the data from the CSV file.
    '''

    def __init__(self, file_name: str):
        if not isinstance(file_name, str):
            raise TypeError("file_name must be a string")
        self._file_name = file_name
        self.df = pd.DataFrame()
        self.get_dataframe_from_csv()

    def get_dataframe_from_csv(self):
        '''
        Reads the CSV file and assigns the data to the df attribute.
        '''
        file_path = os.path.join(files_path, self._file_name)
        if os.path.isfile(file_path):
            self.df = pd.read_csv(file_path)
            logging.debug(file_path + ' - file read successfully')
        else:
            raise FileNotFoundError("File not found:", file_path)


class DatasetWithSQLTable(DatasetCSV):
    '''
    A class to represent a dataset imported from a CSV file and
    an export to a SQLite database table.

    Attributes:
    - file_name (str): the name of the CSV file.
    - table_name (str): the name of the SQLite db table.
    - df (pd.DataFrame): the DataFrame containing the data from the CSV file.
    '''

    def __init__(self, table_name: str,
                 file_name: str, engine):
        if not isinstance(file_name, str):
            raise TypeError("file_name must be a string")
        if not isinstance(table_name, str):
            raise TypeError("table_name must be a string")
        super().__init__(file_name)
        self._table_name = table_name
        self.write_data_to_sql(engine)

    def write_data_to_sql(self, engine):
        self.df.to_sql(self._table_name, con=engine,
                       if_exists='replace', index=False)
        logging.debug(self._table_name + ' - written Data to SQL')


def least_square_regression(df_ideal, df_noisy):
    '''
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
    '''

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
            result[i, 1] = np.linalg.norm(
                (np.polyval(p, np.arange(400)) - f) ** 2)

        # Sort the result array after the sum of squared deviations
        sorted_indices = np.argsort(result[:, 1])
        result = result[sorted_indices]

        # Save the best result into the new array
        noise_free_functions[j, 0] = j+1
        noise_free_functions[j, 1] = int(result[0, 0])

    logging.debug('least_square_regression - done')
    return noise_free_functions


def find_best_fit_for_test_data(df_noisefree, df_test):
    '''
    Checks to which of the four noise-free / ideal functions each test
    datapoint can be approximated.

    Args:
        df_noisefree (pandas DataFrame): The DataFrame needs to contain the
            four noise-free / ideal functions with their x and y values.
            shape: (400r x 5c)
        df_test (pandas DataFrame): The DataFrame needs to contain the test
            points with their x and y values.
            shape: (100r x 2c)

    Returns:
        results_df (pandas.DataFrame):
            The DataFrame contains the test points that are
            nearby the four ideal functions. The DataFrame has the same x
            values as df_test and one column for each ideal function
            containing the points that fitted to this function. The columns
            are named after the ideal function with '_testpoints' appended
            to the column name.
            shape: (100r x 5c)
        fitted_testdp_df (pandas.DataFrame):
            The DataFrame extends the original test data DataFrame
            by 3 columns:

            "Delta Y (Deviation)": shows the distance to the nearest function
            "Nr. of the ideal function": shows the name of the nearest function
            "Nr. of all fitting functions": shows the names of all functions,
                that are nearby
            shape: (100r x 5c)
    '''

    def is_point_nearby(function_df, testpoint):
        '''
        Check if a given test point is nearby a specific function
        in the dataset (pandas.DataFrame).

        Args:
        function_df: A pandas.DataFrame containing the function
            with their x and y values.
        testpoint: The y-value of the test point.

        Returns:
        Boolean: Value indicating whether the test point is nearby the
            function.
        '''

        # get the index of the x position in the 400 row DataFrames
        x_index = function_df.loc[function_df['x']
                                  == df_test.iloc[i, 0]].index[0]
        if i < 10:
            dist = function_df.iloc[x_index:(x_index+2), j].apply(
                lambda x: abs((x - testpoint))).min()
        else:
            dist = function_df.iloc[(x_index-2):(x_index+2), j].apply(
                lambda x: abs((x - testpoint))).min()
        temp_y = function_df.iloc[x_index, j]
        dist_min = ((temp_y*1.3)-temp_y)  # type: ignore
        if dist_min < 0.5:
            dist_min = 0.5
        return dist, (dist < dist_min)

    results_df = pd.DataFrame()
    results_df.loc[:, 'x'] = df_test.loc[:, 'x']
    fitted_testdp_df = df_test
    fitted_testdp_df['Delta Y (Deviation)'] = np.NaN
    fitted_testdp_df['Nr. of the ideal function'] = '-'
    fitted_testdp_df['Nr. of all fitting functions'] = ''

    for j in range(len(df_noisefree.columns)):
        if j > 0:
            c = str(df_noisefree.columns[j])+'_testpoints'
            for i in range(len(df_test)):
                delta_y, is_nearby = is_point_nearby(
                    df_noisefree, df_test.loc[i, 'y'])
                if is_nearby:
                    results_df.loc[i, c] = df_test.loc[i, 'y']

                    # store every matching function
                    if fitted_testdp_df.iloc[i, 4] == '':
                        fitted_testdp_df.iloc[i, 2] = \
                            round(delta_y, 3)
                        fitted_testdp_df.iloc[i, 3] = \
                            str(df_noisefree.columns[j])
                        fitted_testdp_df.iloc[i, 4] = \
                            str(df_noisefree.columns[j])
                    else:
                        # fitted_testdp_df.iloc[i, 2] = \
                        #     fitted_testdp_df.iloc[i, 2] + ', ' +\
                        #     str(round(delta_y, 3))

                        # choose best match by comparing the distances
                        if delta_y < fitted_testdp_df.iloc[i, 2]:
                            fitted_testdp_df.iloc[i, 2] = \
                                round(delta_y, 3)
                            fitted_testdp_df.iloc[i, 3] = \
                                str(df_noisefree.columns[j])

                        fitted_testdp_df.iloc[i, 4] = \
                            fitted_testdp_df.iloc[i, 4] + ', ' + \
                            str(df_noisefree.columns[j])

    # fitted_testdp_df['Delta Y (Deviation)'].replace(
    #     '', '-', inplace=True)
    fitted_testdp_df['Nr. of all fitting functions'].replace(
        '', '-', inplace=True)
    logging.debug('find_best_fit_for_test_data - done')
    return results_df, fitted_testdp_df


def plot_ideal_funcs():
    '''
    Generates line charts representing all columns of the DataFrame using the
    Bokeh library. Each column is displayed in a different color and given
    its own label. Because the 'ideal' dataset is so big, it is divided
    into 5 seperate DataFrames that each contain 10 functions

    Args:
        None

    Returns:
        None

    Example Usage:
        plot_ideal_funcs()
    '''
    output_file('ideal_data_diagram.html')

    # split the ideal dataframe into 5 dataframes with 10 functions each
    # shape splits from [400 r x 51 c] to 5x [400 r x 10 c]
    dfs = [ideal_dataset.df.iloc[:, i:i+10] for i in range(1, 50, 10)]

    for df in dfs:
        plot = figure(width=1200, height=900,
                      title='ideal.csv Line Plot' + str(df.index),
                      x_axis_label='x', y_axis_label='y')
        min_max_values = ideal_dataset.df['x'].agg(['min', 'max'])
        plot.x_range = Range1d(min_max_values.iloc[0], min_max_values.iloc[1])
        for i, column in enumerate(df.columns):
            plot.line(ideal_dataset.df.iloc[:, 0], df[column],
                      line_color=Spectral11[i % len(Spectral11)],
                      legend_label=str(column))
        plot.legend.location = 'top_left'
        show(plot)  # type: ignore


def plot_noisefree_funcs(df):
    '''
    Generates a line plot of two sets of data using the Bokeh library.
    The first set of data is the new ideal functions that match the noisy
    train functions, which are the second set of data.

    Args:
        df (DataFrame): The DataFrame containing the ideal or 'noisefree'
            data. It should contain at the the column 'x' and one column
            for each of the four functions.

    Returns:
        None

    Example Usage:
        plot_noisefree_functions(noisefree_df)
    '''
    output_file('noisefree_data_diagram.html')
    plot = figure(width=1200, height=900,
                  title='Noisefree Functions Line Plot',
                  x_axis_label='x', y_axis_label='y')
    min_max_values = df['x'].agg(['min', 'max'])
    plot.x_range = Range1d(min_max_values.iloc[0],
                           min_max_values.iloc[1])
    for i, column in enumerate(train_dataset.df.columns):
        if i > 0:
            plot.line(train_dataset.df.iloc[:, 0], train_dataset.df[column],
                      line_color=Spectral11[i % len(Spectral11)],
                      legend_label='train_'+str(column))
    for i, column in enumerate(df.columns):
        if i > 0:
            plot.line(df.iloc[:, 0], df[column],
                      line_color=Spectral11[i % len(Spectral11)],
                      legend_label='ideal_'+str(column))
    plot.legend.location = 'top_left'
    show(plot)  # type: ignore


def plot_noisefree_funcs_w_tps(df_testpoints,
                               df_noisefree,
                               df_table3):
    '''
    Creates one plot for each noisefree function with the corresponding
    testpoint in the same color and one plot with all functions.
    Also shows all testpoints, that aren't associated to the function in black.
    The plots are displayed using the Bokeh library.

    Args:
        df_testpoints (DataFrame): A DataFrame containing test points
            with columns 'x' and 'y1', 'y2', etc.
        df_noisefree (DataFrame): A DataFrame containing noise-free data
            with columns 'x' and 'y1', 'y2', etc.
        df_table3 (DataFrame): A DataFrame containing the test points
            approached to the noisefree functions

    Returns:
        None

    Example Usage:
        plot_noisefree_funcs_w_tps(
            functions_testdatapoints_df, noisefree_df, table3_df)

    '''

    def new_plot_for_each_func():
        output_file(column+'_data_diagram.html')
        plot = figure(width=1200, height=900,
                      title=column+' with all fitting points',
                      x_axis_label='x', y_axis_label='y')
        min_max_values = df_noisefree['x'].agg(['min', 'max'])
        plot.x_range = Range1d(min_max_values.iloc[0],
                               min_max_values.iloc[1])
        return plot

    # temp_df = df_testpoints.drop('y', axis=1, inplace=False)
    for i, column in enumerate(df_noisefree.columns):
        if i > 0:
            p = new_plot_for_each_func()
            p.line(df_noisefree.iloc[:, 0], df_noisefree[column],
                   line_color=Bokeh5[i],
                   legend_label='ideal_'+str(column))
            p.scatter(test_dataset.df.iloc[:, 0], test_dataset.df.iloc[:, 1],
                      marker='circle', size=5, fill_color='black')
            p.scatter(df_testpoints.iloc[:, 0], df_testpoints.iloc[:, i],
                      marker='circle', size=10,
                      fill_color=Bokeh5[i])
            p.legend.location = 'top_left'
            show(p)  # type: ignore

    # create one plot with all funcs and points
    output_file('Table3_data_diagram.html')
    p2 = figure(width=1200, height=900,
                title='all Testpoints with the ideal functions',
                x_axis_label='x', y_axis_label='y')
    min_max_values = df_noisefree['x'].agg(['min', 'max'])
    p2.x_range = Range1d(min_max_values.iloc[0],
                         min_max_values.iloc[1])
    p2.scatter(test_dataset.df.iloc[:, 0], test_dataset.df.iloc[:, 1],
               marker='circle', size=5, fill_color='black')
    for i, column in enumerate(df_noisefree.columns):
        if i > 0:
            p2.line(df_noisefree.iloc[:, 0], df_noisefree[column],
                    line_color=Bokeh5[i],
                    legend_label='ideal_'+str(column))
            index_testdp = df_table3.loc[
                df_table3['Nr. of the ideal function'] ==
                str(column)].index
            for j in index_testdp:
                p2.scatter(df_table3.iloc[j, 0], df_table3.iloc[j, 1],
                           marker='circle', size=10,
                           fill_color=Bokeh5[i])
    show(p2)  # type: ignore


if __name__ == '__main__':
    try:
        os.remove(my_db_path)
    except FileNotFoundError:
        pass
    else:
        if not os.path.exists(my_db_path.rsplit('/', 1)[0]):
            os.makedirs(my_db_path.rsplit('/', 1)[0].replace('/', '\\'))

    train_dataset = DatasetWithSQLTable(table_name='train_data',
                                        file_name='train.csv',
                                        engine=engine)
    ideal_dataset = DatasetWithSQLTable(table_name='ideal_data',
                                        file_name='ideal.csv',
                                        engine=engine)
    # plot_ideal_funcs()

    noisefree_funcs_index = least_square_regression(
        ideal_dataset.df, train_dataset.df)
    # noisefree_funcs_index has a (4, 2) shape

    # create df with the 4 new 'ideal' functions instead of
    # the 4 noisy functions from the 'train' dataset
    noisefree_df = pd.DataFrame(
        columns=['x'], index=range(400))
    noisefree_df['x'] = ideal_dataset.df.iloc[:, 0]
    for i in range(4):
        row_nr = noisefree_funcs_index[i, 1]
        noisefree_df['y'+str(row_nr)] = ideal_dataset.df.iloc[:, row_nr]
    # plot_noisefree_funcs(noisefree_df)

    test_dataset = DatasetCSV(file_name='test.csv')
    test_dataset.df = test_dataset.df.\
        sort_values(by='x').reset_index(drop=True)
    functions_testdp_df, table3_df = find_best_fit_for_test_data(
        noisefree_df, test_dataset.df)
    table3_df.to_sql('Test_Datapoints_Fitted', con=engine, index=False)
    logging.info('finished exporting all relevant data to SQL DB')
    plot_noisefree_funcs_w_tps(
        functions_testdp_df, noisefree_df, table3_df)
