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
plottenQ = False

files_path = 'Python_Course_IU'
my_db_path = os.path.join(files_path, '/db/my_db.db')
td_filename = 'test.csv'


class DatasetCSV():
    '''
    A class to represent a dataset imported from a CSV file.

    Instance Attributes (Parameters):
    - file_name (str): the name of the CSV file.

    Class Attributes:
    - df (pandas.DataFrame): DataFrame containing the data from the CSV file.

    Example Usage:
        test_dataset = DatasetCSV(file_name='test.csv')
        print(test_dataset.df)
    '''

    def __init__(self, file_name: str):
        '''
        Constructor to assign the provided parameters to the attributes
        '''
        if not isinstance(file_name, str):
            raise TypeError("file_name must be a string")
        # _ before the attribute name shows that it's a "private" attribute
        # to indicate it should not be changed later
        self._file_name = file_name
        # read the csv file into a pandas.DataFrame at initialization
        self.get_dataframe_from_csv()

    # method created instead of directly putting it into _init_ so it can
    # be called again after initialization
    def get_dataframe_from_csv(self):
        '''
        Reads the CSV file and assigns the data to the df attribute.

        Args:
            None

        Returns:
            None
        '''
        file_path = os.path.join(files_path, self._file_name)
        if os.path.isfile(file_path):
            self.df = pd.read_csv(file_path)
            logging.debug(file_path + ' - file read successfully')
        else:
            raise FileNotFoundError("File not found:", file_path)

    def plot_data(self):
        '''
        Generates line charts representing all columns of the DataFrame using
        the Bokeh library. Each column is displayed in a different color and
        given its own label. If a dataset is to big for one plot, it is
        divided into seperate DataFrames that each contain 10 functions. That
        makes it easier to identify each function lines.

        Args:
            None

        Returns:
            None

        Example Usage:
            plot_all_ideal_funcs()
        '''
        output_file('ideal_data_diagram.html')

        # first column needs to be the x values
        if self.df.columns.tolist()[0] != 'x':
            raise ValueError(
                "The first column should be 'x', but currently is: "
                f"{self.df.columns.tolist()[0]}")

        def plot_lines(plot_df):
            plot = figure(width=1200, height=900,
                          title='ideal.csv Line Plot' + str(plot_df.index),
                          x_axis_label='x', y_axis_label='y')
            min_max_values = self.df['x'].agg(['min', 'max'])
            plot.x_range = Range1d(
                min_max_values.iloc[0], min_max_values.iloc[1])
            for i, column in enumerate(plot_df.columns):
                plot.line(self.df.iloc[:, 0], plot_df[column],
                          line_color=Spectral11[i % len(Spectral11)],
                          legend_label=str(column))
            plot.legend.location = 'top_left'
            show(plot)  # type: ignore

        # get number of columns with functions to plot
        num_columns = (len(self.df.columns)-1)
        if num_columns > 10:
            # split the ideal dataframe into seperate dataframes with 10
            # functions. e.g. df shape splits from [400 r x 51 c] to
            # 5x [400 r x 10 c]
            temp_dfs = [self.df.iloc[:, i:i+10]
                        for i in range(1, num_columns, 10)]
            for temp_df in temp_dfs:
                plot_lines(temp_df)
        else:
            plot_lines(self.df.drop(columns='x'))


class DatasetWithSQLTable(DatasetCSV):
    '''
    A class to represent a dataset imported from a CSV file and additionally
    with an export to a SQLite database table.

    Instance Attributes (Args):
    - file_name (str): the name of the CSV file.
    - table_name (str): the name of the SQLite db table.
    - engine (sqlalchemy.create_engine): An instance of the SQLAlchemy
        create_engine class representing the connection to the SQLite database.

    Class Attributes:
    - df (pd.DataFrame): DataFrame containing the data from the CSV file.

    Example Usage:
        train_dataset = DatasetWithSQLTable(table_name='train_data',
                                        file_name='train.csv',
                                        engine=engine)
        print(train_dataset.df)
    '''

    # constructor to assign the provided parameters to the attributes
    def __init__(self, table_name: str,
                 file_name: str, engine):
        '''
        Constructor to assign the provided parameters to the attributes
        '''
        if not isinstance(file_name, str):
            raise TypeError("file_name must be a string")
        if not isinstance(table_name, str):
            raise TypeError("table_name must be a string")
        # call constructor of parent class ("DatasetCSV")
        super().__init__(file_name)
        self._table_name = table_name
        self._engine = engine
        # write the data at initialization
        self.write_data_to_sql()

    # method created instead of directly putting it into _init_ so it can
    # be called again after initialization
    def write_data_to_sql(self):
        '''
        Write the data from a DataFrame to a SQLite database table.

        Args:
            None

        Returns:
            None
        '''
        self.df.to_sql(self._table_name, con=self._engine,
                       if_exists='replace', index=False)
        logging.debug(self._table_name + ' - written Data to SQL')


def approx_funcs_with_lsr(df_i: pd.DataFrame, df_n: pd.DataFrame):
    '''
    Apply least squares regression to approximate functions from a
    noisy dataset to ideal functions.

    Args:
        df_i (pandas.DataFrame): The first dataset containing the
            x and y values of the ideal functions.
        df_n (pandas.DataFrame): The second dataset containing the
            x and y values of the functions which are to be approximated
            to the ideal functions .

    Returns:
        DataFrame: Dataframe with same shape as the second dataset but the
            data is replaced by the new "more ideal" functions


    Example Usage:
        approxed_funcs_index, approxed_funcs_df = approx_funcs_with_lsr(
            ideal_dataset.df, train_dataset.df)
    '''

    # check if args are correct type
    if not isinstance(df_i, pd.DataFrame):
        raise TypeError("df_i must be a pandas DataFrame")
    if not isinstance(df_n, pd.DataFrame):
        raise TypeError("df_n must be a pandas DataFrame")
    # Check if each element is numeric
    numeric_check = df_i.map(np.isreal)
    numeric_check2 = df_n.map(np.isreal)
    # If there are non-real values, raise TypeError
    if not numeric_check.all().all():
        raise TypeError("df_i includes wrong value types")
    if not numeric_check2.all().all():
        raise TypeError("df_n includes wrong value types")
    # check for missing / wrong values
    if df_i.isnull().values.any():
        raise TypeError("df_i has missing values")
    if df_n.isnull().values.any():
        raise TypeError("df_n has missing values")

    # Define a function, that calculates the residuals for a given set of
    # polynomial coefficients p,  observed y-values (y), and x-values (x).
    def residuals(p, y, x):
        return y - np.polyval(p, x)

    # get length of the columns and store value into a variable
    df_i_col_len = len(df_i.columns)-1
    df_n_col_len = len(df_n.columns)-1

    # Transforming the columns into Numpy-Arrays with the help of a loop
    ideal_functions = [df_i[column].tolist()
                       for column in df_i.columns if column != 'x']
    noisy_functions = [df_n[column].tolist()
                       for column in df_n.columns if column != 'x']

    # Initalize array to save the results
    result = np.zeros((df_i_col_len, 2), dtype=float)
    approxed_funcs_index = np.zeros((len(df_n.columns)-1, 2), dtype=int)

    # Loop over the functions and find the best match
    for j, np_array in enumerate(noisy_functions):
        # Iterate over the ideal functions
        for i in range(df_i_col_len):
            # get parameters of the ideal function with least squares method
            p = least_squares(residuals, np.ones(3), method='trf',
                              args=(ideal_functions[i],
                                    np.arange(len(df_i))),
                              verbose=0).x
            # Note: method=... trf and lm throw same result

            # Save the results into an array
            result[i, 0] = i+1
            # calculate difference between the predicted values of the polynom
            # and the actual values of the noisy function
            result[i, 1] = np.linalg.norm(
                (np.polyval(p, np.arange(len(df_i))) - np_array) ** 2)

        # Sort the result array after the sum of squared deviations
        sorted_indices = np.argsort(result[:, 1])
        result = result[sorted_indices]

        # Save the best result into the new array
        approxed_funcs_index[j, 0] = j+1
        approxed_funcs_index[j, 1] = int(result[0, 0])

    # create df with the new "ideal" functions that replace
    # the old noisy functions
    # Note: noisefree_funcs_index has a (df_i_col_len, 2) shape
    approxed_funcs_df = pd.DataFrame(
        columns=['x'], index=range(len(df_i)))
    # add the x values
    approxed_funcs_df['x'] = df_i.iloc[:, 0]
    # add column after column the "ideal" functions
    for i in range(df_n_col_len):
        row_nr = approxed_funcs_index[i, 1]
        approxed_funcs_df['y'+str(row_nr)] = df_i.iloc[:, row_nr]

    logging.debug('least_square_regression - done')
    return approxed_funcs_df


def approx_test_datapoints_to_funcs(df_funcs, df_test):
    '''
    Checks to which of the four noise-free / ideal functions each test
    datapoint can be approximated.

    Args:
        df_funcs (pandas DataFrame): The DataFrame needs to contain the
            functions that the test datapoints should be approximated to.
            Should include their x and y values (y values should be labeled).
            example columns: ['x', 'y1', 'y2', ...]
        df_test (pandas DataFrame): The DataFrame needs to contain the test
            points with their x and y values.
            ! Should have 2 columns: ['x', 'y']

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

    # check if args are correct type
    if not isinstance(df_funcs, pd.DataFrame):
        raise TypeError("df_funcs must be a pandas DataFrame")
    if not isinstance(df_test, pd.DataFrame):
        raise TypeError("df_test must be a pandas DataFrame")
    # Check if each element is numeric
    numeric_check = df_funcs.map(np.isreal)
    numeric_check2 = df_test.map(np.isreal)
    # If there are non-real values, raise TypeError
    if not numeric_check.all().all():
        raise TypeError("df_funcs includes wrong value types")
    if not numeric_check2.all().all():
        raise TypeError("df_test includes wrong value types")
    # check for missing / wrong values
    if df_funcs.isnull().values.any():
        raise TypeError("df_funcs has missing values")
    if df_test.isnull().values.any():
        raise TypeError("df_test has missing values")
    if df_funcs.columns.tolist()[0] != 'x':
        raise ValueError(
            f"first column of df_funcs should be 'x', but is "
            f"{df_funcs.columns.tolist()[0]}")
    if df_test.columns.tolist() != ['x', 'y']:
        raise ValueError(
            f"df_test should have two columns: ['x', 'y'], but has "
            f"{df_test.columns.values}")

    def is_point_nearby(functions_df, testpoint):
        '''
        Check if a given test point is nearby a specific function
        in the dataset (pandas.DataFrame).

        Args:
        functions_df: A pandas.DataFrame containing the function
            with their x and y values.
        testpoint: The y-value of the test point.

        Returns:
        Boolean: Value indicating whether the test point is nearby the
            function.
        '''

        # get the index of the x position in the DataFrame containing
        # the functions
        x_index = functions_df.loc[functions_df['x']
                                   == df_test.iloc[i, 0]].index[0]
        if i < 10:
            dist = functions_df.iloc[x_index:(x_index+2), j].apply(
                lambda x: abs((x - testpoint))).min()
        else:
            dist = functions_df.iloc[(x_index-2):(x_index+2), j].apply(
                lambda x: abs((x - testpoint))).min()
        temp_y = functions_df.iloc[x_index, j]
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

    for j in range(len(df_funcs.columns)):
        if j > 0:
            c = str(df_funcs.columns[j])+'_testpoints'
            for i in range(len(df_test)):
                delta_y, is_nearby = is_point_nearby(
                    df_funcs, df_test.loc[i, 'y'])
                if is_nearby:
                    results_df.loc[i, c] = df_test.loc[i, 'y']

                    # store every matching function
                    if fitted_testdp_df.iloc[i, 4] == '':
                        fitted_testdp_df.iloc[i, 2] = \
                            round(delta_y, 3)
                        fitted_testdp_df.iloc[i, 3] = \
                            str(df_funcs.columns[j])
                        fitted_testdp_df.iloc[i, 4] = \
                            str(df_funcs.columns[j])
                    else:
                        # choose best match by comparing the distances
                        if delta_y < fitted_testdp_df.iloc[i, 2]:
                            fitted_testdp_df.iloc[i, 2] = \
                                round(delta_y, 3)
                            fitted_testdp_df.iloc[i, 3] = \
                                str(df_funcs.columns[j])

                        fitted_testdp_df.iloc[i, 4] = \
                            str(fitted_testdp_df.iloc[i, 4]) + ', ' + \
                            str(df_funcs.columns[j])

    # fitted_testdp_df['Delta Y (Deviation)'].replace(
    #     '', '-', inplace=True)
    fitted_testdp_df['Nr. of all fitting functions'].replace(
        '', '-', inplace=True)
    logging.debug('approx_test_datapoints_to_funcs - done')
    return results_df, fitted_testdp_df


def plot_data(df):
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
        plot_all_ideal_funcs()
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


def plot_two_dataframes(df1, df2):
    '''
    Generates a line plot of two sets of data using the Bokeh library.
    The DataFrames need to have the first column named 'x' with the x values
    and the other columns need to contain the y values
    ...

    Args:
        df1 (DataFrame): ...
        df2 (DataFrame): ...

    Returns:
        None

    Example Usage:
        plot_noisefree_functions(df1, df2)
    '''
    if df1.columns.tolist()[0] != 'x' or df2.columns.tolist()[0] != 'x':
        raise ValueError("The first column of both df1 and df2 should be 'x'.")

    output_file('noisefree_data_diagram.html')
    plot = figure(width=1200, height=900,
                  title='Noisefree Functions Line Plot',
                  x_axis_label='x', y_axis_label='y')
    # get min and max 'x' value
    min_max_values = df1['x'].agg(['min', 'max'])
    # plot x_axis
    plot.x_range = Range1d(min_max_values.iloc[0],
                           min_max_values.iloc[1])
    # plot the lines
    for i, column in enumerate(df1.columns):
        if i > 0:
            plot.line(df1.iloc[:, 0], df1[column],
                      line_color=Spectral11[i % len(Spectral11)],
                      legend_label='df1_'+str(column))
    for i, column in enumerate(df2.columns):
        if i > 0:
            plot.line(df2.iloc[:, 0], df2[column],
                      line_color=Spectral11[i % len(Spectral11)],
                      legend_label='df2_'+str(column))
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
    # try to remove the db when it already exists
    # -> we'd like a fresh db without any data in it
    try:
        os.remove(my_db_path)
    except FileNotFoundError:
        # create the path if path isn't already created
        if not os.path.exists(my_db_path.rsplit('/', 1)[0]):
            os.makedirs(my_db_path.rsplit('/', 1)[0].replace('/', '\\'))
    # create SQLite Database
    engine = create_engine(f'sqlite:///{my_db_path}', echo=False)

    # create new class instance that represents the datasets
    # from the "train.csv" and "ideal.csv" files
    # and because we want the data to be saved in the SQL db
    # we choose the "DatasetWithSQLTable" class
    train_dataset = DatasetWithSQLTable(table_name='train_data',
                                        file_name='train.csv',
                                        engine=engine)
    ideal_dataset = DatasetWithSQLTable(table_name='ideal_data',
                                        file_name='ideal.csv',
                                        engine=engine)
    train_dataset.plot_data() if plottenQ is True else None
    ideal_dataset.plot_data() if plottenQ is True else None
    # apply the least squares regression method to approximate
    # the noisy functions to the ideal functions and get a new
    # dataframe with the ideal functions that fitted the best
    # shape of noisefree_df is: (5, 400)
    noisefree_df = approx_funcs_with_lsr(
        ideal_dataset.df, train_dataset.df)

    # plot the result by laying the noisy functions over the approxed
    # ideal functions
    plot_two_dataframes(
        noisefree_df, train_dataset.df) if plottenQ is True else None

    # create new instance of the "DatasetCSV" class that reresents our
    # test data from the "test.csv" file
    test_dataset = DatasetCSV(file_name='test.csv')
    # sort the data by x value
    test_dataset.df = test_dataset.df.\
        sort_values(by='x').reset_index(drop=True)
    test_dataset.plot_data() if plottenQ is True else None
    # find the best fits for each test data points by checking which of the
    # four ideal functions each test data point can be approximated to
    functions_testdp_df, table3_df = approx_test_datapoints_to_funcs(
        noisefree_df, test_dataset.df)
    # export the resulting data to the SQL db
    table3_df.to_sql('Test_Datapoints_Fitted', con=engine, index=False)
    logging.info('finished exporting all relevant data to SQL DB')
    # plots the ideal functions with the fitted test data points.
    plot_noisefree_funcs_w_tps(
        functions_testdp_df, noisefree_df, table3_df) \
        if plottenQ is True else None
