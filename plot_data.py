from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral11, Bokeh5
from bokeh.models import Range1d
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)


def plot_ideal_functions(ideal_df):
    '''
    Generates line charts representing all columns of the DataFrame using the
    Bokeh library. Each column is displayed in a different color and given
    its own label. Because the 'ideal' dataset is so big, it is divided
    into 5 seperate DataFrames that each contain 10 functions

    Args:
        ideal_df (_type_): DataFrame containing the whole set of
            'ideal' functions as columns with following shape:
            [400 r x 51 c]

    Returns:
        None

    Example Usage:
        plot_ideal_functions(ideal_df)
    '''
    output_file('ideal_data_diagram.html')

    # split the ideal DataFrame into 5 DataFrames with 10 functions each
    # dfs = [ideal_df.iloc[:, i:i+10] for i in range(1, 50, 10)]
    # split the ideal dataframe into 5 dataframes with 10 functions each
    dfs = [ideal_df.iloc[:, i:i+10] for i in range(1, 50, 10)]

    for df in dfs:
        plot = figure(width=1200, height=900,
                      title='ideal.csv Line Plot' + str(df.index),
                      x_axis_label='x', y_axis_label='y')
        min_max_values = ideal_df['x'].agg(['min', 'max'])
        plot.x_range = Range1d(min_max_values.iloc[0], min_max_values.iloc[1])
        for i, column in enumerate(df.columns):
            plot.line(ideal_df.iloc[:, 0], df[column],
                      line_color=Spectral11[i % len(Spectral11)],
                      legend_label=str(column))
        plot.legend.location = 'top_left'
        show(plot)  # type: ignore


def plot_noisefree_funcs(df, train_df):
    '''
    Generates a line plot of two sets of data using the Bokeh library.
    The first set of data is the new ideal functions that match the noisy
    train functions, which are the second set of data.

    Args:
        df (DataFrame): The DataFrame containing the ideal or 'noisefree'
            data. It should contain at the the column 'x' and one column
            for each of the four functions.
        train_df (DataFrame): A subset of the main DataFrame containing the
            columns to be plotted. It should contain at the the column 'x'
            and one column for each of the four functions.

    Returns:
        None

    Example Usage:
        plot_noisefree_functions(noisefree_df, train_df)
    '''
    output_file('noisefree_data_diagram.html')
    plot = figure(width=1200, height=900,
                  title='Noisefree Functions Line Plot',
                  x_axis_label='x', y_axis_label='y')
    min_max_values = df['x'].agg(['min', 'max'])
    plot.x_range = Range1d(min_max_values.iloc[0],
                           min_max_values.iloc[1])
    for i, column in enumerate(train_df.columns):
        if i > 0:
            plot.line(train_df.iloc[:, 0], train_df[column],
                      line_color=Spectral11[i % len(Spectral11)],
                      legend_label='train_'+str(column))
    for i, column in enumerate(df.columns):
        if i > 0:
            plot.line(df.iloc[:, 0], df[column],
                      line_color=Spectral11[i % len(Spectral11)],
                      legend_label='ideal_'+str(column))
    plot.legend.location = 'top_left'
    show(plot)  # type: ignore


def plot_noisefree_funcs_w_tps(df_testdata,
                               df_testpoints,
                               df_noisefree,
                               df_table3):
    '''
    Creates one plot for each noisefree function with the corresponding
    testpoint in the same color and one plot with all functions.
    Also shows all testpoints, that aren't associated to the function in black.
    The plots are displayed using the Bokeh library.

    Args:
        df_testdata (DataFrame): A DataFrame containing test data with
            columns 'x' and 'y'.
        df_testpoints (DataFrame): A DataFrame containing test points
            with columns 'x' and 'y1', 'y2', etc.
        df_noisefree (DataFrame): A DataFrame containing noise-free data
            with columns 'x' and 'y1', 'y2', etc.

    Returns:
        None

    Example Usage:
        ...
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
            p.scatter(df_testdata.iloc[:, 0], df_testdata.iloc[:, 1],
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
    p2.scatter(df_testdata.iloc[:, 0], df_testdata.iloc[:, 1],
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
