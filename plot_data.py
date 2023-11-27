from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral11
from bokeh.models import Range1d
import numpy as np
import pandas as pd

# Die Funktion plot_data nimmt ein Pandas Dataframe df als Eingabe und gibt
# ein Liniendiagramm zurück, das alle Spalten des Dataframes darstellt. Jede
# Spalte wird in einer anderen Farbe dargestellt und mit einem eigenen Label
# versehen.


def plot_sql_data(train_df, ideal_df):
    """_summary_
    The plot_sql_data function takes the Pandas dataframes as inputs and
    returns line charts representing all columns of the dataframes. Each
    column is displayed in a different color and given its own label.
    Because the "ideal" dataset is so big, it is divided into 5
    dataframes that each contain 10 functions

    Args:
        train_df (_type_): df with [400 r x 5 c]
        ideal_df (_type_): df with [400 r x 51 c]
    """
    output_file("sql_data_diagram.html")
    plot = figure(width=1200, height=900, title="train.csv Line Plot",
                  x_axis_label="x", y_axis_label="y")
    min_max_values = train_df['x'].agg(['min', 'max'])
    plot.x_range = Range1d(min_max_values.iloc[0], min_max_values.iloc[1])
    for i, column in enumerate(train_df.columns):
        if i > 0:
            plot.line(train_df.iloc[:, 0], train_df[column],
                      line_color=Spectral11[i % len(Spectral11)],
                      legend_label=str(column))
    plot.legend.location = "top_left"
    show(plot)  # type: ignore

    # split the dataframe into 5 dataframes with 10 functions each
    dfs = [ideal_df.iloc[:, i:i+10] for i in range(1, 50, 10)]
    print(dfs)

    for df in dfs:
        plot = figure(width=1200, height=900,
                      title="ideal.csv Line Plot" + str(df.index),
                      x_axis_label="x", y_axis_label="y")
        min_max_values = ideal_df['x'].agg(['min', 'max'])
        plot.x_range = Range1d(min_max_values.iloc[0], min_max_values.iloc[1])
        for i, column in enumerate(df.columns):
            plot.line(train_df.iloc[:, 0], df[column],
                      line_color=Spectral11[i % len(Spectral11)],
                      legend_label=str(column))
        plot.legend.location = "top_left"
        show(plot)  # type: ignore

# def plot_sql_data(*args):
#     output_file("sql_data_diagram.html")
#     for df in args:
#         plot = figure(width=1200, height=900, title="Line Plot",
#                       x_axis_label="x", y_axis_label="y")
#         min_max_values = df['x'].agg(['min', 'max'])
#         plot.x_range = Range1d(min_max_values.iloc[0],
#                                   min_max_values.iloc[1])
#         for i, column in enumerate(df.columns):
#             if i > 0:
#                 plot.line(df.iloc[:, 0], df[column],
#                           line_color=Spectral11[i % len(Spectral11)],
#                           legend_label=str(column))
#         plot.legend.location = "top_left"
#         show(plot)  # type: ignore
