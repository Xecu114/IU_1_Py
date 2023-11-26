from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral11
import numpy as np
import pandas as pd

# Die Funktion plot_data nimmt ein Pandas Dataframe df als Eingabe und gibt
# ein Liniendiagramm zurück, das alle Spalten des Dataframes darstellt. Jede
# Spalte wird in einer anderen Farbe dargestellt und mit einem eigenen Label
# versehen.


def plot_sql_data(*args):
    output_file("sql_data_diagram.html")
    for df in args:
        plot = figure(width=1200, height=900, title="Line Plot",
                      x_axis_label="x", y_axis_label="y")
        numlines = len(df.columns)
        mypalette = Spectral11[0:numlines]
        for column, color in zip(df.columns, mypalette):
            plot.line(df.index, df[column],
                      line_color=color, legend_label=str(column))
        plot.legend.location = "top_left"
        show(plot)  # type: ignore
