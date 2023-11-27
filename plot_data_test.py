import unittest
import os
from plot_data import plot_sql_data
import main as m


dir_path = "Python_Course_IU"
if os.path.exists(dir_path+"\\my_db.db"):
    os.remove(dir_path+"\\my_db.db")

m.import_datasets_to_sqllite_table(dir_path)
train_df, ideal_df = m.get_datasets_from_sql_database(dir_path)
plot_sql_data(train_df, ideal_df)
