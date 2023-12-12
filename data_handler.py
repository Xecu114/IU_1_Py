# import modules
import csv
from sqlalchemy import create_engine, Column, Float
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)


def transfer_csv_to_sqllite_table(csv_path, db_path, train_dataset,
                                  ideal_dataset):
    '''_summary_
    1. creates and connects to a new SQLite3 database
    2. defines classes for the sql tables that inherit from the (sqlalchemy)
        base class
    3. loads train.csv and ideal.csv from the passed csv_path into the
        tables
    4. queries data back from tables into lists to check them

    Args:
        csv_path (String): pass (relative) path in which the CSV files
            are located
        db_path (String): pass (relative) path where to create the database

    Returns:
        tuple[List[Table_train_csv], List[Table_ideal_csv]]

    Example Usage:
        import_datasets_to_sqllite_table('myfiles', '/db')
        or:
        train_list, ideal_list = import_datasets_to_sqllite_table('myfiles',
                                                                        '/db')
    '''
    # Create an engine that connects to a SQLite database
    engine = create_engine(f'sqlite:///{db_path}', echo=False)

    # Create a base class for declarative class definitions
    # Base = declarative_base()

    # create the tables in sql
    train_dataset.create_table_class(engine)
    ideal_dataset.create_table_class(engine)

    with open(csv_path+'\\'+train_dataset.file_name, newline='') as csvfile:
        df = pd.read_csv(csvfile)
        df.to_sql(train_dataset.table_name, con=engine,
                  if_exists='replace', index=False)
    with open(csv_path+'\\'+ideal_dataset.file_name, newline='') as csvfile:
        df = pd.read_csv(csvfile)
        df.to_sql(ideal_dataset.table_name, con=engine,
                  if_exists='replace', index=False)

    # # Create a session factory bound to the engine
    # Session = sessionmaker(bind=engine)
    # # Create a session object
    # session = Session()

    # with open(csv_path+'\\ideal.csv', newline='') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         ideal_data = Table_ideal_csv(x=row['x'])
    #         for i in range(1, 51):
    #             setattr(ideal_data, f'y{i}', row[f'y{i}'])
    #         session.add(ideal_data)

    # # Commit the changes to the database
    # session.commit()
    # # Query the database for all rows -> return type: list
    # return_train_list = session.query(Table_train_csv).all()
    # # Query the database for all rows -> return type: list
    # return_ideal_list = session.query(Table_ideal_csv).all()
    # # Close session
    # session.close()

    # if len(return_train_list) == 400 and\
    #         len(return_ideal_list) == 400:
    #     logging.debug('transfer_csv_to_sqllite_table - done')
    # else:
    #     logging.error('Something went wrong ! Data in SQL DB is not correct')
    return


def get_dataframe_from_sql_table(db_path, db_table_name):
    '''
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
                                            db_name, 'train_data')
    '''

    # Create a connection to the SQLite database
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    connection = engine.connect()
    # Query the databases and store the results in a pandas dataframe
    df = pd.read_sql_query(f'SELECT * FROM {db_table_name}', connection)
    # Close the database connection
    connection.close()
    logging.debug('get_dataframe_from_sql_table - done')
    return df


def fitted_testdata_to_sql(db_path, df):
    # Create an engine that connects to a SQLite database
    engine = create_engine(f'sqlite:///{db_path}', echo=False)

    # write the DataFrame into a SQL table
    df.to_sql('Test_Datapoints_Fitted', con=engine, index=False)

    logging.debug('fitted_testdata_to_sql - done')
