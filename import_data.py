# import modules
import csv
from sqlalchemy import create_engine, Column, Float
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base


def import_datasets_to_sqllite_table(relative_path):

    # Create an engine that connects to a SQLite database
    engine = create_engine("sqlite:///"+relative_path+"\\my_db.db", echo=True)

    # Create a base class for declarative class definitions
    Base = declarative_base()

    # Define a class for the table for train.csv that inherits from base class
    class Table_train_csv(Base):
        __tablename__ = 'train_data'

        x = Column(Float, primary_key=True)
        for i in range(1, 5):
            locals()[f'y{i}'] = Column(Float)

        def __repr__(self):
            return f"train_data(x={self.x}, y1={self.y1}, y2={self.y2},"\
                f" y3={self.y3}, y4={self.y4})"

    # Define a class for the table for ideal.csv that inherits from base class
    class Table_ideal_csv(Base):
        __tablename__ = 'ideal_data'

        x = Column(Float, primary_key=True)
        for i in range(1, 51):
            locals()[f'y{i}'] = Column(Float)

        def __repr__(self):
            return f"ideal_data(x={self.x}, y1={self.y1},..., y50={self.y50}"

    # Create the tables in the database
    Base.metadata.create_all(engine)

    # Create a session factory bound to the engine
    Session = sessionmaker(bind=engine)

    # Create a session object
    session = Session()

    # Read the CSV files and add each row as a new user to the database
    with open(relative_path+"\\train.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_data = Table_train_csv(x=row['x'])
            for i in range(1, 5):
                setattr(train_data, f'y{i}', row[f'y{i}'])
            session.add(train_data)

    with open(relative_path+"\\ideal.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ideal_data = Table_ideal_csv(x=row['x'])
            for i in range(1, 51):
                setattr(ideal_data, f'y{i}', row[f'y{i}'])
            session.add(ideal_data)

    # Commit the changes to the database
    session.commit()
    # Query the database for all rows -> return type: list
    return_train_list = session.query(Table_train_csv).all()
    # Query the database for all rows -> return type: list
    return_ideal_list = session.query(Table_ideal_csv).all()

    # Close session
    session.close()
    return return_train_list, return_ideal_list
