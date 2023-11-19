# import modules
import csv
import os
from sqlalchemy import create_engine, Column, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


def import_train_csv_to_sqllite_table(relative_path):

    # Create an engine that connects to a SQLite database
    engine = create_engine("sqlite:///"+relative_path+"\\my_db.db", echo=True)

    # Create a base class for declarative class definitions
    Base = declarative_base()

    # Define a class that inherits from the base class

    class TrainTable(Base):
        __tablename__ = 'train_data'

        x = Column(Float, primary_key=True)
        y1 = Column(Float)
        y2 = Column(Float)
        y3 = Column(Float)
        y4 = Column(Float)

        def __repr__(self):
            return f"train_data_row(x={self.x}, y1={self.y1}, y2={self.y2},"\
                f" y3={self.y3}, y4={self.y4})"

    # Create the table in the database
    Base.metadata.create_all(engine)

    # Create a session factory bound to the engine
    Session = sessionmaker(bind=engine)

    # Create a session object
    session = Session()

    # Read the CSV file and add each row as a new user to the database
    with open(relative_path+"\\train.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_data = TrainTable(x=row['x'], y1=row['y1'],
                                    y2=row['y2'], y3=row['y3'], y4=row['y4'])
            session.add(train_data)

    # Commit the changes to the database
    session.commit()
    # Query the database for all rows -> return type: list
    return_list = session.query(TrainTable).all()
    # Close session
    session.close()
    return return_list
