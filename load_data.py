import pandas as pd
from sqlalchemy import create_engine

def load_data_from_csv(filepath):
    """
    This function takes a CSV filepath and reads the data from that CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def load_data_from_db(connection_str, tbl_name):
    """
    This function takes in connection string and table name and retrieves the data from that table from a database.
    """
    engine = create_engine(connection_str)
    with engine.connect() as connection:
        df = pd.read_sql_table(tbl_name, connection)
    engine.dispose()
    return df