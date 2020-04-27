import os
import sqlalchemy
import pandas as pd

BASE_DIR = 'C:\\Users\\SGG\\Google Drive\\Atuariais2\\python\\PortoSeguroSafeDriverPrediction'
BASE_DIR = os.path.dirname ( os.path.dirname( os.path.dirname(__file__) ) )
DATA_DIR = os.path.join( BASE_DIR, 'data' )
SRC_DIR = os.path.join( BASE_DIR, 'src' )
PY_DIR = os.path.join( SRC_DIR, 'python' )
SQL_DIR = os.path.join( SRC_DIR, 'sql' )

engine = sqlalchemy.create_engine( os.path.join( 'sqlite:///', DATA_DIR, 'train.db' ) )
conn = engine.connect()

train = pd.read_csv( os.path.join( DATA_DIR, 'train.csv' ) )

train.to_sql(   "train",
                conn,
                if_exists = 'replace'
                )