import os
import sqlalchemy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
import matplotlib.pyplot as plt
import pickle

#localizar pastas no sistema
BASE_DIR = 'C:\\Users\\SGG\\Google Drive\\Atuariais2\\python\\PortoSeguroSafeDriverPrediction'
BASE_DIR = os.path.dirname ( os.path.dirname( os.path.dirname(__file__) ) )
DATA_DIR = os.path.join( BASE_DIR, 'data' )
SRC_DIR = os.path.join( BASE_DIR, 'src' )
PY_DIR = os.path.join( SRC_DIR, 'python' )
SQL_DIR = os.path.join( SRC_DIR, 'sql' )
MODELS_DIR = os.path.join( SRC_DIR, 'models' )

test = pd.read_csv( os.path.join( DATA_DIR, 'test.csv' ), nrows = 1000 )
features = list(set(test) - set(test[['id']]))

with open(os.path.join( MODELS_DIR, 'model1.pkl' ), 'rb') as f:
    model1 = pickle.load( f )

clf = tree.DecisionTreeRegressor()
predicao = clf.predict( test[features] )

