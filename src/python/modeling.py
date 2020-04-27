import os
import sqlalchemy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = 'C:\\Users\\SGG\\Google Drive\\Atuariais2\\python\\PortoSeguroSafeDriverPrediction'
BASE_DIR = os.path.dirname ( os.path.dirname( os.path.dirname(__file__) ) )
DATA_DIR = os.path.join( BASE_DIR, 'data' )
SRC_DIR = os.path.join( BASE_DIR, 'src' )
PY_DIR = os.path.join( SRC_DIR, 'python' )
SQL_DIR = os.path.join( SRC_DIR, 'sql' )

train = pd.read_csv( os.path.join( DATA_DIR, 'train.csv' ) )

#valores -1 significam misses

print(train.head())
train.columns
train.describe()

cat_variavel = [i for i in train.columns if 'cat' in i]
bin_variavel = [i for i in train.columns if 'bin' in i]

#one hot encoder
#onehotencoder = OneHotEncoder(categorical_features = [0]) 
#data = onehotencoder.fit_transform(data).toarray() 