import os
import sqlalchemy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree

BASE_DIR = 'C:\\Users\\SGG\\Google Drive\\Atuariais2\\python\\PortoSeguroSafeDriverPrediction'
BASE_DIR = os.path.dirname ( os.path.dirname( os.path.dirname(__file__) ) )
DATA_DIR = os.path.join( BASE_DIR, 'data' )
SRC_DIR = os.path.join( BASE_DIR, 'src' )
PY_DIR = os.path.join( SRC_DIR, 'python' )
SQL_DIR = os.path.join( SRC_DIR, 'sql' )

train = pd.read_csv( os.path.join( DATA_DIR, 'train.csv' ) )
test = pd.read_csv( os.path.join( DATA_DIR, 'test.csv' ) )

#valores -1 significam misses

print(train.head())
train.columns
train.describe()

cat_variavel = [i for i in train.columns if 'cat' in i]
bin_variavel = [i for i in train.columns if 'bin' in i]

cat_list = train[ cat_variavel ].columns.to_list()

cat_database = train[ cat_variavel ].values

#one hot encoder aprender a fazer
#cat_onehotencoder = OneHotEncoder(
#    categorical_features = train[ cat_list ],
#    #sparse=False,
#    handle_unknown='ignore'
#) 
#cat_data = cat_onehotencoder.fit_transform(cat_database).toarray()
#
#enc_df = pd.DataFrame(cat_onehotencoder.fit_transform(cat_database).toarray())

#Tree

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit()