import os
import sqlalchemy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)

#localizar pastas no sistema
BASE_DIR = 'C:\\Users\\SGG\\Google Drive\\Atuariais2\\python\\PortoSeguroSafeDriverPrediction'
BASE_DIR = os.path.dirname ( os.path.dirname( os.path.dirname(__file__) ) )
DATA_DIR = os.path.join( BASE_DIR, 'data' )
SRC_DIR = os.path.join( BASE_DIR, 'src' )
PY_DIR = os.path.join( SRC_DIR, 'python' )
SQL_DIR = os.path.join( SRC_DIR, 'sql' )

#carregando os dados
train = pd.read_csv( os.path.join( DATA_DIR, 'train.csv' ), nrows= 1000 )
#test = pd.read_csv( os.path.join( DATA_DIR, 'test.csv' ) )

#Descritiva da variável target
sns.catplot(x="target", kind="count", palette="ch:.25", data=train)
plt.show()