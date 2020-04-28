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

#carregando os dados
train = pd.read_csv( os.path.join( DATA_DIR, 'train.csv' ), nrows= 1000 )
#test = pd.read_csv( os.path.join( DATA_DIR, 'test.csv' ) )

#organizando as colunas
features = list(set(train) - set(train[['target', 'id']]))

cat_variavel = [i for i in train.columns if 'cat' in i]
bin_variavel = [i for i in train.columns if 'bin' in i]

cat_list = train[ cat_variavel ].columns.to_list()

#cat_database = train[ cat_variavel ].values

#tratamento de misses
#valores -1 significam misses



#one hot encoder aprender a fazer
#cat_onehotencoder = OneHotEncoder(
#    categorical_features = train[ cat_list ],
#    #sparse=False,
#    handle_unknown='ignore'
#) 
#cat_data = cat_onehotencoder.fit_transform(cat_database).toarray()
#
#enc_df = pd.DataFrame(cat_onehotencoder.fit_transform(cat_database).toarray())

#Tree (faltou memoria)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train[features], train['target'])

#Construindo tabela de importancia para visualizar depois
importancia = pd.DataFrame( 
                            clf.feature_importances_,
                            index =list(train[features].columns),
                            columns= ['valores']
                        )

importancia = importancia.sort_values(by='valores',
                                      ascending = False)

#plot
#plt.barh(importancia.index, importancia.valores)
#plt.xlabel('Importância')
#plt.show()


with open(os.path.join( MODELS_DIR, 'model1.pkl' ), 'wb') as f:
    pickle.dump(clf, f)