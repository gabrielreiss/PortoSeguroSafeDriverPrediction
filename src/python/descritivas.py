import os
import sqlalchemy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)
import numpy as np

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
g = sns.catplot(x="target", kind="count", palette="ch:.25", data=train,
legend = True)
train.pivot(value='target')
#g.axes.text(str(train.target.value_counts().to_list()[0]),
#            str(train.target.value_counts().to_list()[1]))
plt.title('Descritiva da variável target')
plt.show()

#matriz de correlação
variaveis = list(set(train) - set(train[['id', 'ps_ind_13_bin', 'ps_ind_10_bin']]))

correlacao = train[variaveis].corr()
mask = np.triu(np.ones_like(correlacao, dtype=np.bool))

f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(correlacao, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Matriz de Correlação')
plt.show()

#correlação do target
cor_target = correlacao['target'].sort_values( ascending = False)
cor_target = cor_target.drop(index='target')
cor_target = cor_target.dropna()

plt.barh(cor_target.index, cor_target.values)
plt.xlabel('Correlação')
plt.title('Correlação das Variáveis pelo target')
plt.show()