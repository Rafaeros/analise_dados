#!/usr/bin/env python
# coding: utf-8

# 1° Importando CSV

# 

# In[10]:


import pandas as pd
# 1° Carregando o dataframe
df = pd.read_csv('waterQuality1.csv')


# 2° Analisando Dados
# 

# In[11]:


# Analisando formato dos dados
print("df info")
df.info()

# Visualizando primeiras linhas do dataframe
print ("df head")
print(df.head())

# Visualizando ultimas linhas do dataframe
print("df tail")
print(df.tail())

print("df shape")
print(df.shape)

# Visualizando tipos dos dados
print("df dtypes")
print(df.dtypes)

# o display() Dentro do jupyter notebook já mostra o df.head() e o df.tail() e o df.shape
# display(df)


# 3° Tratamento de Dados
# 

# In[12]:


# Verificando valores unicos
print("Antes do tratamento")
print(df['ammonia'].unique())
print(df['is_safe'].unique())

# Convertendo valores object para float
df['ammonia'] = pd.to_numeric(df['ammonia'], errors='coerce')
df['is_safe'] = pd.to_numeric(df['is_safe'], errors='coerce')

# Removendo valores NaN
df = df.dropna(subset=['ammonia', 'is_safe'])

# Verificando se foi removido valores NaN
print("Depois do tratamento")
print(df['ammonia'].unique())
print(df['is_safe'].unique())

# Após o tratamento a coluna de "ammonia" e "is_safe" mudaram seu tipo de object para float64
print(df.dtypes)


# 4° Problema de Desbalanceamento
# 
# 

# In[13]:


# Verificando se há desbalanceamento
print(df['is_safe'].value_counts())

# Após a analise do dataset, percebemos que o dataset possui
# desbalanceamento onde o valor 0 tem 7084 registros e o valor
# 1 tem 912 registros

# Reamostragem para balancear as classes
c_0 = df[df['is_safe'] == 0].sample(n=912, random_state=1)
c_1 = df[df['is_safe'] == 1]

# Concatenar as amostras reequilibradas
df_balanced = pd.concat([c_0, c_1]).reset_index(drop=True)

# Verificar novamente o balanceamento, depois da reamostragem
print(df_balanced['is_safe'].value_counts())


# 5° Análise Exploratória
# 

# In[14]:


# Nenhum valor nulo identificado
df.isnull().sum()

# Mostra os dados estatisticos do dataframe
df.describe()

import seaborn as sns
import matplotlib.pyplot as plt

# Compostos presentes na água
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_balanced.drop(columns=['is_safe']), orient='h',)
plt.title("Compostos presentes na água")
plt.show()

# Visualizar a contagem de amostras seguras (1) e inseguras (0)
sns.countplot(x='is_safe', data=df_balanced)
plt.title('Distribuição de Amostras Seguras vs Inseguras')
plt.show()



# 6° Separe os dados em conjuntos de treinamento (70%) e teste (30%)
# 

# In[15]:


from sklearn.model_selection import train_test_split

# Separando variáveis independentes (X) e a variável dependente (y)
X = df_balanced.drop(columns=['is_safe'])
y = df_balanced['is_safe']

# Dividindo os dados em treinamento (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# 7° Aplique os classificadores Gaussian Naive Bayes, K Nearest Neighbours (n_neighbors=3 e metric='euclidean') e Decision Tree;

# In[16]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Treinar o modelo Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predizer no conjunto de teste
y_pred_gnb = gnb.predict(X_test)

# Relatório de classificação
print("Gaussian Naive Bayes")
print("Acertou %d de %d." %((y_test == y_pred_gnb).sum(), len(X_test)))
print(classification_report(y_test, y_pred_gnb))


# In[17]:


from sklearn.neighbors import KNeighborsClassifier

# Treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)

# Predizer no conjunto de teste
y_pred_knn = knn.predict(X_test)

# Relatório de classificação
print("K-Nearest Neighbors")
print("Acertou %d de %d" % ((y_test == y_pred_knn).sum(), len(X_test)))
print(classification_report(y_test, y_pred_knn))


# In[18]:


from sklearn.tree import DecisionTreeClassifier

# Treinar o modelo Decision Tree
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, y_train)

# Predizer no conjunto de teste
y_pred_dt = dt.predict(X_test)

# Relatório de classificação
print("Decision Tree")
print("Acertou %d de %d" % ((y_test == y_pred_dt).sum(), len(X_test)))
print(classification_report(y_test, y_pred_dt))


# 8° Aplique o classification report para analisar a performance dos modelos e identifique o estimador com os melhores resultados.
# 
# O Modelo que apresentou maior performance e acuracia nos acertos foi o Decision Tree com
# Precision: 0.90 Recall: 0.89 e f1-score: 0.90
