#!/usr/bin/env python
# coding: utf-8

# 1° Importando CSV

# In[1]:


import pandas as pd
# 1° Carregando o dataframe
df = pd.read_csv('waterQuality1.csv')


# 2° Analisando Dados
# 

# In[2]:


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
#display(df)


# 3° Tratamento de Dados
# 

# In[3]:


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

# In[4]:


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

# In[23]:


# Nenhum valor nulo identificado
df.isnull().sum()

# Mostra os dados estatisticos do dataframe
df.describe()

import seaborn as sns
import matplotlib.pyplot as plt

# Visualizar a contagem de amostras seguras (1) e inseguras (0)
sns.countplot(x='is_safe', data=df_balanced)
plt.title('Distribuição de Amostras Seguras vs Inseguras')
plt.show()

# Boxplots das variáveis numéricas
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_balanced.drop(columns=['is_safe'])*100/df.shape[0], orient='h')
plt.title("Compostos presentes na água")
plt.show()

