## Trabalho de Ciência da dados Fórum 2.

Você foi contratado por uma companhia de distribuição de água para criar um modelo que auxilie na análise de qualidade de água. Os dados (disponível em: https://www.kaggle.com/mssmartypants/water-quality. Acesso em: 22 ago. 2022) estão em um arquivo CSV que descreve a quantidade mensurada de 20 compostos presentes em amostras de água, como alumínio, amônia, cobre e outros. O link do dataset fornece uma referência sobre a quantidade aceitável de cada composto e contém uma coluna adicional (is_safe) que descreve se a amostra é segura (valor 1) ou não é segura (valor 0). Assim sendo, siga o roteiro a seguir para realizar a tarefa:

Carregue o arquivo CSV como um dataframe Pandas;
Faça uma análise prévia sobre o formato dos dados (os recursos head, tail e dtypes podem ser úteis);
As colunas “ammonia” e “is_safe” possuem alguns dados com erros de leitura que devem ser removidos do conjunto. Identifique os registros com problema, remova-os do dataframe e resolva possíveis problemas de tipagem das colunas (a função pandas.to_numeric pode ser útil);
Verifique e resolva um possível problema de desbalanceamento dos dados (checar o final deste texto para instruções);
Faça uma análise exploratória de dados para entende-los melhor;
Separe os dados em conjuntos de treinamento (70%) e teste (30%);
Aplique os classificadores Gaussian Naive Bayes, K Nearest Neighbours (n_neighbors=3 e metric='euclidean') e Decision Tree;
Aplique o classification report para analisar a performance dos modelos e identifique o estimador com os melhores resultados.
 

Dica: para tornar os resultados comparáveis, utilize random_state=1.

OBS.: o problema de desbalanceamento consiste em uma situação onde a quantidade de amostras pertencentes a uma determinada classe é muito superior às amostras das demais classes. Uma forma simples de tratar o problema de desbalanceamento é realizar uma reamostragem dos dados conforme os passos a seguir:

Verificar o problema de desbalanceamento:
print(df.is_safe.value_counts())

Reamostragem:
c_0 = df[df.is_safe == 0].sample(n = 912, random_state=1)

c_1 = df[df.is_safe == 1]

df = pd.concat([c_0, c_1])

df = df.reset_index(drop=True)

Verificar o resultado obtido:
print(df.is_safe.value_counts())