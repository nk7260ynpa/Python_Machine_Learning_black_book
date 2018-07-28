import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
plt.scatter(df[0][:50],df[2][:50],color="red",marker="o",label="setosa")
plt.scatter(df[0][50:100],df[2][50:100],color="blue",marker="x",label="versicolor")
plt.xlabel("petal length")
plt.ylabel("sepal length")
plt.legend(loc="upper left")

print(df.head())




