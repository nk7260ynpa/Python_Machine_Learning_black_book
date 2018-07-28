from P023 import *
from P028 import *
import pandas as pd
import  matplotlib.pyplot as plt

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)



#--------------------------------------------
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel("sepal length [cm]")
plt.ylabel("sepal length [cm]")
plt.legend(loc="upper left")
plt.show()
