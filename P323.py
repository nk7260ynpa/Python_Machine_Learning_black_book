from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X,y = make_moons(n_samples=200,
                noise=0.05,
                random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.show()