import numpy as np

aa = np.array([[1, 2, 3], [4 , 5, 6], [7, 8, 9]])


print(np.linalg.eig(aa)[1])
print(np.linalg.eig(aa)[0])
print(aa)
