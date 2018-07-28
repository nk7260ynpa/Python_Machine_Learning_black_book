from scipy.misc import comb
import math
import numpy as np
import matplotlib.pyplot as plt

def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probs = [comb(n_classifier, k) *
             error**k *
             (1-error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)

#print(ensemble_error(11, 0.25))

#--------------------------------------------------------------

error_range = np.arange(0.0, 1.01, 0.01)
ens_error = [ensemble_error(n_classifier=11, error=error) for error in error_range]
plt.plot(error_range, ens_error, label='Ensemble error', linewidth=2)
plt.plot(error_range, error_range, label='Base error', linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Enselble error')
plt.legend(loc='upper left')
plt.grid()
plt.show()
