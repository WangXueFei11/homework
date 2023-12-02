import numpy as np
mat = np.array([[2,1],[4,5]])
eigenvalue, featurevector = np.linalg.eig(mat)
print(eigenvalue)
print(featurevector)
