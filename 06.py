import numpy as np

I = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def eig_power(A, v0, eps):
    uk = v0
    flag = 1
    val_old = 0
    n = 0
    while flag:
        n = n + 1
        vk = (A) * uk
        val = vk[np.argmax(np.abs(vk))]
        uk = vk / val
        if (np.abs(val - val_old) < eps):
            flag = 0
        val_old = val
    return val, uk

if __name__ == '__main__':
    A = np.matrix([[6.3333333333, 2.5, -5], [2.5, 1., -2], [-5., -2., 4]], dtype='float')

    B = np.matrix([[6.3333333333, 2.5, -5], [2.5, 1., -2], [-5., -2., 4]], dtype='float')
    eigen,feature=np.linalg.eig(B)
    
    print("A:\n", A)
    print(eigen)
    print(feature)

    v1 = np.matrix([[1], [1], [1]], dtype='float')
    v2 = np.matrix([[1], [1], [1]], dtype='float')
    v3 = np.matrix([[1], [1], [1]], dtype='float')
    eps = 1e-10

    print("\nX1:")
    val1, uk1 = eig_power(A, v1, eps)
    x1 = float(val1[0, 0])
    print(x1)
    print(uk1)

    print("\nX2:")
    val2, uk2 = eig_power(A - x1 * I, v2, eps)
    x2 = float(val2[0, 0])
    print(x1 + x2)
    print(uk2)

    print("\nX3:")
    val3, uk3 = eig_power(A - x1 * I - x2 * I, v3, eps)
    x3 = float(val3[0, 0])
    print(x1 + x2 + x3)
    print(uk3)
