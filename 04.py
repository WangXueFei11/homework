import numpy as np

def eig_power(A,v0,eps):
    uk = v0
    flag = 1
    val_old = 0
    n = 0
    while flag:
        n = n+1
        vk = A*uk        
        val = vk[np.argmax(np.abs(vk))]        
        uk = vk/val              
        if (np.abs(val-val_old)<eps):
            flag = 0
        val_old = val
    return val, uk  
    
if __name__ == '__main__':
    A = np.matrix([[2,1],[4,5]], dtype='float')
    v0 = np.matrix([[1],[1]], dtype='float')
    eps = 1e-10
    val,uk = eig_power(A,v0,eps)
    print(val,uk)

                                            
