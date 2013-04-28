import numpy as np
def Gradient_decent(K, S, a0, max_iter)
    """
    input:
        K: labeled data x anchors
        S: pairwise label matrix, similar: +1, nonsimilar: -1
        a0: init value of a
        max_iter: maximum iteration in gradient decent
    output:
        a_star: result of a
    """
    (n, m) = K.shape
    y = np.dot(K, a0)
    y = 2 * (1 + np.exp(-1 * y)) ** (-1) -1
    
    a1 = a0
    delta = np.zeros((1, max_iter + 2))
    delta[0] = 0
    delta[1] = 1
    beta = np.zeros((1, max_iter + 1))
    beta[0] = 1

    for i in xrange(max_iter):
        alpha = (delta[i] - 1) / delta[i + 1]
        v = a1 + alpha * (a1 - a0)
        y0 = np.dot(K, v)
        y1 = 2 * ( 1 + np.exp( -1 * y0))**(-1) -1
        gv = -np.dot( np.dot(y1.T, S), y1 )
        ty = np.dot(S, y1) * (np.ones(n, 1) - y1 ** 2)
        dgv = - np.dot(K.T, ty)

        flag = 0 
        for j in xrange(50):
            b = 2 ** j * beta[j]
            z = v - dgv / b
            y0 = np.dot(K, z)
            y1 = 2 * (1 + np.exp( -1 * y0)) ** (-1) - 1
            gz = -np.dot(np.dot(y1.T, S), y1)
            dif = z - v
            gvz = gv + np.dot(dgv.T, dif) + b * np.dot(dif.T, dif) / 2

            if gv <= gvz
                flag = 1
                beta[i+1] = b
                a0 = a1
                a1 = z
                break

        if flag == 0
            t = t - 1
            break
        else:
            delta[i + 2] = (1 + np.sqrt(1 + 4 * delta[i + 1] ** 2 ) / 2
    a = a1
    return a
