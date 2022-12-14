import numpy as np

def check_integrability(normals):
    #  CHECK_INTEGRABILITY check the surface gradient is acceptable
    #   normals: normal image
    #   p : df / dx
    #   q : df / dy
    #   SE : Squared Errors of the 2 second derivatives

    # initalization
    p = np.zeros(normals.shape[:2])
    q = np.zeros(normals.shape[:2])
    SE = np.zeros(normals.shape[:2])

    """
    ================
    Your code here
    ================
    Compute p and q, where
    p measures value of df / dx
    q measures value of df / dy
    
    """
    p = normals[:,:,0] / normals[:,:,2]
    q = normals[:,:,1] / normals[:,:,2]

    # change nan to 0
    p[p!=p] = 0
    q[q!=q] = 0

    """
    ================
    Your code here
    ================
    approximate second derivate by neighbor difference
    and compute the Squared Errors SE of the 2 second derivatives SE

    """
    # Create shifted p and q and approximate derivatives.
    p1 = np.roll(p, -1, axis=0)
    p1[:,-1] = 0
    q1 = np.roll(q, -1, axis=1)
    q1[-1,:] = 0
    py = p1 - p
    qx = q1 - q
    SE = (py - qx)**2

    return p, q, SE


if __name__ == '__main__':
    normals = np.zeros([10,10,3])
    check_integrability(normals)