import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sp_linalg
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.interpolate as scinterp
import sys
sys.path.append("..")

def default_drift(x):
    # Set Parameters for defining Two Well Potential, Drift
    p1 = np.array([-1/np.sqrt(2), -0.0])
    p2 = np.array([1/np.sqrt(2), 0.0])
    p = [p1, p2]
    mat = np.array([[0.5, 0.0], [0.0, 1.0]])
    c1 = mat
    c2 = mat
    c_inv = [np.linalg.inv(c1), np.linalg.inv(c2)] #store inverse covariances
    E = 10.0

    return manywell_drift(x, p, c_inv, E)


def default_potential(x):
    # Set Parameters for defining Two Well Potential, Drift
    p1 = np.array([-1/np.sqrt(2), -0.0])
    p2 = np.array([1/np.sqrt(2), 0.0])
    p = [p1, p2]
    mat = np.array([[0.5, 0.0], [0.0, 1.0]])
    c1 = mat
    c2 = mat
    c_inv = [np.linalg.inv(c1), np.linalg.inv(c2)] #store inverse covariances
    E = 10.0

    return manywell_potential(x, p, c_inv, E)


def manywell_drift(x, p, c_inv, E):
    r"""Returns -1*gradient of potential function in method manywell_potential()

    Wells are positioned at p1, p2, p3

    Parameters
    ----------
    x : array-like
        Two dimensional vector, input point
    p : array-like  
        Two dim vector, list of centers for wells
    c_inv : array-like
        2 x 4 array, [c1_inv c2_inv] respective inverse covariances for wells
    E : scalar
        Controls height of barrier between wells
    Returns
    -------
    out : array-like
        2-dim. vector, -1*gradient of potential at 'x'.

    """
    s = 0
    grad = np.zeros(2)
    for i in range(2):
        z = np.reshape(x - p[i], (2, 1))
        #mat = np.matmul(c_inv[i], z)
        #r = np.matmul(z.T, mat)
        mat = np.dot(c_inv[i],z)
        r = np.dot(z.T, mat)
        e = np.exp(-r)
        s += e
        grad += (-2*e*mat).flatten()
    grad = -E * grad
    grad[0] = grad[0] + 4*x[0]**3
    grad[1] = grad[1] + 4*x[1]**3
    out = -grad      # Overdamped Langevin drift is -1*grad U
    return out


def manywell_potential(x, p, c_inv, E, symmetric=False):
    
    r"""2D potential function 
    
    2 Wells are positioned at p = [p1, p2]

    Parameters
    ----------
    x : array-like
        Two dimensional vector.
    p : array-like  
        Two dim vector, list of centers for wells
    c_inv : array-like
        2 x 4 array, [c1_inv c2_inv] respective inverse covariances for wells
    E : scalar
        Controls height of barrier between wells
    Returns
    -------
    U : scalar
        Potential evaluated at 'x'.

    """
    s = 0
    e = np.zeros(2)
    for i in range(2):
        z = np.reshape(x - p[i], (2, 1))
        r = np.matmul(z.T, np.matmul(c_inv[i], z))
        e = np.exp(-r)
        s += e
    U = -E*s + x[0]**4 + x[1]**4

    if symmetric: 
        # symmetric -> all covariances are the same, can normalize potential to 0
        # Compute constant to bring minimum to 0
        z = np.reshape(p[1] - p[0], (2, 1))
        C = E*(1 + np.exp(-np.matmul(z.T, np.matmul(c_inv[1], z))) 
                 + p[0][0]**4 + p[0][1]**4)
        U = U + C 
    return U


def threewell_drift(v):
    
    r"""    Parameters
    ----------
    v : array-like
        Two dimensional vector.
    Returns
    -------
    out : vector
        Drift evaluated at 'v'.
    """
    x = v[0] 
    y = v[1]

    dx_exp1 = -6*x*np.exp(-x**2 - (y - 1./3.)**2) 
    dy_exp1 = -6*(y - 1./3.)*np.exp(-x**2 - (y - 1./3.)**2) 
    
    dx_exp2 = -6*x*np.exp(-x**2 - (y - 5./3.)**2) 
    dy_exp2 = -6*(y - 5./3.)*np.exp(-x**2 - (y - 5./3.)**2) 
    
    dx_exp3 = -10*(x - 1)*np.exp(-(x - 1)**2 - y**2) 
    dy_exp3 = -10*y*np.exp(-(x - 1)**2 - y**2) 
    
    dx_exp4 = -10*(x + 1)*np.exp(-(x + 1)**2 - y**2) 
    dy_exp4 = -10*y*(np.exp(-(x + 1)**2 - y**2))
    
    dx_poly1 = 0.8*x**3
    dy_poly2 = 0.8*(y - 1./3.)**3
    grad = np.array([0.0, 0.0])
    grad[0] = dx_exp1 - dx_exp2 - dx_exp3 - dx_exp4 + dx_poly1
    grad[1] = dy_exp1 - dy_exp2 - dy_exp3 - dy_exp4 + dy_poly2
    out = -grad
    return out 


def threewell_potential(v):
    
    r"""    Parameters
    ----------
    v : array-like
        Two dimensional vector.
    Returns
    -------
    U : scalar
        Potential evaluated at 'v'.

    """
    x = v[0] 
    y = v[1]

    exp1 = 3*np.exp(-x**2 - (y - 1./3.)**2) 
    exp2 = 3*np.exp(-x**2 - (y - 5./3.)**2) 
    exp3 = 5*np.exp(-(x - 1)**2 - y**2) 
    exp4 = 5*np.exp(-(x + 1)**2 - y**2) 
    poly1 = 0.2*x**4
    poly2 = 0.2*(y - 1./3.)**4
    U = exp1 - exp2 - exp3 - exp4 + poly1 + poly2
    return U


def muller_drift(x):

    # Define list of parameters
    A = np.array([-200., -100., -170., 15.])
    nu = np.array([[1., 0.], 
                   [0., 0.5], 
                   [-0.5, 1.5], 
                   [-1., 1.]])
    sig_inv = np.array([[[1., 0.],
                         [0., 10.]],
                        [[1., 0.],
                         [0., 10.]],
                        [[6.5, -5.5],
                         [-5.5, 6.5]],
                        [[-0.7, -0.3],
                         [-0.3, -0.7]]])
    force = np.array([0., 0.])
    for i in range(4):
        u = x - nu[i, :]
        M = sig_inv[i, :, :]
        force += 2*A[i]*np.exp(-u.dot(M.dot(u)))*M.dot(u) # THIS IS -grad V not grad V!!!
    return force

def muller_potential(x):

    # Define list of parameters
    A = np.array([-200., -100., -170., 15.])
    nu = np.array([[1., 0.], 
                   [0., 0.5], 
                   [-0.5, 1.5], 
                   [-1., 1.]])
    sig_inv = np.array([[[1., 0.],
                         [0., 10.]],
                        [[1., 0.],
                         [0., 10.]],
                        [[6.5, -5.5],
                         [-5.5, 6.5]],
                        [[-0.7, -0.3],
                         [-0.3, -0.7]]])
    V = 0.
    for i in range(4):
        u = x - nu[i, :]
        M = sig_inv[i, :, :]
        V += A[i]*np.exp(-u.dot(M.dot(u)))
    return V

def matlab_muller(x):
    a = np.array([-1.,-1.,-6.5,0.7])
    b = np.array([0.,0.,11.,0.6])
    c = np.array([-10.,-10.,-6.5,0.7])
    D = np.array([-200.,-100.,-170.,15])
    X = np.array([1.,0.,-0.5,-1.])
    Y = np.array([0.,0.5,1.5,1.])
    V=0.
    for i in range(4):
        Vnew = (D[i]*np.exp(a[i]*(x[0]-X[i])**2 + b[i]*(x[0]-X[i])*(x[1]-Y[i])
                + c[i]*(x[1]-Y[i])**2))
        V=V+Vnew
    
    return V

def muller_laplacian(x):

    # Define list of parameters
    A = np.array([-200., -100., -170., 15.])
    nu = np.array([[1., 0.], 
                   [0., 0.5], 
                   [-0.5, 1.5], 
                   [-1., 1.]])
    sig_inv = np.array([[[1., 0.],
                         [0., 10.]],
                        [[1., 0.],
                         [0., 10.]],
                        [[6.5, -5.5],
                         [-5.5, 6.5]],
                        [[-0.7, -0.3],
                         [-0.3, -0.7]]])
    Delta = 0.
    for i in range(4):
        u = x - nu[i, :]
        M = sig_inv[i, :, :]
        quad = -u.dot(M.dot(u))
        gaussian = A[i]*np.exp(quad)
        grad_quad = -2*M.dot(u)
        Delta_quad = -2*(M[0, 0] + M[1, 1])
        Delta_quad = -2*(np.trace(M))
        Delta_gaussian = (Delta_quad + np.sum(grad_quad**2, axis=None))*gaussian
        Delta += Delta_gaussian
    return Delta

def bereszabo_drift(v, beta_inv=1.0):

    x0 = 2.2
    omega_sq = 4.0
    bigomega_sq = 1.01*omega_sq
    x = v[0] 
    y = v[1]

    drift = np.array([0., 0.])
    drift[0] = bigomega_sq*(x - y)
    drift[1] = -bigomega_sq*(x - y)

    if (x < -0.5*x0):
        drift[0] += omega_sq*(x + x0)
    elif (x < 0.5*x0):
        drift[0] += -omega_sq*x
    else:
        drift[0] += omega_sq*(x - x0)
    drift *= beta_inv
    drift *= -1
    return drift

def bereszabo_potential(v, beta_inv=1.0):

    x0 = 2.2
    omega_sq = 4.0
    bigomega_sq = 1.01*omega_sq
    Delta = 0.25*omega_sq*(x0**2)
    x = v[0] 
    y = v[1]

    U = 0
    if (x < -0.5*x0):
        U += -Delta + 0.5*omega_sq*(x + x0)**2 
    elif (x < 0.5*x0):
        U += -0.5*omega_sq*(x**2)
    else:
        U += -Delta + 0.5*omega_sq*(x - x0)**2
    U *= beta_inv
    U += 0.5*beta_inv*bigomega_sq*(x - y)**2
    return U


# def twowell_drift(x):
#     h = 5.0     # Barrier Height
#     alpha = np.deg2rad(140);    # aperature angle at saddle point
#     drift = np.zeros(2)
#     drift[0] = -4*h*(x[0]**2 - 1)*x[0]
#     drift[1] = -4*h*x[1]/np.tan(alpha/2.)
#     return drift


# def twowell_potential(x):
#     h = 5.0     # Barrier Height
#     alpha = np.deg2rad(140);    # aperature angle at saddle point

#     V = h*(x[0]**2 - 1)**2 + 2*h*(x[1]/np.tan(alpha/2.))**2
#     return V

def twowell_potential(x):
    mu = np.array([[-1.0, 0.0],[1.0, 0.0]]) # gaussian means
    c_inv = np.array([[2.0, 0.0],[0.0 ,1.0]])    # gaussian inverse covariance
    energy = 10.0
    my_sum = 0.0
    for i in range(2):
        z = (x - mu[i, :])
        my_sum = my_sum + np.exp(-z@(c_inv@z.T))
    V = -energy*my_sum + np.sum(x**4)
    return V

def twowell_drift(x): 
    mu = np.array([[-1.0, 0.0],[1.0, 0.0]]) # gaussian means
    c_inv = np.array([[2.0, 0.0],[0.0 ,1.0]])    # gaussian inverse covariance
    energy = 10.0
    dV = np.zeros(x.shape)
    for i in range(2):
        z = (x - mu[i, :])
        mat = z@c_inv
        e = np.exp(-z@(c_inv@z.T))
        dV = dV - 2*e*mat
    dV = energy*dV - 4*(x**3)
    return dV

def temp_switch_potential(x):
    hx = 0.5
    hy = 1.0
    x0 = 0.
    delta = 1/20.

    a = (1/5.)*(1. + 5*np.exp(-(1/delta)*(x[0] - x0)**2))**2
    U = hx*(x[0]**2 - 1.)**2 + (hy + a)*(x[1]**2 - 1)**2
    return U

def temp_switch_drift(x):
    hx = 0.5
    hy = 1.0
    x0 = 0.
    delta = 1/20.

    a = (1/5.)*(1. + 5*np.exp(-(1/delta)*(x[0] - x0)**2))**2
    ax = (2/5)*(1. + 5*np.exp(-(1/delta)*(x[0] - x0)**2))
    ax *= (-(10/delta)*(x[0] - x0)*np.exp(-(1/delta)*(x[0] - x0)**2))
    Ux = 4*hx*(x[0]**2 - 1.)*x[0] + ax*(x[1]**2 - 1)**2
    Uy = 4*(hy + a)*(x[1]**2 - 1)*x[1]
    drift = -1*np.array([Ux, Uy])
    return drift


def ornstein_potential(x):
    V = 0.5*np.sum(x**2)
    return V

def ornstein_drift(x):
    out = -x
    return out

if __name__ == '__main__':
    main()



