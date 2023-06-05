import numpy as np


def df_vect(x, f, k, n_r, dim):
    """
      Uses high order finite difference method to compute numerical derivatives along arbitrary dimension of an array.
      Uses Frohnberg's method to compute coefficents.

    :param x: Elements in domain of known function values, should be a 1-D array
    :param f: Known function values, f(x)
    :param k: Order of the derivative
    :param n_r: About half of the window size, i.e. how many points in front and behind to be considered
         when computing the derivaitve. Practically, n_r = 3 or n_r = 4 works well
    :param dim: Dimension to compute the derivative along (usually temporal or spatial dimension)

    :return df: Computed finite difference vector
    """

    sz0 = np.shape(f)
    f = np.reshape(f, (int(sz0[dim]), int(f.size / sz0[dim]))).T
    df = np.zeros(np.shape(f))

    for j in range(0, n_r):
        df[:, j] = np.sum(fd_coefficient(k, x[j], x[0:n_r+j+1]) * f[:, 0:n_r+j+1], axis=1)

    for i in range(n_r, sz0[dim]-n_r):
        df[:, i] = np.sum(fd_coefficient(k, x[i], x[i-n_r:i+n_r+1]) * f[:, i-n_r:i+n_r+1], axis=1)

    for j in range(sz0[dim]-n_r, sz0[dim]):
        df[:, j] = np.sum(fd_coefficient(k, x[j], x[j-n_r:]) * f[:, j-n_r:], axis=1)

    df = np.reshape(np.transpose(df), sz0)
    
    return df


def fd_coefficient(k, xbar, x):
    """
      Compute coefficients for finite difference approximation for the
      derivative of order k at xbar based on grid values at points in x.
     
      This function returns a row vector c of dimension 1 by n, where n=length(x),
      containing coefficients to approximate u^{(k)}(xbar),
      the k'th derivative of u evaluated at xbar,  based on n values
      of u at x(1), x(2), ... x(n).
     
      If U is a column vector containing u(x) at these n points, then
      c*U will give the approximation to u^{(k)}(xbar).
     
      Note for k=0 this can be used to evaluate the interpolating polynomial
      itself.
     
      Requires length(x) > k.
      Usually the elements x(i) are monotonically increasing
      and x(1) <= xbar <= x(n), but neither condition is required.
      The x values need not be equally spaced but must be distinct.
     
      This program should give the same results as fdcoeffV.m, but for large
      values of n is much more stable numerically.
     
      Based on the program "weights" in
        B. Fornberg, "Calculation of weights in finite difference formulas",
        SIAM Review 40 (1998), pp. 685-691.
     From  http://www.amath.washington.edu/~rjl/fdmbook/  (2007)

    :param k: 
    :param xbar: 
    :param x: 
    :return: 
    """

    n = len(x)

    if k >= n:
        raise ValueError('*** len(x) must be larger than k')

    m = k

    c1 = 1
    c4 = x[0] - xbar
    c_capital = np.zeros((n, m + 1))
    c_capital[0, 0] = 1

    for i in range(n):
        mn = np.minimum(i, m)
        c2 = 1
        c5 = c4
        c4 = x[i] - xbar
        for j in range(0, i):
            c3 = x[i] - x[j]
            c2 = c2 * c3

            if j == i - 1:
                for s in range(mn, 0, -1):
                    c_capital[i, s] = c1 * (s * c_capital[i - 1, s - 1] - c5 * c_capital[i - 1, s]) / c2
                c_capital[i, 0] = -c1 * c5 * c_capital[i - 1, 0] / c2
            for s in range(mn, 0, -1):
                c_capital[j, s] = (c4 * c_capital[j, s] - s * c_capital[j, s - 1]) / c3

            c_capital[j, 0] = c4 * c_capital[j, 0] / c3

        c1 = c2

    c = np.transpose(c_capital[:, -1])

    return c
