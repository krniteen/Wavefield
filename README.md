# Wavefield
# Wavefeild simulation using nodal integral method
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
import scipy.sparse as ssp
import scipy.sparse.linalg

class coeffs_A:
    def __init__(self,ax, az, i, j, n, m, k):

        # general node coefficients for first equation
        self.A1 = (-3 * ax**2 * az) / (-3 * az**2 + ax**2 * (-3 + az**2 * k[i, j]**2))
        self.A4 = (2 + (9 * ax * ax) / (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))) / (2 * az)
        self.A6 = (9 * az) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
        self.A8 = (9 * az) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))

        # general node coefficients for rest of first equation
        if j < m-1 :
            self.A2 = (-3 * ax * ax * az) / (-3 * az * az + ax * ax * (-3 + az * az * k[i, j+1] * k[i, j+1]))
            self.A3 = (8 + 9 * ax * ax * (1 / (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])) + 1 / (
                -3 * az * az + ax * ax * (-3 + az * az * k[i, j+1] * k[i, j+1])))) / (2 * az)
            self.A5 = (2 + (9 * ax * ax) / (-3 * az * az + ax * ax * (-3 + az * az * k[i, j+1] * k[i, j+1]))) / (
                2 * az)
            self.A7 = (9 * az) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j+1] * k[i, j+1])))
            self.A9 = (9 * az) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j+1] * k[i, j+1])))

        # bottom boundary coefficients
        if j == 0 :
            self.A11 = (3 * ax * ax * az) / (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))
            self.A22 = 1 / (2 * az) + (9 * az) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))) - (
                              3 * ax * ax * az * k[i, j] * k[i, j]) / (
                              2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
            self.A33 = -1 / (2 * az) + (9 * az) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j+1] * k[i, j]))) - (
                              3 * ax * ax * az * k[i, j] * k[i, j]) / (
                              2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
            self.A44 = (-9 * az) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
            self.A55 = (-9 * az) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))



        # top boundary coefficients
        if j == m-1 :
            self.B11 = (- 3 * ax * ax * az) / (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))
            self.B22 = 1 / (2 * az) - (9 * az) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))) + (
            3 * ax * ax * az * k[i, j] * k[i, j]) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
            self.B33 = -(1 / (2 * az) + (9 * az) / (2 * (-3 * az ** 2 + ax ** 2 * (-3 + az ** 2 * k[i, j] ** 2))) - (
            3 * ax ** 2 * az * k[i, j] ** 2) / (2 * (-3 * az ** 2 + ax ** 2 * (-3 + az ** 2 * k[i, j] ** 2))))
            self.B44 = (9 * az) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
            self.B55 = (9 * az) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))


class coeffs_B:
    def __init__(self,ax, az, i, j, n, m, k):

        #general node coefficients for second equation
        self.B1 = (-3 * ax * az * az) / (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))
        self.B3 = (9 * ax) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
        self.B4 = (9 * ax) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
        self.B8 = (2 + (9 * az * az) / (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))) / (2 * ax)

        # general node coefficients for rest of second equation
        if i < n-1 :
            self.B2 = (-3 * ax * az * az) / (-3 * az * az + ax * ax * (-3 + az * az * k[i+1, j] * k[i+1, j]))
            self.B5 = (9 * ax) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i+1, j] * k[i+1, j])))
            self.B6 = (9 * ax) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i+1, j] * k[i+1, j])))
            self.B7 = (8 + 9 * az * az * (1 / (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])) + 1 / (
                -3 * az * az + ax * ax * (-3 + az * az * k[i+1, j] * k[i+1, j])))) / (2 * ax)
            self.B9 = (2 + (9 * az * az) / (-3 * az * az + ax * ax * (-3 + az * az * k[i+1, j] * k[i+1, j]))) / (
                2 * ax)

        # left boundary coefficients
        if i == 0 :
            self.C11 = (3 * ax * az * az) / (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))
            self.C22 = (-9 * ax) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
            self.C33 = 1 / (2 * ax) + (9 * ax) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))) - (
                              3 * ax * az * az * k[i, j] * k[i, j]) / (
                              2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
            self.C44 = (-9 * ax) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
            self.C55 = -1 / (2 * ax) + (9 * ax) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))) - (
                              3 * ax * az * az * k[i, j] * k[i, j]) / (
                              2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))

        # right boundary coefficients
        if i == n-1 :
            self.D11 = (-3 * ax * az * az) / (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))
            self.D22 = (9 * ax) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
            self.D33 = (9 * ax) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
            self.D44 = -1 / (2 * ax) - (9 * ax) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))) + (
            3 * ax * az * az * k[i, j] * k[i, j]) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))
            self.D55 = 1 / (2 * ax) - (9 * ax) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j]))) + (
            3 * ax * az * az * k[i, j] * k[i, j]) / (2 * (-3 * az * az + ax * ax * (-3 + az * az * k[i, j] * k[i, j])))


def populate_A(k,ax,az,S):

    [nx, nz] = k.shape

    # top boundary coefficients
    BC_DT = 1
    BC_ET = 1j*np.real(k[0:nx-1,0])
    BC_FT = 0
    # bottom boundary coefficients
    BC_DB = 1
    BC_EB = -1j*np.real(k[0:nx-1,0])
    BC_FB = 0

    A_Uz = dok_matrix(((nx-1)*nz, (nz-1)*nx), dtype = 'complex_')
    A_Ux = dok_matrix(((nx-1)*nz, (nx-1)*nz), dtype = 'complex_')
    BB_1 = dok_matrix(((nx - 1) * nz,1), dtype = 'complex_')

    for j in np.arange(0, nz) :
        for i in np.arange(1, nx):
            row_ind = i - 1 + j * (nx - 1)
            coeff = coeffs_A(ax, az, i, j, nx, nz, k)
            if (j == 0) :
                BB_1[row_ind] = - BC_FB + coeff.A11 * BC_DB * S[i, j + 1]
                A_Uz[row_ind, i - 1] = BC_DB * coeff.A55                                           # Uz_{i - 1, j + 1}
                A_Uz[row_ind, i] = BC_DB * coeff.A44                                               # Uz_{i, j + 1}
                A_Ux[row_ind, i - 1 + (nx - 1) * j] = BC_EB[i - 1] + BC_DB * coeff.A33             # Ux_{i, j}
                A_Ux[row_ind, i - 1 + (nx - 1) * (j+1)] = BC_DB * coeff.A22                        # Ux_{i, j + 1}

            elif (j > 0) and (j < nz-1) :
                BB_1[row_ind] = coeff.A1 * S[i, j] + coeff.A2 * S[i, j + 1]
                A_Uz[row_ind, i - 1 + nx * (j - 1)] = coeff.A8                                     # Uz_{i - 1, j}
                A_Uz[row_ind, i - 1 + nx * j] = coeff.A9                                           # Uz_{i - 1, j + 1}
                A_Uz[row_ind, i + nx * (j - 1)] = coeff.A6                                         # Uz_{i, j}
                A_Uz[row_ind, i + nx * j] = coeff.A7                                               # Uz_{i, j + 1}
                A_Ux[row_ind, i - 1 + (nx - 1) * (j - 1)] = coeff.A4                               # Ux_{i, j - 1}
                A_Ux[row_ind, i - 1 + (nx - 1) * j] = coeff.A3                                     # Ux_{i, j}
                A_Ux[row_ind, i - 1 + (nx - 1) * (j + 1)] = coeff.A5                               # Ux_{i, j + 1}

            elif (j == nz-1) :

                BB_1[row_ind] = - BC_FT + coeff.B11 * BC_DT * S[i, j]
                A_Uz[row_ind, i - 1 + nx * (j - 1)] = BC_DT * coeff.B55                            # Uz_{i - 1, j}
                A_Uz[row_ind, i + nx * (j - 1)] = BC_DT * coeff.B44                                # Uz_{i, j}
                A_Ux[row_ind, i - 1 + (nx - 1) * (j - 1)] = BC_DT * coeff.B33                      # Ux_{i, j - 1}
                A_Ux[row_ind, i - 1 + (nx - 1) * j] = BC_ET[i - 1] + BC_DT * coeff.B22             # Ux_{i, j}

    return A_Uz, A_Ux, BB_1

def populate_B(k,ax,az,S):

    [nx, nz] = k.shape

    # right boundary coefficients
    BC_DR = 1
    BC_ER = 1j*np.real(k[0,0:nz-1])
    BC_FR = 0
    # left boundary coefficients
    BC_DL = 1
    BC_EL = -1j*np.real(k[0,0:nz-1])
    BC_FL = 0

    B_Uz = dok_matrix(((nz-1)*nx, (nz-1)*nx), dtype = 'complex_')
    B_Ux = dok_matrix(((nz-1)*nx, nz*(nx-1)), dtype = 'complex_')
    BB_2 = dok_matrix(((nz-1)*nx,1), dtype = 'complex_')

    for j in np.arange(1, nz) :
        for i in np.arange(0, nx):
            row_ind = i  + (j - 1) * nx
            coeff = coeffs_B(ax, az, i, j, nx, nz, k)
            if i==0 :
                BB_2[row_ind] = - BC_FL + BC_DL * coeff.C11 * S[i + 1, j]
                B_Uz[row_ind, i + (j - 1) * nx] = BC_EL[j - 1] + BC_DL * coeff.C55          # Uz_{i, j + 1}
                B_Uz[row_ind, i + 1 + (j - 1) * nx] = BC_DL * coeff.C33                     # Uz_{i + 1, j + 1}
                B_Ux[row_ind, i + (j - 1) * (nx - 1)] = BC_DL * coeff.C22                   # Ux_{i + 1, j - 1}
                B_Ux[row_ind, i + j * (nx - 1)] = BC_DL * coeff.C44                         # Ux_{i + 1, j}

            elif (i > 0) and (i < nx-1) :
                BB_2[row_ind] = coeff.B1 * S[i, j] + coeff.B2 * S[i + 1, j]
                B_Uz[row_ind, i - 1 + nx * (j - 1)] = coeff.B8                              # Uz_{i - 1, j}
                B_Uz[row_ind, i + nx * (j - 1)] = coeff.B7                                  # Uz_{i, j}
                B_Uz[row_ind, i + 1 + nx * (j - 1)] = coeff.B9                              # Uz_{i + 1, j}
                B_Ux[row_ind, i - 1 + (nx - 1) * (j - 1)] = coeff.B4                        # Ux_{i, j - 1}
                B_Ux[row_ind, i + (nx - 1) * (j - 1)] = coeff.B6                            # Ux_{i + 1, j - 1}
                B_Ux[row_ind, i - 1 + (nx - 1) * j] = coeff.B3                              # Ux_{i, j}
                B_Ux[row_ind, i + (nx - 1) * j ] = coeff.B5                                 # Ux_{i + 1, j}

            elif i == nx-1 :
                BB_2[row_ind] = - BC_FR + BC_DR * coeff.D11 * S[i, j]
                B_Uz[row_ind, i - 1 + nx * (j - 1)] = BC_DR * coeff.D44                     # Uz_{i - 1, j}
                B_Uz[row_ind, i + nx * (j - 1)] = BC_DR * coeff.D55 + BC_ER[j - 1]          # Uz_{i, j}
                B_Ux[row_ind, i - 1 + (nx - 1) * (j - 1)] = BC_DR * coeff.D33               # Ux_{i, j - 1}
                B_Ux[row_ind, i - 1 + (nx - 1) * j] = BC_DR * coeff.D22                     # Ux_{i, j}

    return B_Uz, B_Ux, BB_2

def solve_NIM_system(k,ax,az,S):

    # check the number of grid points per wavelength
    #print('Number of grid points per wavelength='+str(np.round(np.pi/(np.max(np.real(k))*np.min(ax,az)), 2)))

    A_Uz, A_Ux, BB_1 = populate_A(k, ax, az, S)
    B_Uz, B_Ux, BB_2 = populate_B(k, ax, az, S)
    nx, nz = k.shape
    b = csr_matrix(ssp.vstack([BB_1, BB_2]))
    A = ssp.hstack([A_Uz, A_Ux])
    B = ssp.hstack([B_Uz, B_Ux])
    M = csr_matrix(ssp.vstack([A, B]),dtype = 'complex_')

    Uxz = scipy.sparse.linalg.spsolve(M,(b))

    X1 = Uxz[0:(nx)*(nz-1)]
    X2 = Uxz[(nx) * (nz-1):]
    Uz = (np.reshape(X1, [nx, nz-1], order='F'))
    Ux = (np.reshape(X2, [nx-1, nz], order='F'))

    return Ux, Uz

if __name__=='__main__':

    import matplotlib.pyplot as plt
    import scipy.special as sspl
    import scipy.signal

    vp0 = 2.0*np.ones((101, 101))
    f = 4.0
    k = 2.0 * np.pi*f/vp0
    ax = 0.025
    az = 0.025
    h = [2*ax, 2*az]
    nx, nz  = vp0.shape
    x  = np.arange(-0.5*(nx-1),0.5*(nx+1))*2*ax
    z  = np.arange(-0.5*(nz-1),0.5*(nz+1))*2*az
    zz,xx = np.meshgrid(z,x)

    xs = x[int(nx/2)]
    zs = z[int(nz/2)]

    # Spatial source supplied to the solver
    sx = 1 * h[0]
    sz = 1 * h[1]

    spat_source = dok_matrix(xx.shape)
    spat_source[int(nx/2),int(nz/2)] = 1.0
    spat_source = -spat_source/(4*ax*az)

    Ux, Uz = solve_NIM_system(k,ax,az,spat_source)

    z_Ux = z + az
    x_Ux = np.delete(x, 0, 0)
    zz_Ux, xx_Ux = np.meshgrid(z_Ux, x_Ux)
    G_2D_Ux = 0.25*1j*sspl.hankel2(0, k[0,0] * np.power((np.power((zz_Ux - zs), 2)
                        + np.power((xx_Ux - xs), 2)),0.5) + np.finfo(np.float32).eps)

    x_Uz = x + ax
    z_Uz = np.delete(z, 0, 0)
    zz_Uz, xx_Uz = np.meshgrid(z_Uz, x_Uz)
    G_2D_Uz = 0.25*1j*sspl.hankel2(0, k[0,0] * np.power((np.power((zz_Uz - zs), 2)
                        + np.power((xx_Uz - xs), 2)),0.5) + np.finfo(np.float32).eps)


    ##### Simpson's rule
    """
    Nsim = 50
    G_an = np.zeros((nx-1,nz), dtype = 'complex_')
    for i in np.arange(nx-1):
        lb = x[0] + i*h[0]
        rb = lb + h[0]
        x1 = np.linspace(lb,rb,Nsim)
        f = np.zeros((Nsim, nz), dtype = 'complex_')
        for p in np.arange(nz):
            for q in np.arange(Nsim):
                r = k[0,0] * ((x1[q]-xs) ** 2 + (z[p]-zs) ** 2) ** 0.5
                f[q, p] = 0.25*1j* sspl.hankel2(0, r + np.finfo(np.float16).eps)

            G_an[i, p] = (1 / (3 * Nsim)) * (f[0, p] + 2 * np.sum(f[np.arange(2,Nsim-1,2), p])
                      + 4 * np.sum(f[np.arange(1,Nsim-1,2), p]) + f[-1, p])

    G_sim = np.conj(G_an)
    """

    plt.figure()
    plt.imshow((np.imag(G_2D_Ux) - np.imag(Ux))/(np.imag(G_2D_Ux)+np.finfo(np.float16).eps),
               extent=[x_Ux[0], x_Ux[-1],z_Ux[-1],z_Ux[0]], cmap='RdBu',vmin=-1,vmax=1)
    plt.xlabel('x [km]')
    plt.ylabel('z [km]')
    plt.colorbar()
    plt.title('Imaginary, (Ux - G)/G')

    plt.figure()
    plt.imshow((np.real(G_2D_Ux) - np.real(Ux))/(np.abs(G_2D_Ux)+np.finfo(np.float16).eps),
               extent=[x_Ux[0], x_Ux[-1],z_Ux[-1],z_Ux[0]], cmap='RdBu', vmin=-1,vmax=1)
    plt.xlabel('x [km]')
    plt.ylabel('z [km]')
    plt.colorbar()
    plt.title('Real, (Ux - G)/G')

    plt.figure()
    plt.imshow((np.abs(G_2D_Ux) - np.abs(Ux))/(np.abs(G_2D_Ux)+np.finfo(np.float16).eps),
               extent=[x[0], x[-1],z[-1],z[0]], cmap='RdBu',vmin=-1,vmax=1)
    plt.xlabel('x [km]')
    plt.ylabel('z [km]')
    plt.colorbar()
    plt.title('Abs, (Ux - G)/G')


    plt.figure()
    plt.imshow((np.imag(G_2D_Uz) - np.imag(Uz))/(np.imag(G_2D_Uz)+np.finfo(np.float16).eps),
               extent=[x[1], x[-1],z[-1],z[1]], cmap='RdBu', vmin=-1, vmax=1)
    plt.xlabel('x [km]')
    plt.ylabel('z [km]')
    plt.colorbar()
    plt.title('Imaginary, (Uz - G)/G')

    plt.figure()
    plt.imshow((np.real(G_2D_Uz) - np.real(Uz))/(np.real(G_2D_Uz)+np.finfo(np.float16).eps),
               extent=[x[1], x[-1],z[-1],z[1]], cmap='RdBu',vmin=-1, vmax=1)
    plt.xlabel('x [km]')
    plt.ylabel('z [km]')
    plt.colorbar()
    plt.title('Real,(Uz - G)/G')

    plt.figure()
    plt.imshow((np.abs(G_2D_Uz) - np.abs(Uz))/(np.abs(G_2D_Uz)+np.finfo(np.float16).eps),
               extent=[x[1], x[-1],z[-1],z[1]], cmap='RdBu',vmin=-1, vmax=1)
    plt.xlabel('x [km]')
    plt.ylabel('z [km]')
    plt.colorbar()
    plt.title('Abs, (Uz - G)/G')


    plt.figure()
    plt.plot(x_Ux,np.real(Ux[:,51]))
    plt.plot(x_Ux,(np.real(G_2D_Ux[:,51])), 'k')

    plt.figure()
    plt.plot(x_Ux,np.imag(Ux[:,51]))
    plt.plot(x_Ux,(np.imag(G_2D_Ux[:,51])), 'k')

    plt.figure()
    plt.plot(z_Ux,np.real(Ux[50,:]))
    plt.plot(z_Ux,(np.real(G_2D_Ux[50,:])), 'k')

    plt.figure()
    plt.plot(z_Ux,np.imag(Ux[50,:]))
    plt.plot(z_Ux,(np.imag(G_2D_Ux[50,:])), 'k')

    """
    plt.figure()
    plt.plot(z,np.real(Uz[51,:]))
    plt.plot(z,(np.real(G_2D[51,:])), 'k')

    plt.figure()
    plt.plot(z,np.imag(Uz[51,:]))
    plt.plot(z,(np.imag(G_2D[51,:])), 'k')
    """

    plt.show()
