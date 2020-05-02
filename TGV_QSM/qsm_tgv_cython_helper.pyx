#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from numpy cimport *
from cython.parallel cimport prange

cdef extern from "math.h" nogil:
    float fabs(float)

cdef extern from "math.h" nogil:
    float sqrtf(float)

###############################################
# TGV-QSM routines
    
def tgv_update_eta(ndarray[float, ndim=3, mode="c"] eta not None, \
                   ndarray[float, ndim=3, mode="c"] phi not None, \
                   ndarray[float, ndim=3, mode="c"] chi not None, \
                   ndarray[float, ndim=3, mode="c"] laplace_phi0 not None, \
                   ndarray[float, ndim=3, mode="c"] mask not None, \
                   float sigma, float res0, float res1, float res2):
    """Update eta <- eta + sigma*mask*(-laplace(phi) + wave(chi) - laplace_phi0). """

    cdef float res0inv = <float>(1.0)/(res0**2)
    cdef float res1inv = <float>(1.0)/(res1**2)
    cdef float res2inv = <float>(1.0)/(res2**2)

    cdef float wres0inv = <float>(1.0/3.0)/(res0**2)
    cdef float wres1inv = <float>(1.0/3.0)/(res1**2)
    cdef float wres2inv = <float>(-2.0/3.0)/(res2**2)

    cdef int nx = eta.shape[0]
    cdef int ny = eta.shape[1]
    cdef int nz = eta.shape[2]

    cdef int i, j, k
    cdef float phi0, phi1m, phi1p, phi2m, phi2p, phi3m, phi3p, laplace
    cdef float chi0, chi1m, chi1p, chi2m, chi2p, chi3m, chi3p, wave

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                # compute -laplace(phi)
                phi0  = phi[i,j,k]
                phi1m = phi[i-1,j,k] if (i > 0)    else phi0
                phi1p = phi[i+1,j,k] if (i < nx-1) else phi0
                phi2m = phi[i,j-1,k] if (j > 0)    else phi0
                phi2p = phi[i,j+1,k] if (j < ny-1) else phi0
                phi3m = phi[i,j,k-1] if (k > 0)    else phi0
                phi3p = phi[i,j,k+1] if (k < nz-1) else phi0

                laplace =   (<float>(2.0)*phi0 - phi1m - phi1p)*res0inv \
                          + (<float>(2.0)*phi0 - phi2m - phi2p)*res1inv \
                          + (<float>(2.0)*phi0 - phi3m - phi3p)*res2inv

                # compute wave(chi)
                chi0  = chi[i,j,k]
                chi1m = chi[i-1,j,k] if (i > 0)    else chi0
                chi1p = chi[i+1,j,k] if (i < nx-1) else chi0
                chi2m = chi[i,j-1,k] if (j > 0)    else chi0
                chi2p = chi[i,j+1,k] if (j < ny-1) else chi0
                chi3m = chi[i,j,k-1] if (k > 0)    else chi0
                chi3p = chi[i,j,k+1] if (k < nz-1) else chi0

                wave =   (<float>(-2.0)*chi0 + chi1m + chi1p)*wres0inv \
                       + (<float>(-2.0)*chi0 + chi2m + chi2p)*wres1inv \
                       + (<float>(-2.0)*chi0 + chi3m + chi3p)*wres2inv

                eta[i,j,k] += sigma*mask[i,j,k] \
                              *(laplace + wave - laplace_phi0[i,j,k])


def tgv_update_p(ndarray[float, ndim=4, mode="c"] p not None, \
                 ndarray[float, ndim=3, mode="c"] chi not None, \
                 ndarray[float, ndim=4, mode="c"] w not None, \
                 ndarray[float, ndim=3, mode="c"] mask not None, \
                 ndarray[float, ndim=3, mode="c"] mask0 not None, \
                 float sigma, float alpha,
                 float res0, float res1, float res2):
    """Update p <- P_{||.||_\infty <= alpha}(p + sigma*(mask0*grad(phi_f) - mask*w). """

    cdef float alphainv = <float>1.0/alpha
    cdef float res0inv = <float>1.0/res0
    cdef float res1inv = <float>1.0/res1
    cdef float res2inv = <float>1.0/res2

    cdef int nx = p.shape[0]
    cdef int ny = p.shape[1]
    cdef int nz = p.shape[2]

    cdef int i, j, k
    cdef float dxp, dyp, dzp, px, py, pz, pabs, chi0, sigmaw0, sigmaw

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                chi0 = chi[i,j,k]

                dxp = (chi[i+1,j,k] - chi0)*res0inv if (i < nx-1) else <float>0.0
                dyp = (chi[i,j+1,k] - chi0)*res1inv if (j < ny-1) else <float>0.0
                dzp = (chi[i,j,k+1] - chi0)*res2inv if (k < nz-1) else <float>0.0

                sigmaw0 = sigma*mask0[i,j,k]
                sigmaw  = sigma*mask[i,j,k]

                px = p[i,j,k,0] + sigmaw0*dxp - sigmaw*w[i,j,k,0]
                py = p[i,j,k,1] + sigmaw0*dyp - sigmaw*w[i,j,k,1]
                pz = p[i,j,k,2] + sigmaw0*dzp - sigmaw*w[i,j,k,2]
                pabs = sqrtf(px*px + py*py * pz*pz)*alphainv
                pabs = <float>1.0/pabs if (pabs > <float>1.0) else <float>1.0

                p[i,j,k,0] = px*pabs
                p[i,j,k,1] = py*pabs
                p[i,j,k,2] = pz*pabs

def tgv_update_q(ndarray[float, ndim=4, mode="c"] q not None, \
                 ndarray[float, ndim=4, mode="c"] u not None, \
                 ndarray[float, ndim=3, mode="c"] weight not None, \
                 float sigma, float alpha,
                 float res0, float res1, float res2):
    """Update q <- P_{||.||_\infty <= alpha}(q + sigma*weight*symgrad(u)). """

    cdef float alphainv = <float>1.0/alpha
    cdef float res0inv = <float>1.0/res0
    cdef float res1inv = <float>1.0/res1
    cdef float res2inv = <float>1.0/res2
    cdef float res0inv2 = <float>0.5/res0
    cdef float res1inv2 = <float>0.5/res1
    cdef float res2inv2 = <float>0.5/res2

    cdef int nx = q.shape[0]
    cdef int ny = q.shape[1]
    cdef int nz = q.shape[2]

    cdef int i, j, k
    cdef float wxx, wxy, wxz, wyy, wyz, wzz, qabs, sigmaw

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                # compute symgrad(u)
                if (i < nx-1):
                    wxx = res0inv* (u[i+1,j,k,0] - u[i,j,k,0])
                    wxy = res0inv2*(u[i+1,j,k,1] - u[i,j,k,1])
                    wxz = res0inv2*(u[i+1,j,k,2] - u[i,j,k,2])
                else:
                    wxx = <float>0.0
                    wxy = <float>0.0
                    wxz = <float>0.0

                if (j < ny-1):
                    wxy = wxy + res1inv2*(u[i,j+1,k,0] - u[i,j,k,0])
                    wyy =       res1inv *(u[i,j+1,k,1] - u[i,j,k,1])
                    wyz =       res1inv2*(u[i,j+1,k,2] - u[i,j,k,2])
                else:
                    wyy = <float>0.0
                    wyz = <float>0.0

                if (k < nz-1):
                    wxz = wxz + res2inv2*(u[i,j,k+1,0] - u[i,j,k,0])
                    wyz = wyz + res2inv2*(u[i,j,k+1,1] - u[i,j,k,1])
                    wzz =       res2inv *(u[i,j,k+1,2] - u[i,j,k,2])
                else:
                    wzz = <float>0.0

                sigmaw = sigma*weight[i,j,k]

                wxx = q[i,j,k,0] + sigmaw*wxx
                wxy = q[i,j,k,1] + sigmaw*wxy
                wxz = q[i,j,k,2] + sigmaw*wxz
                wyy = q[i,j,k,3] + sigmaw*wyy
                wyz = q[i,j,k,4] + sigmaw*wyz
                wzz = q[i,j,k,5] + sigmaw*wzz

                qabs = sqrtf(wxx*wxx + wyy*wyy + wzz*wzz + \
                            <float>2.0*(wxy*wxy + wxz*wxz + wyz*wyz))*alphainv
                qabs = <float>1.0/qabs if (qabs > <float>1.0) else <float>1.0

                q[i,j,k,0] = wxx*qabs
                q[i,j,k,1] = wxy*qabs
                q[i,j,k,2] = wxz*qabs
                q[i,j,k,3] = wyy*qabs
                q[i,j,k,4] = wyz*qabs
                q[i,j,k,5] = wzz*qabs



def tgv_update_phi(ndarray[float, ndim=3, mode="c"] phi_dest not None, \
                   ndarray[float, ndim=3, mode="c"] phi not None, \
                   ndarray[float, ndim=3, mode="c"] eta not None, \
                   ndarray[float, ndim=3, mode="c"] mask not None, \
                   ndarray[float, ndim=3, mode="c"] mask0 not None, \
                   float tau, float res0, float res1, float res2):
    """Update phi_dest <- (phi + tau*laplace(mask0*eta))/(1+mask*tau). """

    cdef float taup1inv = <float>1.0/(tau + <float>1.0)

    cdef float res0inv = <float>(1.0)/(res0**2)
    cdef float res1inv = <float>(1.0)/(res1**2)
    cdef float res2inv = <float>(1.0)/(res2**2)

    cdef int nx = phi.shape[0]
    cdef int ny = phi.shape[1]
    cdef int nz = phi.shape[2]

    cdef int i, j, k
    cdef float v0, v1m, v1p, v2m, v2p, v3m, v3p, laplace, fac

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                # compute laplace(mask*eta)
                v0  = mask0[i,j,k]  *eta[i,j,k]
                v1m = mask0[i-1,j,k]*eta[i-1,j,k] if (i > 0)    else v0
                v1p = mask0[i+1,j,k]*eta[i+1,j,k] if (i < nx-1) else v0
                v2m = mask0[i,j-1,k]*eta[i,j-1,k] if (j > 0)    else v0
                v2p = mask0[i,j+1,k]*eta[i,j+1,k] if (j < ny-1) else v0
                v3m = mask0[i,j,k-1]*eta[i,j,k-1] if (k > 0)    else v0
                v3p = mask0[i,j,k+1]*eta[i,j,k+1] if (k < nz-1) else v0

                laplace =   (<float>(-2.0)*v0 + v1m + v1p)*res0inv \
                          + (<float>(-2.0)*v0 + v2m + v2p)*res1inv \
                          + (<float>(-2.0)*v0 + v3m + v3p)*res2inv

                fac = taup1inv if mask[i,j,k] else <float>1.0
                phi_dest[i,j,k] = (phi[i,j,k] + tau*laplace)*fac


def tgv_update_chi(ndarray[float, ndim=3, mode="c"] chi_dest not None, \
                   ndarray[float, ndim=3, mode="c"] chi not None, \
                   ndarray[float, ndim=3, mode="c"] v not None, \
                   ndarray[float, ndim=4, mode="c"] p not None, \
                   ndarray[float, ndim=3, mode="c"] mask0 not None, \
                   float tau, float res0, float res1, float res2):
    """Update chi_dest <- chi + tau*(div(p) - wave(mask*v)). """

    cdef float res0inv = <float>1.0/res0
    cdef float res1inv = <float>1.0/res1
    cdef float res2inv = <float>1.0/res2

    cdef float wres0inv = <float>(1.0/3.0)/(res0**2)
    cdef float wres1inv = <float>(1.0/3.0)/(res1**2)
    cdef float wres2inv = <float>(-2.0/3.0)/(res2**2)

    cdef int nx = chi.shape[0]
    cdef int ny = chi.shape[1]
    cdef int nz = chi.shape[2]

    cdef int i, j, k
    cdef float v0, v1m, v1p, v2m, v2p, v3m, v3p
    cdef float div, wave, m0

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                m0 = mask0[i,j,k]

                # compute div(weight*v)
                div =       m0            *p[i,j,k,0]  *res0inv if i < nx - 1 else <float>0.0
                div = div - mask0[i-1,j,k]*p[i-1,j,k,0]*res0inv if i > 0      else div

                div = div + m0            *p[i,j,k,1]  *res1inv if j < ny - 1 else div
                div = div - mask0[i,j-1,k]*p[i,j-1,k,1]*res1inv if j > 0      else div

                div = div + m0            *p[i,j,k,2]  *res2inv if k < nz - 1 else div
                div = div - mask0[i,j,k-1]*p[i,j,k-1,2]*res2inv if k > 0      else div

                # compute wave(mask*v)
                v0  = m0            *v[i,j,k]
                v1m = mask0[i-1,j,k]*v[i-1,j,k] if (i > 0)    else v0
                v1p = mask0[i+1,j,k]*v[i+1,j,k] if (i < nx-1) else v0
                v2m = mask0[i,j-1,k]*v[i,j-1,k] if (j > 0)    else v0
                v2p = mask0[i,j+1,k]*v[i,j+1,k] if (j < ny-1) else v0
                v3m = mask0[i,j,k-1]*v[i,j,k-1] if (k > 0)    else v0
                v3p = mask0[i,j,k+1]*v[i,j,k+1] if (k < nz-1) else v0

                wave =   (<float>(-2.0)*v0 + v1m + v1p)*wres0inv \
                       + (<float>(-2.0)*v0 + v2m + v2p)*wres1inv \
                       + (<float>(-2.0)*v0 + v3m + v3p)*wres2inv

                chi_dest[i,j,k] = chi[i,j,k] + tau*(div - wave)


def tgv_update_w(ndarray[float, ndim=4, mode="c"] w_dest not None, \
                 ndarray[float, ndim=4, mode="c"] w not None, \
                 ndarray[float, ndim=4, mode="c"] p not None, \
                 ndarray[float, ndim=4, mode="c"] q not None, \
                 ndarray[float, ndim=3, mode="c"] mask not None, \
                 ndarray[float, ndim=3, mode="c"] mask0 not None, \
                 float tau, float res0, float res1, float res2):
    """Update w_dest <- w + tau*(mask*p + div(mask0*q)). """

    cdef float res0inv = <float>1.0/res0
    cdef float res1inv = <float>1.0/res1
    cdef float res2inv = <float>1.0/res2

    cdef int nx = w.shape[0]
    cdef int ny = w.shape[1]
    cdef int nz = w.shape[2]

    cdef int i, j, k
    cdef float q0x, q1y, q2z, q1x, q3y, q4z, q2x, q4y, q5z
    cdef float w0, w1, w2, w3, m0

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                w0 = mask0[i,j,k]
                w1 = mask0[i-1,j,k] if i > 0 else <float>0.0
                w2 = mask0[i,j-1,k] if j > 0 else <float>0.0
                w3 = mask0[i,j,k-1] if k > 0 else <float>0.0

                q0x = w0*q[i,j,k,0]        *res0inv if i < nx - 1 else <float>0.0
                q0x = q0x - w1*q[i-1,j,k,0]*res0inv if i > 0 else q0x
                q1y = w0*q[i,j,k,1]        *res0inv if j < ny - 1 else <float>0.0
                q1y = q1y - w2*q[i,j-1,k,1]*res0inv if j > 0 else q1y
                q2z = w0*q[i,j,k,2]        *res0inv if k < nz - 1 else <float>0.0
                q2z = q2z - w3*q[i,j,k-1,2]*res0inv if k > 0 else q2z

                q1x = w0*q[i,j,k,1]        *res1inv if i < nx - 1 else <float>0.0
                q1x = q1x - w1*q[i-1,j,k,1]*res1inv if i > 0 else q1x
                q3y = w0*q[i,j,k,3]        *res1inv if j < ny - 1 else <float>0.0
                q3y = q3y - w2*q[i,j-1,k,3]*res1inv if j > 0 else q3y
                q4z = w0*q[i,j,k,4]        *res1inv if k < nz - 1 else <float>0.0
                q4z = q4z - w3*q[i,j,k-1,4]*res1inv if k > 0 else q4z

                q2x = w0*q[i,j,k,2]        *res2inv if i < nx - 1 else <float>0.0
                q2x = q2x - w1*q[i-1,j,k,2]*res2inv if i > 0 else q2x
                q4y = w0*q[i,j,k,4]        *res2inv if j < ny - 1 else <float>0.0
                q4y = q4y - w2*q[i,j-1,k,4]*res2inv if j > 0 else q4y
                q5z = w0*q[i,j,k,5]        *res2inv if k < nz - 1 else <float>0.0
                q5z = q5z - w3*q[i,j,k-1,5]*res2inv if k > 0 else q5z

                m0 = mask[i,j,k]

                w_dest[i,j,k,0] = w[i,j,k,0] + tau*(m0*p[i,j,k,0] + q0x + q1y + q2z)
                w_dest[i,j,k,1] = w[i,j,k,1] + tau*(m0*p[i,j,k,1] + q1x + q3y + q4z)
                w_dest[i,j,k,2] = w[i,j,k,2] + tau*(m0*p[i,j,k,2] + q2x + q4y + q5z)

                
def extragradient_update(ndarray[float, ndim=1, mode="c"] u_,
                         ndarray[float, ndim=1, mode="c"] u):
    cdef int nx, i

    nx = u_.shape[0]
    for i in prange(nx, nogil=True):
        u_[i] = <float>2.0*u[i] - u_[i]

###############################################
# TGV-deconvolution routines

def copy_into_submatrix(ndarray[complex64_t, ndim=3, mode="c"] dest,
                        ndarray[float, ndim=3, mode="c"] src,
                        int i0, int j0, int k0):

    cdef int nx = dest.shape[0]
    cdef int ny = dest.shape[1]
    cdef int nz = dest.shape[2]

    cdef int i1 = src.shape[0] + i0
    cdef int j1 = src.shape[1] + j0
    cdef int k1 = src.shape[2] + k0

    cdef int i, j, k

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                dest[i,j,k] = src[i-i0,j-j0,k-k0] if (i >= i0) and (i < i1) and \
                                                     (j >= j0) and (j < j1) and \
                                                     (k >= k0) and (k < k1) else <float>0.0

def extract_from_submatrix(ndarray[float, ndim=3, mode="c"] dest, \
                           ndarray[complex64_t, ndim=3, mode="c"] src, \
                           int i0, int j0, int k0):

    cdef int nx = dest.shape[0]
    cdef int ny = dest.shape[1]
    cdef int nz = dest.shape[2]

    cdef int i, j, k

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                dest[i,j,k] = src[i+i0,j+j0,k+k0].real



def mul_complex_real(ndarray[complex64_t, ndim=3, mode="c"] u,
                     ndarray[float, ndim=3, mode="c"] fac):

    cdef int nx = u.shape[0]
    cdef int ny = u.shape[1]
    cdef int nz = u.shape[2]

    cdef int i, j, k

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                u[i,j,k].real *= fac[i,j,k]
                u[i,j,k].imag *= fac[i,j,k]


def tgv_deconv_update_v(ndarray[float, ndim=3, mode="c"] v not None, \
                        ndarray[float, ndim=3, mode="c"] conv_phi not None, \
                        ndarray[float, ndim=3, mode="c"] conv_phi0 not None, \
                        float sigma, float res0, float res1, float res2):
    """Update v <- (v + sigma*(conv_phi - grad_phi0))/(1 + sigma). """

    cdef float sigmap1inv = <float>1.0/(1.0 + sigma)

    cdef int nx = v.shape[0]
    cdef int ny = v.shape[1]
    cdef int nz = v.shape[2]

    cdef int i, j, k

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                v[i,j,k] = (v[i,j,k] + sigma*(conv_phi[i,j,k]
                                              - conv_phi0[i,j,k]))*sigmap1inv

def tgv_deconv_update_p(ndarray[float, ndim=4, mode="c"] p not None, \
                        ndarray[float, ndim=3, mode="c"] chi not None, \
                        ndarray[float, ndim=4, mode="c"] w not None, \
                        float sigma, float alpha,
                        float res0, float res1, float res2):
    """Update p <- P_{||.||_\infty <= alpha}(p + sigma*(grad(chi) - w)). """

    cdef float alphainv = <float>1.0/alpha
    cdef float res0inv = <float>1.0/res0
    cdef float res1inv = <float>1.0/res1
    cdef float res2inv = <float>1.0/res2

    cdef int nx = p.shape[0]
    cdef int ny = p.shape[1]
    cdef int nz = p.shape[2]

    cdef int i, j, k
    cdef float dxp, dyp, dzp, px, py, pz, pabs, chi0

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                chi0 = chi[i,j,k]
                dxp = (chi[i+1,j,k] - chi0)*res0inv if (i < nx-1) else <float>0.0
                dyp = (chi[i,j+1,k] - chi0)*res1inv if (j < ny-1) else <float>0.0
                dzp = (chi[i,j,k+1] - chi0)*res2inv if (k < nz-1) else <float>0.0

                px = p[i,j,k,0] + sigma*(dxp - w[i,j,k,0])
                py = p[i,j,k,1] + sigma*(dyp - w[i,j,k,1])
                pz = p[i,j,k,2] + sigma*(dzp - w[i,j,k,2])
                pabs = sqrtf(px*px + py*py * pz*pz)*alphainv
                pabs = <float>1.0/pabs if (pabs > <float>1.0) else <float>1.0

                p[i,j,k,0] = px*pabs
                p[i,j,k,1] = py*pabs
                p[i,j,k,2] = pz*pabs


def tgv_deconv_update_q(ndarray[float, ndim=4, mode="c"] q not None, \
                        ndarray[float, ndim=4, mode="c"] w not None, \
                        float sigma, float alpha,
                        float res0, float res1, float res2):
    """Update q <- P_{||.||_\infty <= alpha}(q + sigma*symgrad(w)). """

    cdef float alphainv = <float>1.0/alpha
    cdef float res0inv = <float>1.0/res0
    cdef float res1inv = <float>1.0/res1
    cdef float res2inv = <float>1.0/res2
    cdef float res0inv2 = <float>0.5/res0
    cdef float res1inv2 = <float>0.5/res1
    cdef float res2inv2 = <float>0.5/res2

    cdef int nx = q.shape[0]
    cdef int ny = q.shape[1]
    cdef int nz = q.shape[2]

    cdef int i, j, k
    cdef float wxx, wxy, wxz, wyy, wyz, wzz, qabs
    cdef float w0, w1, w2

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                w0 = w[i,j,k,0]
                w1 = w[i,j,k,1]
                w2 = w[i,j,k,2]

                # compute symgrad(u)
                if (i < nx-1):
                    wxx = res0inv* (w[i+1,j,k,0] - w0)
                    wxy = res0inv2*(w[i+1,j,k,1] - w1)
                    wxz = res0inv2*(w[i+1,j,k,2] - w2)
                else:
                    wxx = <float>0.0
                    wxy = <float>0.0
                    wxz = <float>0.0

                if (j < ny-1):
                    wxy = wxy + res1inv2*(w[i,j+1,k,0] - w0)
                    wyy =       res1inv *(w[i,j+1,k,1] - w1)
                    wyz =       res1inv2*(w[i,j+1,k,2] - w2)
                else:
                    wyy = <float>0.0
                    wyz = <float>0.0

                if (k < nz-1):
                    wxz = wxz + res2inv2*(w[i,j,k+1,0] - w0)
                    wyz = wyz + res2inv2*(w[i,j,k+1,1] - w1)
                    wzz =       res2inv *(w[i,j,k+1,2] - w2)
                else:
                    wzz = <float>0.0

                wxx = q[i,j,k,0] + sigma*wxx
                wxy = q[i,j,k,1] + sigma*wxy
                wxz = q[i,j,k,2] + sigma*wxz
                wyy = q[i,j,k,3] + sigma*wyy
                wyz = q[i,j,k,4] + sigma*wyz
                wzz = q[i,j,k,5] + sigma*wzz

                qabs = sqrtf(wxx*wxx + wyy*wyy + wzz*wzz + \
                            <float>2.0*(wxy*wxy + wxz*wxz + wyz*wyz))*alphainv
                qabs = <float>1.0/qabs if (qabs > <float>1.0) else <float>1.0

                q[i,j,k,0] = wxx*qabs
                q[i,j,k,1] = wxy*qabs
                q[i,j,k,2] = wxz*qabs
                q[i,j,k,3] = wyy*qabs
                q[i,j,k,4] = wyz*qabs
                q[i,j,k,5] = wzz*qabs


def tgv_deconv_update_chi(ndarray[float, ndim=3, mode="c"] chi_dest not None, \
                          ndarray[float, ndim=3, mode="c"] chi not None, \
                          ndarray[float, ndim=3, mode="c"] conv_dwv not None, \
                          ndarray[float, ndim=4, mode="c"] p not None, \
                          float tau, float res0, float res1, float res2):
    """Update chi_dest <- chi + tau*(div(p) - conv_dwv). """

    cdef float res0inv = <float>1.0/res0
    cdef float res1inv = <float>1.0/res1
    cdef float res2inv = <float>1.0/res2

    cdef int nx = chi.shape[0]
    cdef int ny = chi.shape[1]
    cdef int nz = chi.shape[2]

    cdef float div
    cdef int i, j, k

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                # compute div(p)
                div =       p[i,j,k,0]  *res0inv if i < nx - 1 else <float>0.0
                div = div - p[i-1,j,k,0]*res0inv if i > 0      else div

                div = div + p[i,j,k,1]  *res1inv if j < ny - 1 else div
                div = div - p[i,j-1,k,1]*res1inv if j > 0      else div

                div = div + p[i,j,k,2]  *res2inv if k < nz - 1 else div
                div = div - p[i,j,k-1,2]*res2inv if k > 0      else div

                chi_dest[i,j,k] = chi[i,j,k] + tau*(div - conv_dwv[i,j,k])


def tgv_deconv_update_w(ndarray[float, ndim=4, mode="c"] w_dest not None, \
                        ndarray[float, ndim=4, mode="c"] w not None, \
                        ndarray[float, ndim=4, mode="c"] p not None, \
                        ndarray[float, ndim=4, mode="c"] q not None, \
                        float tau, float res0, float res1, float res2):
    """Update w_dest <- w + tau*(p + div(q)). """

    cdef float res0inv = <float>1.0/res0
    cdef float res1inv = <float>1.0/res1
    cdef float res2inv = <float>1.0/res2

    cdef int nx = w.shape[0]
    cdef int ny = w.shape[1]
    cdef int nz = w.shape[2]

    cdef int i, j, k
    cdef float q0x, q1y, q2z, q1x, q3y, q4z, q2x, q4y, q5z

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                q0x = q[i,j,k,0]        *res0inv if i < nx - 1 else <float>0.0
                q0x = q0x - q[i-1,j,k,0]*res0inv if i > 0 else q0x
                q1y = q[i,j,k,1]        *res0inv if j < ny - 1 else <float>0.0
                q1y = q1y - q[i,j-1,k,1]*res0inv if j > 0 else q1y
                q2z = q[i,j,k,2]        *res0inv if k <     nz - 1 else <float>0.0
                q2z = q2z - q[i,j,k-1,2]*res0inv if k > 0 else q2z

                q1x = q[i,j,k,1]        *res1inv if i < nx - 1 else <float>0.0
                q1x = q1x - q[i-1,j,k,1]*res1inv if i > 0 else q1x
                q3y = q[i,j,k,3]        *res1inv if j < ny - 1 else <float>0.0
                q3y = q3y - q[i,j-1,k,3]*res1inv if j > 0 else q3y
                q4z = q[i,j,k,4]        *res1inv if k < nz - 1 else <float>0.0
                q4z = q4z - q[i,j,k-1,4]*res1inv if k > 0 else q4z

                q2x = q[i,j,k,2]        *res2inv if i < nx - 1 else <float>0.0
                q2x = q2x - q[i-1,j,k,2]*res2inv if i > 0 else q2x
                q4y = q[i,j,k,4]        *res2inv if j < ny - 1 else <float>0.0
                q4y = q4y - q[i,j-1,k,4]*res2inv if j > 0 else q4y
                q5z = q[i,j,k,5]        *res2inv if k < nz - 1 else <float>0.0
                q5z = q5z - q[i,j,k-1,5]*res2inv if k > 0 else q5z

                w_dest[i,j,k,0] = w[i,j,k,0] + tau*(p[i,j,k,0] + q0x + q1y + q2z)
                w_dest[i,j,k,1] = w[i,j,k,1] + tau*(p[i,j,k,1] + q1x + q3y + q4z)
                w_dest[i,j,k,2] = w[i,j,k,2] + tau*(p[i,j,k,2] + q2x + q4y + q5z)


###############################################
# Misc routines


def erode_mask(ndarray[float, ndim=3, mode="c"] dest not None, \
               ndarray[float, ndim=3, mode="c"] src not None):
    """Perform dest <- erode(src). """

    cdef int nx = src.shape[0]
    cdef int ny = src.shape[1]
    cdef int nz = src.shape[2]

    cdef int i, j, k
    cdef int v0, v1, v2, v3, v4, v5, v6

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                # compute laplace(mask*eta)
                v0 = <int>(src[i,j,k] != 0) 
                v1 = <int>(src[i-1,j,k] != 0) if (i > 0)    else 1
                v2 = <int>(src[i+1,j,k] != 0) if (i < nx-1) else 1
                v3 = <int>(src[i,j-1,k] != 0) if (j > 0)    else 1
                v4 = <int>(src[i,j+1,k] != 0) if (j < ny-1) else 1
                v5 = <int>(src[i,j,k-1] != 0) if (k > 0)    else 1
                v6 = <int>(src[i,j,k+1] != 0) if (k < nz-1) else 1
                
                dest[i,j,k] = <float>1.0 if (v0+v1+v2+v3+v4+v5+v6 == 7) else <float>0.0

