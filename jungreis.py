#!/usr/bin/env python
"""
    Python code by Matthias Meschede 2014
    http://pythology.blogspot.fr/2014/08/parametrized-mandelbrot-set-boundary-in.html
    https://en.wikibooks.org/wiki/Fractals/Iterations_in_the_complex_plane/Mandelbrot_set/boundary
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from ctypes import *

j_dll = cdll.LoadLibrary('jungreis')

j_dll.reserveCache.argtypes = [c_size_t, c_size_t]
j_dll.betaF.argtypes = [c_longlong, c_longlong]
j_dll.betaF.restype = c_double



nstore = 5000  #cachesize should be more or less as high as the coefficients
betaF_cachedata = np.zeros( (nstore,nstore))
betaF_cachemask = np.zeros( (nstore,nstore),dtype=bool)
def betaF(n,m):
    """
    This function was translated to python from
    http://fraktal.republika.pl/mset_jungreis.html
    It computes the Laurent series coefficients of the jungreis function
    that can then be used to map the unit circle to the Mandelbrot
    set boundary. The mapping of the unit circle can also
    be seen as a Fourier transform. 
    I added a very simple global caching array to speed it up
    """
    global betaF_cachedata,betaF_cachemask

    nnn=2**(n+1)-1
    if betaF_cachemask[n,m]:
        return betaF_cachedata[n,m]
    elif m==0:
        return 1.0
    elif ((n>0) and (m < nnn)):
        return 0.0
    else: 
        value = 0.
        for k in range(nnn,m-nnn+1):
            value += betaF(n,k)*betaF(n,m-k)
        value = (betaF(n+1,m) - value - betaF(0,m-nnn))/2.0 
        betaF_cachedata[n,m] = value
        betaF_cachemask[n,m] = True
        return value

use_c = True

def do_stuff(ncoeffs, npoints):
    t0 = time.time()


    #compute coefficients (reduce ncoeffs to make it faster)
    coeffs = np.zeros( (ncoeffs) )
    for m in range(ncoeffs):
        if m%100==0: print('%d/%d'%(m,ncoeffs))
        if use_c:
            coeffs[m] = j_dll.betaF(0, m + 1)
        else:
            coeffs[m] = betaF(0,m+1)

    t1 = time.time()
    print('Elapsed time: {}'.format(t1 - t0))

    #map the unit circle  (cos(nt),sin(nt)) to the boundary
    points = np.linspace(0,2*np.pi,npoints)
    xs     = np.zeros(npoints)
    ys     = np.zeros(npoints)
    xs = np.cos(points)
    ys = -np.sin(points)
    for ic,coeff in enumerate(coeffs):
        xs += coeff*np.cos(ic*points)
        ys += coeff*np.sin(ic*points)

    return (xs, ys)


def main():


    # Create the cache for the C implementation
    # j_dll.reserveCache(12, 4097)


    #plot the function
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')


    ncoeffs_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    ncoeffs_list = [16384]

    for ncoeffs in ncoeffs_list:
        (xs, ys) = do_stuff(ncoeffs, 20000)
        ax.plot(xs,ys, linewidth=0.5)

    j_dll.printCacheUsage()

    plt.show()

if __name__ == "__main__":
    main()