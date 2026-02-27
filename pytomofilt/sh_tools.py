from functools import partial
import multiprocessing as mp
import os.path

import numpy as np
import pyshtools as shtools


def rts_to_sh(rts_coefs):
    """
    Convert between the SH format from the 'sph' file to pyshtools format

    In '.sph' files used for the SP12RTS tomography model (for example) the 
    spherical harmonic coefficents are complex, fully normalised, with the CS
    phase. For most of our use we use real fully normalised real coefficents 
    without the CS phase. This function does the conversion from 'sph'/'rts' to 
    our normal format.
    """

    inp_shape = np.shape(rts_coefs)
    assert inp_shape[0] == 2, 'rts_coefs must have real and imag parts on 1st dim'
    lmax = inp_shape[1] - 1
    assert inp_shape[2] == lmax + 1, 'must have all ms'

    coefs= shtools.SHCoeffs.from_zeros(lmax, kind='complex',
                                       normalization='ortho', csphase=-1)

    # We don't need to set -m coeffs. minus sign for imaginary part
    # because real coefficients for sin store negative m degrees,
    # where sin(-m*phi) = -sin(m*phi)
    coefs.coeffs[0] = rts_coefs[0] - rts_coefs[1] * 1j

    sh_coefs = coefs.convert(normalization='ortho', csphase=1, kind='real',
                             check=False).to_array()

     # RTS format multiples non-zero order (m) components by 2
    sh_coefs[:,1:,1:] /= 2

    return sh_coefs


def sh_to_rts(sh_coefs):
    """
    Convert between pyshtools format and that from from the 'sph' files

    In '.sph' files used for the SP12RTS tomography model (for example) the 
    spherical harmonic coefficents are complex, fully normalised, with the CS
    phase. For moust of our use we use real fully normalised real coefficents 
    without the CS phase. This function does the conversion from our nornal 
    format to the 'sph'/'rts' format.
    """

    inp_shape = np.shape(sh_coefs)
    assert inp_shape[0] == 2, 'sh_coefs must have real and imag parts on 1st dim'
    lmax = inp_shape[1] - 1
    assert inp_shape[2] == lmax + 1, 'must have all ms'

    real_coefs = shtools.SHCoeffs.from_zeros(lmax, kind='real', 
                                             normalization='ortho', csphase=1)
    
    real_coefs.coeffs = sh_coefs
    sh_coefs[1] = -sh_coefs[1]  # Minus sign for imaginary part because real coefficients for
                        # sin store negative m degrees, where sin(-m*phi) = -sin(m*phi)

    complex_coefs = real_coefs.convert(normalization='ortho', csphase=-1, kind='complex',
                              check=False).to_array()

    # SHTOOLS stores the two arrays in axis 0 as positive m and negative m shells. We
    # want to match the RTS format of storing real part in the first array and
    # imaginary aprt in the second array of axis 0
    real = complex_coefs[0].real
    imag = complex_coefs[0].imag

    rts_coefs = np.array([real, imag])

    # RTS format multiples non-zero order (m) components by 2
    rts_coefs[:,1:,1:] *= 2

    return rts_coefs


def sh_expand(directory, filenames, lmax):
    """
    Expands grids at a list of depths into spherical harmonics.

    Each data file should have an '.dat' extension, numbered in increasing order of depth. The data
    files be organised into 3 columns, where the columns from left to right are longitude,latitude
    and the value at the (lon, lat) point respectively.

    Parameters
    ----------
    directory : str
        The directory that contains the files to be read.
    filenames : lis
        List of filenames in the directory to be read.
    lmax: int
        The maximum spherical harmonic degree parameterised in this
        model. 

    Returns
    -------
    sh_coefs : array_like
        Spherical harmonic coefficients expanded for each of the files.
    """

    # Create partial function to fix lmax and directory for inner function
    _sh_expand_partial = partial(_sh_expand_inner, lmax=lmax, directory=directory)

    # Determine if sh expansion should be conducted in parallel
    if len(filenames) >= 3*os.cpu_count():
        parallel = True
    else:
        parallel = False

    if parallel:
        with mp.Pool() as pool: # FIXME: this creates a large memory overhead, especially with many
                                # processes or large number of data points in the file. e.g. a grid
                                # of 739840 data points evaluated at lmax=12 will require inverting
                                # a matrix of size 739840 x (12+1)^2 x 8 ≈ 1 GB. Perhaps limit
                                # number of processes/workers by (total RAM) / (memory per task)
            sh_coefs = pool.map(_sh_expand_partial, filenames, chunksize=8)
    else:   # Run in series if calculation cost is low
        sh_coefs = np.zeros((len(filenames),2,lmax+1,lmax+1))

        for i,filename in enumerate(filenames):
            sh_coefs[i] = _sh_expand_partial(filename)

    return np.asarray(sh_coefs)


def _sh_expand_inner(filename, directory, lmax=40):
    """
    Inner function to expand spherical harmonic coefficients. Used by sh_conversion to be
    iterated over every file (depth), either sequentially in parallel.

    Parameters
    ----------
    filename : str
        Filename in the directory to be read.
    directory : str
        The directory that contains the files to be read.
    lmax: int
        The maximum spherical harmonic degree parameterised in this
        model. 

    Returns
    -------
    _type_
        _description_
    """
    data = np.loadtxt(os.path.join(directory, filename))
    vals = data[:,2]
    lats = data[:,1]
    lons = data[:,0]

    coefs,_ = shtools.expand.SHExpandLSQ(vals, lats, lons, lmax=lmax, norm=4, csphase=1)

    return coefs