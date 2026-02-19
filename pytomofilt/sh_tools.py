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

    # RTS format multiples non-zero order (m) components by 2 and 
    rts_coefs[:,1:,1:] *= 2

    return rts_coefs