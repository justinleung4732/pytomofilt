import dataclasses

import numpy as np
import pyshtools as shtools

from . import spline
from . import filter

KNOT_RADII = [1.00000, 0.96512, 0.92675, 0.88454, 0.83810, 0.78701,
              0.73081, 0.66899, 0.60097, 0.52615, 0.44384, 0.35329,
              0.25367, 0.14409, 0.02353, -0.10909, -0.25499, -0.41550,
              -0.59207, -0.78631, -1.00000]


@dataclasses.dataclass
class RealLayer:
    depth: float
    lats: list[float]
    lons: list[float]
    vals: list[float]


@dataclasses.dataclass
class RealLayerModel:
    layers: list[RealLayer]


class RTS_Model:

    def __init__(self):
        self.rcmb = 3480.0
        self.rmoho = 6346.691

        # Declare spline knots and calculate spline
        self.knots_r = (self.rmoho - self.rcmb) / 2.0 * np.array(KNOT_RADII) + \
                       (self.rmoho + self.rcmb) / 2.0
        self.knot_splines = spline.calculate_splines(self.knots_r)
        self.filter_obj = None


    def from_file(self, filename):
        pass


    def from_layers(self, filename, header = None, sh_deg = 40):
        # Read file
        f = open(filename, 'r')

        vs_points = np.zeros((5,7000,7000))
        lat = points
        lon = points

        # Reparameterise file laterally
        sh_coefs = np.zeros((len(points), 2, sh_deg, sh_deg))

        for i, p in vs_points:
            real_coefs = shtools.SHCoeffs.from_zeros(lmax, kind='real', 
                                        normalization='ortho', csphase=1)

            cilm, chi2 = shtools.expand.SHExpandLSQ(p, lat, lon, lmax = sh_deg, norm = 4, csphase = -1) # SHExpandLSQ only works for real coefficients
            real_coefs.coeffs = cilm
            cilm[1] = -cilm[1]  # Minus sign for imaginary part because real coefficients for 
                                # sin store negative m degrees, where sin(-m*phi) = -sin(m*phi)

            complex_coefs = coefs.convert(normalization='ortho', csphase=-1, kind='complex', 
                                    check=False).to_array()
                                
            # SHTOOLS stores the two arrays in axis 0 as positive m and negative m shells. We 
            # want to match the RTS format of storing real part in the first array and
            # imaginary aprt in the second array of axis 0
            real = complex_coefs[0].real
            imag = complex_coefs[0].imag

            rts_coefs = np.array([real, imag])

            # RTS format multiples non-zero order (m) components by 2 and 
            # divides all coefficients by 100
            rts_coefs /= 100
            rts_coefs[:,1:,1:] *= 2

            sh_coefs[i] = rts_coefs

        # Calculate coefficients at spline knots
        self.s_model = spline.cubic_spline(sh_coefs, depths, self.knot_splines) # Coefficients at the spline knot 


    def filter_from_file(self, evec_file, wght_file, damping, verbose=False):
        """
        Adds a resolution operator (filter) from Fortran formatted files
        
        Parameters
        ----------
        evec_file: file name of the Fortran unformatted file containing the
                   eigenvectors (and values) used to build the resolution
                   operator.
        wght_file: file name of the Fortran unformatted file containing the
                   weights.
        damping: damping parameter used in the inversion.
        verbose: optional, adds extra output for all filter operations
        Setting the optional verbose argument to True results in the more
        output being created. This is useful for debugging and comparing the
        run to the Fortran equivalent

        Returns
        -------
        None
        """
        self.filter_obj = filter.Filter(evec_file, wght_file, damping, verbose)


    def filter(self, model):
        """
        Apply this model's resolution filter to another model instance
        
        Both models must be parameterized the same way (i.e. spherical 
        harmonics at the knots of cubic splines). Typically the model
        with a resolution operator will be from a tomographic inversion
        and the model to be operated on (the argument to this method)
        will be from some other high resolution source - e.g. a geodynamic
        simulation. Note that the filter must be loaded before calling this
        method.
        
        Parameters
        ----------
        model: another instance of the RTS_Model class
        
        Returns
        -------
        filtered_model: a copy of model, having been filtered using the
            resolution operator. This is the same resolution of the model
            containing the filter.
        """
        assert self.filter_obj is not None, "You must use filter_from_file() to add the filter"
        assert isinstance(model, RTS_Model), "Input model must be an instance of RTS_Model to be filtered" 
        x = model.as_vector()
        x = model.filter.apply_filter(x)
        output_model = RTS_Model() # FIXME: need to pass the dimensions of self here (which need to be in our __init__)
        output_model.from_vector(x)
        return output_model
        

    def as_vector(self):
        """
        Return 1D vector representing the model object
        
        This is the order needed for filtering
        (and matches the elements in the files)
        """
        # FIXME: size paraemeters. We should be able
        # to get these from the shell definitions...
        lmx = 12 # This is just lmax
        ndp = 21 # What is this? number of depths?
        # ndp comes from (24 - 4 + 1) + (24 - 4 + 1) = 42
        # where 24 and 4 are numbers in the .spt header...
        # we only need half for S or P
        natd = (lmx + 1)**2 
        lenatd = natd * ndp
    
        vector = np.empty((lenatd))
        counter = 0
        # FIXME: Maybe numba this loop?
        for ri in range(len(self.knots_r)):
            for li in range(self.degree + 1):
                for mi in range(li+1):
                    vector[counter] = self.coefs[ri,0,li,mi]
                    counter = counter + 1
                    if mi != 0:
                        vector[counter] = self.coefs[ri,1,li,mi]
                        counter = counter + 1

        assert counter == lenatd, "missing vals" 
        return vector
    

    def from_vector(self, vector):
        """
        Fill out the model coefficients from a 1D vector
        """
        lmx = 12 # This is just lmax
        ndp = 21 # What is this? number of depths?
        # ndp comes from (24 - 4 + 1) + (24 - 4 + 1) = 42
        # where 24 and 4 are numbers in the .spt header...
        # we only need half for S or P
        natd = (lmx + 1)**2 
        lenatd = natd * ndp
        
        counter = 0
        # FIXME: Maybe numba this loop?
        for ri in range(len(self.knots_r)):
            for li in range(self.degree + 1):
                for mi in range(li+1):
                    self.coefs[ri,0,li,mi] = vector[counter]
                    counter = counter + 1
                    if mi != 0:
                        self.coefs[ri,1,li,mi] = vector[counter]
                        counter = counter + 1


    def correlate(self, model):
        pass

    
    def write(self, filename):
        pass