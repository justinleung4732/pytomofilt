import dataclasses
import os

import numpy as np
import pyshtools as shtools

from . import spline
from . import filter

# Magic numbers from S20RTS model defs (and everything else)
_KNOT_RADII = [1.00000, 0.96512, 0.92675, 0.88454, 0.83810, 0.78701,
              0.73081, 0.66899, 0.60097, 0.52615, 0.44384, 0.35329,
              0.25367, 0.14409, 0.02353, -0.10909, -0.25499, -0.41550,
              -0.59207, -0.78631, -1.00000]
_rcmb = 3480.0
_rmoho = 6346.691

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

    def __init__(self, lmax, rmin=_rcmb, rmax=_rmoho, knots=None):
        self.rmax = rmax
        self.rmin = rmin
        if knots is None:
            # Declare spline knots from default
            self.knots_r = (self.rmin - self.rmax) / 2.0 * np.array(_KNOT_RADII) + \
                           (self.rmin + self.rmax) / 2.0
        else:
            self.knots_r = np.asarray(knots) # Should cast into array
        # Build splines
        self.knot_splines = spline.calculate_splines(self.knots_r)
        self.filter_obj = None
        # storage for model - radius, real/imag, degree, order:
        self.coefs = np.empty((len(self.knots_r), 2, lmax+1, lmax+1))


    def from_file(self, filename):
        """
        Inputs values from .sph files. 

        Parameters
        ----------
        filename : str
            Filename of the .sph file to be read.
        
        """
        f = open(filename, 'r')
        header = None
        dataline = []
        ri = 0
        li = 0
        for line in f:
            if header is None:
                header = line.split()
                assert int(header[0]) == self.lmax, "wrong SH degree"
            else:
                dataline.extend(line.split())
                if len(dataline) == (li * 2) + 1:
                    # We have all the data, process the line
                    assert ri <= len(self.knots_r), "Too many lines!"

                    mi = 0
                    for m, coef in enumerate(dataline):
                        if m == 0:
                            self.coefs[ri,0,li,mi] = float(coef)
                            mi = mi + 1
                        elif m%2 == 1:
                            # Odd number in list, real coef
                            self.coefs[ri,0,li,mi] = float(coef)
                            # don't increment mi!
                        else:
                            # even number in list, imag coef
                            self.coefs[ri,1,li,mi] = float(coef)
                            mi = mi + 1

                    li = li + 1
                    if li > self.lmax:
                        li = 0
                        ri = ri + 1 
                    dataline = []

                assert len(dataline) < (li * 2) + 1, "Too much data"

        f.close()
    

    def from_terra(self, terra_file):
        pass

    
    def from_directory(self, directory):
        """
        Inputs values from files stored in a directory, and automatically reparameterises the data (see 
        reparam for more information). 
        
        There should be a "depth_layers.dat" file that contains the list of depth layers. Each data file 
        should have an '.dat' extension, numbered in increasing order of depth. The data files be 
        organised into 3 columns, where the columns from left to right are longitude, latitude and the 
        value at the (lon, lat) point respectively (see below):

        Parameters
        ----------
        directory : str
            The directory that contains the files to be read.
        
        sh_deg: int
            The maximum spherical degree calculated for the reparameterisation. The default is set to 40.

        
        File format example
        ----------
        0.0 90.0 10.0
        0.0 0.0 20.0
        90.0 0.0 20.0
        180.0 0.0 20.0
        270.0 0.0 20.0
        0.0 -90.0 10.0
        
        """

        layers = []
        depth = np.loadtxt(os.path.join(directory, 'depth_layers.dat'))
        data_files = sorted([f for f in os.listdir(directory) if f.endswith('.dat') and f != 'depth_layers.dat'])

        for i, filename in enumerate(data_files):
            data = np.loadtxt(os.path.join(directory, filename))
            layers.append(RealLayer(depth[i], lons = data[:,0],
                                    lats = data[:,1], vals = data[:,2]))
        
        self.reparam(RealLayerModel(layers))
    

    def reparam(self, layer_model):
        """
        Reparameterises the model laterally into spherical harmonics and vertically in a three-point clamped
        cubic spline functions. The output is the coefficients evaluated at the spline knots, and is saved
        in the class s_model attribute.
        
        Parameters
        ----------
        layer_model : RealLayerModel class object
            A list of RealLayer objects, where each RealLayer object represents a list of values at a list
            of longitude and latitude points at a certain depth.
        
        sh_deg: int
            The maximum spherical degree calculated for the reparameterisation. The default is set to 40.
        """
        assert isinstance(layer_model, RealLayerModel), "layer model must be an instance of RealLayerModel"
        # Reparameterise file laterally
        sh_coefs = np.zeros_like(self.coefs)

        for i, layer in layer_model.layers:
            real_coefs = shtools.SHCoeffs.from_zeros(self.lmax, kind='real', 
                                                     normalization='ortho', csphase=1)

            # SHExpandLSQ only works for real coefficients
            cilm, chi2 = shtools.expand.SHExpandLSQ(layer.vals, layer.lats, layer.lons, 
                                                    lmax = self.lmax, norm = 4, csphase = -1)
            real_coefs.coeffs = cilm
            cilm[1] = -cilm[1]  # Minus sign for imaginary part because real coefficients for 
                                # sin store negative m degrees, where sin(-m*phi) = -sin(m*phi)

            complex_coefs = real_coefs.convert(normalization='ortho', csphase=-1, kind='complex', 
                                    check=False).to_array()
                                
            # SHTOOLS stores the two arrays in axis 0 as positive m and negative m shells. We 
            # want to match the RTS format of storing real part in the first array and
            # imaginary part in the second array of axis 0
            real = complex_coefs[0].real
            imag = complex_coefs[0].imag

            rts_coefs = np.array([real, imag])

            # RTS format multiples non-zero order (m) components by 2 and 
            # divides all coefficients by 100
            rts_coefs /= 100
            rts_coefs[:,1:,1:] *= 2

            sh_coefs[i] = rts_coefs

        # Calculate coefficients at spline knots
        self.coefs = spline.cubic_spline(sh_coefs, [l.depth for l in layer_model.layers],
                                         self.knot_splines)


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
        x = self.filter.apply_filter(x)
        output_model = RTS_Model(self.lmax, self.rmin, self.rmax, self.knots_r)
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
            for li in range(self.lmax + 1):
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
            for li in range(self.lmax + 1):
                for mi in range(li+1):
                    self.coefs[ri,0,li,mi] = vector[counter]
                    counter = counter + 1
                    if mi != 0:
                        self.coefs[ri,1,li,mi] = vector[counter]
                        counter = counter + 1


    def correlate(self, model):
        pass

    
    def write(self, filename):
        """
        Writes the coefficients of the model as a .sph file. 

        Parameters
        ----------
        filename : str
            Filename of the .sph file to be saved.
        
        """
        f = open(filename, 'w')

        # Write header - see line 27 of wsphhead.f
        f.write("           {:4} {}{:4} {} \n".format(self.lmax,
                "1"*(self.lmax+1), 24, "000111111111111111111111"))
        # Write the body
        self._write_sph_body_lines(f)
        f.close()


    def _write_sph_body_lines(self, fhandle):
        """
        Write the body of an SPH file to fhandle

        This exists as a seperate method for use in writing SPT files
        """
        # Write coefs (knot by knot, and l by l)
        for ri in range(len(self.knots_r)):
            for li in range(self.lmax+1):
                line = ""
                items = 0
                for mi in range(li+1):
                    line = line + " {: 10.4E}".format(self.coefs[ri,0,li,mi])
                    items = items + 1
                    if items == 11: 
                        line = line + "\n"
                        items = 0
                    if mi != 0:
                        # NB: the examples files are all "0.nnnnE-nn" for the 
                        #     values. This gives "n.nnn0E-nn" for the output 
                        #     (with a larger exponent). This is probably OK.
                        #     We could try to get bit for bit equivelance, but
                        #     I struggled with the fortranformat module.
                        line = line + " {: 10.4E}".format(
                                                    self.coefs[ri,1,li,mi])
                        items = items + 1
                        if items == 11: 
                            line = line + "\n"
                            items = 0
                if items != 0:
                    line = line + "\n"
                fhandle.write(line)