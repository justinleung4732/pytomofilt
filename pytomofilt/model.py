import dataclasses
import os

import numpy as np
import pyshtools as shtools

from . import spline

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


    def from_file(self, filename):
        pass
    

    def from_terra(self, terra_file):
        pass

    
    def from_directory(self, directory, sh_deg = 40):
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
        
        reparam(RealLayerModel(layers), sh_deg = sh_deg)
    

    def reparam(self, layer_model, sh_deg = 40):
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
        sh_coefs = np.zeros((len(layer_model.layers), 2, sh_deg, sh_deg))

        for i, layer in layer_model.layers:
            real_coefs = shtools.SHCoeffs.from_zeros(lmax, kind='real', 
                                        normalization='ortho', csphase=1)

            # SHExpandLSQ only works for real coefficients
            cilm, chi2 = shtools.expand.SHExpandLSQ(layer.vals, layer.lats, layer.lons, 
                                                    lmax = sh_deg, norm = 4, csphase = -1)
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
        self.s_model = spline.cubic_spline(sh_coefs, depths, self.knot_splines) # Coefficients at the spline knot 


    def filter_from_file(self, filename):
        pass


    def filter(self, model):
        pass
    

    def correlate(self, model):
        pass

    
    def write(self, filename):
        pass