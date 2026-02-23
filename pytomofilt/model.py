import dataclasses
import os

import numpy as np
import pyshtools as shtools

from . import spline
from . import filter
from . import sh_tools as sh
from . import plotting

_rcmb = 3480.0
_rmoho = 6346.691

@dataclasses.dataclass
class RealLayer:
    radius: float
    lats: list[float]
    lons: list[float]
    vals: list[float]


@dataclasses.dataclass
class RealLayerModel:
    layers: list[RealLayer]

    def plot_radial_slice(self,layer_no):
        """
        Calls the plotting.plot_grid function to plot velocity variations for one the layers.

        Parameters
        ----------
        layer_no : int
            Layer number that is to be plotted.
        
        """
        plotting.plot_grid(self.layers[layer_no].lons,
                           self.layers[layer_no].lats,
                           self.layers[layer_no].vals,
                           self.layers[layer_no].radius,
                           title='Radial slice')


def _default_radii(rmin=_rcmb, rmax=_rmoho):
    # Magic numbers from S20RTS model defs (and everything else)
    KNOT_RADII = np.array([-1.00000, -0.78631, -0.59207, -0.41550, -0.25499, -0.10909,
                            0.02353, 0.14409, 0.25367, 0.35329, 0.44384, 0.52615,
                            0.60097, 0.66899, 0.73081, 0.78701, 0.83810, 0.88454,
                            0.92675, 0.96512, 1.00000])
    knots_r = (rmax - rmin) / 2.0 * KNOT_RADII + (rmin + rmax) / 2.0
    return knots_r

class RTS_Model:

    def __init__(self, lmax, rmin=_rcmb, rmax=_rmoho, knots=None):
        """
        Create an instance of the RTS model object

        Parameters
        ----------
        lmax: int
            The maximum spherical harmonic degree parameterised in this
            model. 
        rmin: float
            The minimum radii of the RTS_model. If not specified, this
            defaults to the radius of the CMB.
        rmax: float
            The maximum radii of the RTS_model. If not specified, this
            defaults to the radius of the Moho.
        knots: array_like (n)
            The list of n radii points to calculate the model coefficients.
            If None, default knot radii from the RTS models are used, which
            are scaled to the maximum as rmax and minimum as rmin.
        """
        self.lmax = lmax
        self.rmax = rmax
        self.rmin = rmin
        if knots is None:
            # Declare spline knots from default
            self.knots_r = _default_radii(rmin=_rcmb, rmax=_rmoho)
        else:
            self.knots_r = np.asarray(knots) # Should cast into array
        # Build splines
        self.knot_splines = spline.calculate_splines(self.knots_r)
        self.filter_obj = None
        # storage for model - radius, real/imag, degree, order:
        self.coefs = np.zeros((len(self.knots_r), 2, lmax+1, lmax+1))    


    @classmethod
    def from_file(cls, filename, rmin=_rcmb, rmax=_rmoho, knots=None):
        """
        Inputs coefficients from a .sph file. The coefficients should be evaluated
        at the knot points specified. Coefficients are saved under the coefs 
        attribute of this class.

        Parameters
        ----------
        filename : str
            Filename of the .sph file to be read.
        rmin: float
            The minimum radii of the RTS_model. If not specified, this
            defaults to the radius of the CMB.
        rmax: float
            The maximum radii of the RTS_model. If not specified, this
            defaults to the radius of the Moho.
        knots: array_like (n)
            The list of n radii points to calculate the model coefficients.
            If None, default knot radii from the RTS models are used, which
            are scaled to the maximum as rmax and minimum as rmin.
        """
        f = open(filename, 'r')

        header = next(f).split()
        lmax = int(header[0])
        if knots is None:
            knots = _default_radii(rmin=rmin, rmax=rmax)    
        coefs = np.zeros((len(knots), 2, lmax+1, lmax+1))    

        # The format of the rest of the file is a little bit odd. Coefficients 
        # for each radius are listed in blocks and within these blocks coefficients
        # for each l are listed in rows. However, the maximum number of coefficients
        # all of the coefficients for a r,l combination. Things could go wrong if we
        # per line is 11, so for l > 6 we have to read multiple lines before we have
        # end up with too many coefficients for the r,l combination we expect and we
        # trap that at the end of the loop. Otherwise the file could be too short, 
        # in which case we won't have incremented ri enough.
        dataline = []
        ri = 0
        li = 0
        for line in f:  # Reads from second line (header has already been read)
            dataline.extend(line.split())
            
            if len(dataline) == (li * 2) + 1:
                # We have the right number of coefficients for this l
                # so we can process it into the numpy array
                mi = 0
                for m, coef in enumerate(dataline):
                    assert ri < len(knots), f"Too many lines when incrementing ri! ri={ri}"
                    if m == 0:
                        coefs[ri,0,li,mi] = float(coef)
                        mi = mi + 1
                    elif m%2 == 1:
                        # Odd number in list, real coef
                        coefs[ri,0,li,mi] = float(coef)
                        # don't increment mi!
                    else:
                        # even number in list, imag coef
                        coefs[ri,1,li,mi] = float(coef)
                        mi = mi + 1

                li = li + 1

                if li > lmax:
                    # We have read all the coefficients for this radial
                    # layer. Reset L and increment R. 
                    li = 0
                    ri = ri + 1
                dataline = [] # Start adding to a new set of coefficients

            assert len(dataline) < (li * 2) + 1, f"Too much data, li={li}, data={dataline}"

        f.close()
        assert ri == len(knots), f"End of file without seeing all expected radii, ri={ri}, knots={knots}"
        assert li == 0, "End of file without resetting L!"

        # Create new RTS model instance and return
        RTS = cls(lmax, rmin, rmax, knots)
        RTS.coefs = coefs
        return RTS
    

    @classmethod
    def from_terra(self, terra_file, rmin=_rcmb, rmax=_rmoho, knots=None):
        pass

    
    @classmethod
    def from_directory(cls, directory, lmax, rmin=_rcmb, rmax=_rmoho, knots=None):
        """
        Inputs values from files stored in a directory, and automatically reparameterises the data (see 
        reparam for more information). Coefficients are saved under the coefs attribute of this class.
        
        There should be a "depth_layers.dat" file that contains the list of depth layers. Each data file 
        should have an '.dat' extension, numbered in increasing order of depth. The data files be 
        organised into 3 columns, where the columns from left to right are longitude, latitude and the 
        value at the (lon, lat) point respectively (see below):

        Parameters
        ----------
        directory : str
            The directory that contains the files to be read.
        lmax: int
            The maximum spherical harmonic degree parameterised in this
            model. 
        rmin: float
            The minimum radii of the RTS_model. If not specified, this
            defaults to the radius of the CMB.
        rmax: float
            The maximum radii of the RTS_model. If not specified, this
            defaults to the radius of the Moho.
        knots: array_like (n)
            The list of n radii points to calculate the model coefficients.
            If None, default knot radii from the RTS models are used, which
            are scaled to the maximum as rmax and minimum as rmin.
            
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
        depth = np.loadtxt(os.path.join(directory, 'depth_layers.dat')) # list of increasing depth
        radii = 6371 - depth[::-1]  # Change depth into list of radii (deepest to shallowest)

        # Files are listed from shallowest to deepest, we need to reverse the order because we
        # want to save layers from deepest to shallowest
        data_files = sorted([f for f in os.listdir(directory) if f.endswith('.dat') and f != 'depth_layers.dat'],
                            reverse=True)
        assert len(data_files) == len(depth), "Number of data files does not match number of depths"

        for i, filename in enumerate(data_files):
            data = np.loadtxt(os.path.join(directory, filename))
            layers.append(RealLayer(radii[i], lons = data[:,0],
                                    lats = data[:,1], vals = data[:,2]))
        
        # Write to class
        RTS = cls(lmax, rmin, rmax, knots)
        RTS.reparam(RealLayerModel(layers))
        return RTS
    

    def reparam(self, layer_model):
        """
        Reparameterises the model laterally into spherical harmonics and vertically in a three-point clamped
        cubic spline functions. The output is the coefficients evaluated at the spline knots, and is saved
        in the class s_model attribute.
        
        Parameters
        ----------
        layer_model : RealLayerModel class object
            A list of RealLayer objects, where each RealLayer object represents a list of values at a list
            of longitude and latitude points at a certain radii.
        """
        assert isinstance(layer_model, RealLayerModel), "layer model must be an instance of RealLayerModel"
        
        # Reparameterise file laterally
        layer_radii = [l.radius for l in layer_model.layers]
        sh_coefs = np.zeros((len(layer_radii),) + self.coefs.shape[1:])

        for i, layer in enumerate(layer_model.layers):
            # SHExpandLSQ only works for real coefficients
            cilm, chi2 = shtools.expand.SHExpandLSQ(layer.vals, layer.lats, layer.lons, 
                                                    lmax = self.lmax, norm = 4, csphase = 1)

            # Convert coefficients to RTS format
            sh_coefs[i] = sh.sh_to_rts(cilm)

        # Calculate coefficients at spline knots
        self.coefs = spline.cubic_spline(sh_coefs, layer_radii,
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
        filtered_model: RTS_Model class object
            a copy of model, having been filtered using the
            resolution operator. This is the same resolution of the model
            containing the filter.
        """
        assert self.filter_obj is not None, "You must use filter_from_file() to add the filter"
        assert isinstance(model, RTS_Model), "Input model must be an instance of RTS_Model to be filtered" 
        x = model.as_vector()
        x = self.filter_obj.apply_filter(x)
        output_model = RTS_Model(self.lmax, self.rmin, self.rmax, self.knots_r)
        output_model.from_vector(x)
        return output_model
        

    def as_vector(self):
        """
        Return 1D vector representing the model object
        
        This is the order needed for filtering
        (and matches the elements in the files)

        Returns
        -------
        vector: array_like
            a representation of the model coefficients as a 1D vector
        """
        lmx = self.lmax
        ndp = len(self.knots_r)
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
        
        Parameters
        -------
        vector: array_like
            a representation of the model coefficients as a 1D vector
        """
        
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
        return None

    
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
    

    def plot_radial_slice(self, spline_point_np):
        """
        Calls the plotting.plot_shcoef function to plot velocity variations coefficients at one of
        the spline points.

        Parameters
        ----------
        spline_point_np : int
            Index of the spline point at which the coefficients will be plotted.
        
        """
        plotting.plot_shcoefs(self.coefs[spline_point_np],
                              r=self.knots_r[spline_point_np],
                              title='Radial slice')


    def _write_sph_body_lines(self, fhandle):
        """
        Write the body of an SPH file to fhandle

        This exists as a seperate method for use in writing SPT files

        Parameters
        ----------
        fhandle : handle
            Handle of the file to be written.
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