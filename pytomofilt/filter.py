import os

import numpy as np
import scipy.io as spio
import numba

class Filter(object):
    """
    Object to apply tomographic resolution operator to seismic model

    The creation of tomographic models of seismic wave velocities 
    such as S20RTS, S40RTS and SP12RTS can also provide a resolution
    operator that maps a true earth model into the tomography
    model, and fully describes the spatially heterogeneous resolution
    of the tomography. This object allows a resolution operator from
    the "RTS" family of models to be applied to a model of seismic
    velocities, such that the velocities can be fairly compared with
    the tomography. Software for doing this also exists in Fortran.
    The main difference between this implementation and the Fortran
    version is that this version holds the operator in memory allowing
    multiple models to be "filtered". The Fortran implementation reads
    applies one "line" of the operator at a time reading the line from
    disk. This means that this implementation uses much more memory 
    but is much faster if multiple models have to be filtered. On the
    other hand, the Fortran implementation can be used on low memory 
    machines. Within LEMA where we make use of very large numbers of
    low resolution models holding the model in memory is clearly the
    correct choice (indeed, it is ~100 times faster to do it this way).

    To use the filter first create an instance of this class while
    providing the file names of the files containing the resolution
    operator (this is quite slow, see the documentation of the __init__
    method). Then convert your model of seismic velocities into an
    RTSJointShell object before calling the RTSJointShell.filter_sp_model
    method with the instance provided in the "filter" keyword. The 
    RTSJointShell object is then updated to represent the "filtered"
    seismic velocities.
    """
    
    def __init__(self, evec_file, wght_file, damping, verbose=False):
        """
        Create an instance of the resolution operator object

        This needs two files and a damping parameter which are passed
        in as the following arguments:

        evec_file: file name of the Fortran unformatted file containing the
                   eigenvectors (and values) used to build the resolution
                   operator.
        wght_file: file name of the Fortran unformatted file containing the
                   weights;
        damping: damping parameter used in the inversion

        Setting the optional verbose argument to True results in the more
        output being created. This is useful for debugging and comparing the
        run to the Fortran equivalent
        """
        self.evcfl = evec_file
        self.wgtsfl = wght_file
        self.damp = damping
        self.verbose = verbose
        
        # Read weights and eigenvector files
        (self.eigvals, self.eigvecs, self.icrust, 
         self.ismth, self.natd, self.ndep) = self.read_eigenvec_file(self.evcfl)
        self.twts = self.read_wgts_file(self.wgtsfl)

    def read_eigenvec_file(self, filename):
        """
        Read evec file for SP12RTS filtering
        
        This contains the eigenvectors and eigenvalues
        of the model resolution matrix. The file is an
        unformatted fortran file, so we read using 
        scipy.io.FortranFile and note that this format
        does not form part of any standard so is (Fortran)
        processor dependent. We rely on the format being
        readable to us. There are three ways of terminating
        reading. (1) We have read all lenatd eigenvectors,
        (2) the eigenvalues (which are sorted) is smaller
        that a (damping dependent value) or (3) fewer than
        this number of vectors are in the file.
        """
    
        f = spio.FortranFile(filename, 'r')
        
        # Header line 1
        (lmaxh, numatd2, ndep, icrust, idensi1, idum1,
            ismth ) = f.read_ints()
        # Header line 2
        record = f.read_ints()
        mp1 = record[0]
        iparsw = record[1:mp1+1] # What is in the file
        parwts = record[mp1+1:2*mp1+1] # Unclear, only zeros
        # Next bit is a matrix of number of spherical harmonics
        # and number of something else for each part of file.
        # note we need to change order on read and transpose
        # as FortranFile doesn't deal with the different 
        # array order.
        ipardps = record[2*mp1+1:4*mp1+1].reshape((mp1,2)).T 

        # The Fortran version uses data from the spt file to work
        # out how much to read. We use the same information from 
        # the eigenvector file (and check things match before we
        # do any filtering).
        natd = (lmaxh + 1)**2
        assert natd == numatd2, "header size mismatch"
        lenatd = natd * ndep
        
        # Read in all the eigenvectors
        eigvals = np.empty((lenatd))
        eigvecs = np.empty((lenatd,lenatd))
        
        # Read first value and vector record
        # in order to set stop condition
        record = f.read_reals()
        eigvals[0] = record[0]
        eigvecs[0,:] = record[1:lenatd+1]
        eta=eigvals[0] * self.damp
        if self.verbose:
            print("Largest eigenvalue:", eigvals[0], 
                  "ets:", eta)
        stpfact=eta/5000.

        # Read rest of eigen file
        for i in range(1, lenatd+1):
            try:
                record = f.read_reals()
            except:
                if self.verbose:
                    print("Reduced EVC file?")
                    print(i, "vectors read")
                i = i - 1 # Don't use this one
                break
                
            eigvals[i] = record[0]
            eigvecs[i,:] = record[1:lenatd+1]
    
            if eigvals[i] < stpfact:
                if self.verbose:
                    print("Stop building model")
                    print("Last Eigenvector used is", i)
                break
                
        f.close()
                
        return eigvals[:i+1], eigvecs[:i+1, :], icrust, ismth, natd, ndep
    
    
    def read_wgts_file(self, filename):
        """
        Read the weights file
        
        Again unformatted. One interesting feature is that the
        type does not seem to be real*8, reads OK as integer...
        ... so I assume that integer (default prec) is OK and 
        then cast to a float in python (implicit type of twts
        should be real).
        """
        f = spio.FortranFile(filename, 'r')
        (lmaxw, nsmn, nsmx, ndepw, etaz, etah,
          etai, iderh, iderv) = f.read_ints()
        assert ndepw == self.ndep, \
            "Number of depts in wgts and eigv files differ"
        natd = (lmaxw + 1)**2
        assert natd == self.natd, \
            "Number of coefs per depth in wgts and eigv files differ"
        lenatd = natd * ndepw
        twts = np.empty((lenatd))
        for i in range(ndepw):
            ind = (i) * natd
            
            record = f.read_reals('f4')
            
            twts[ind:ind+natd] = record.astype(np.float64)
        
        f.close()

        return twts
    
    def apply_filter(self, x):
        """
        Applies the tomography filter
        
        Reimplementation of the guts of mk3d_res_ESEP.f
        but with array operations when this is easy.
        
        Arguments: x. The input model as a 1D vector. If the 
        model is represented by a set of values varying in radius,
        r, degree, l and order, m, with real and imaginary components the 
        order of the elements (r, l, m, s) in the vector is given by:
        (0, 0, 0, r), (0, 1, 0, r), (0, 1, 1, r), (0, 1, 1, i), 
        (0, 2, 0, r), (0, 2, 1, r), (0, 2, 1, i), (0, 2, 2, r),
        (0, 2, 2, i), (0, 3, 0, r), (0, 3, 1, r), (0, 3, 1, i)
        ... (0, lmax, lamx, r), (0, lmax, lmax, i) ... 
        (rmax, lmax, lmax, r), (rmax, lmax, lmax, i). That is, 
        imaginary coeffs are skipped when they do not exist and the
        data increments through l, m, then r, in that order.
 
        Returns: xout. The filtered model (same structure as x)
        """
        assert x.size >= self.eigvals.size, \
            "Model size / filter mismatch (eval)"
        assert x.size == self.eigvecs[0,:].size, \
            "Model size / filter mismatch (eec)"
        x_out = apply_filter_inner(x, self.eigvals, self.eigvecs, self.twts, self.damp)
                
        if self.ismth == 1:
            if self.verbose:
                print("applying a priori model weights")
            x_out = x_out * self.twts
            
        if self.icrust == 1:
            if self.verbose:
                print("Reweight the crustal thickness")
            x_out[0:self.natd+1] = x_out[0:self.natd+1] * 1000.0
            
        return x_out
    

# jit compiling this bit makes the time per loop go from
# 7.56 s to 59.6 ms, running it in parallel gives us another 
# order of magnitude speedup. This gets hidden by IO if we 
# only filter one model. Note the parallel reduction 
# over x_out and the prange function.
@numba.jit('f8[:](f8[:],f8[:],f8[:,:],f8[:],f8)', nopython=True, parallel=True)
def apply_filter_inner(x, eigvals, eigvecs, twts, damp):
    x_out = np.zeros_like(x)
    twtsinv = 1.0 / twts
    eta = eigvals[0] * damp
    # loop over the eigenvectors and values 
    # we read in
    for i in numba.prange(eigvals.size+1):
            
        w = ( (eigvals[i] / (eigvals[i] + eta)) * 
               np.sum(twtsinv[:] * x[:] * eigvecs[i,:]) )

        x_out += w * eigvecs[i,:]
            
    return x_out

