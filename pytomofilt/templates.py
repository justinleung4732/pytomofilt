#!/usr/bin/env python
"""Templates for geodynamic models and tomographic filters"""

from abc import ABC
from abc import abstractmethod

class AbstractModel(ABC):
    """
    AbstractModel defines the interface for a geodynamic model that can be filtered and compared
    to tomography.
    """

    @classmethod
    @abstractmethod
    def from_file(cls, filename, rmin, rmax, knots):
        """
        Read a geodynamic model from file and return an instance of the model.
        """
        pass

    @abstractmethod
    def filter_from_file(self, *files):
        """
        Read the filter files and apply the filter to the model.
        This is a wrapper for the filter class, which is where the actual filtering is done.
        """
        pass

    @abstractmethod
    def filter(self, model):
        """
        Apply the filter to the model.
        """
        pass

    @abstractmethod
    def write(self, filename):
        """
        Write the model to file.
        """
        pass


class AbstractFilter(ABC):
    """
    AbstractFilter defines the interface for a tomographic filter that can be applied to a
    geodynamic model.
    """

    @abstractmethod
    def read_eigenvec_file(self, filename):
        """
        Read the eigenvector file and return the eigenvalues, eigenvectors and other parameters.
        """
        pass

    @abstractmethod
    def read_wgts_file(self, filename):
        """
        Read the weights file and return the weights and other parameters.
        """
        pass

    @abstractmethod
    def apply_filter(self, x):
        """
        Apply the filter to a model of seismic velocities and return the filtered model.
        """
        pass
