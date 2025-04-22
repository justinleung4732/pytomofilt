import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import numpy.testing as npt

from pytomofilt.filter import Filter

class TestReadEigenvec(unittest.TestCase):

    def setUp(self):
        
        # Setup a dummy Filter object with required attributes
        self.filt = Filter.__new__(Filter)
        self.filt.damp = 0.1
        self.filt.verbose = False

    
    def test_read_eigenvec_header_size_mismatch(self):
        """
        Test assertion that natd == numatd2. numatd2 is
        the second entry of header 1 and natd = (lmaxh+1)**2,
        where lmaxh is the first entry of header 1.        
        """
        mock_file = MagicMock()
        mock_file.read_ints.side_effect = [
            np.array([2, 4, 1, 1, 0, 0, 1]),  # Header 1
            np.array([1, 0, 0, 0, 0, 0, 0])   # Header 2
        ]

        with patch('pytomofilt.filter.spio.FortranFile', return_value=mock_file):
            with self.assertRaises(AssertionError):
                self.filt.read_eigenvec_file("dummy")  


    def test_read_eigenvec_file(self):

        mock_file = MagicMock()
        icrust = 1
        ismth = 1
        lmaxh = 1
        ndep = 1
        eigvals = [1.0, 0.4, 0.3, 0.2] 
        eigvecs = [[1.0, 0.0, 0.0, 0.0],   # First eigenvector
                   [0.0, 1.0, 0.0, 0.0],   # Second eigenvector
                   [0.0, 0.0, 1.0, 0.0],   # Third eigenvector
                   [0.0, 0.0, 0.0, 1.0]]   # Fourth eigenvector
        mock_file.read_ints.side_effect = [
            np.array([lmaxh, 4, ndep, icrust, 0, 0, ismth]),  # Header 1
            np.array([1, 0, 0, 0, 0, 0, 0])   # Header 2
        ]
        mock_file.read_reals.side_effect = [np.array([val]+vec) for val,vec in zip(eigvals,eigvecs)]

        with patch('pytomofilt.filter.spio.FortranFile', return_value=mock_file):
            result = self.filt.read_eigenvec_file("dummy")
            npt.assert_array_equal(eigvals, result[0])
            npt.assert_array_equal(eigvecs, result[1])
            self.assertEqual(icrust, result[2])
            self.assertEqual(ismth, result[3])
            self.assertEqual((lmaxh+1)**2, result[4])
            self.assertEqual(ndep, result[5])


    def test_eigenvec_file_small_eigvals(self):
        """
        Tests that file reading stops after the first eigenvalue smaller 
        than stpfact. Stpfact is given by: largest_eigval * damp / 5000.
        In this example, stpfact = 1.0 * 0.1 / 5000 = 0.002, so the last
        eigenvector should not be recorded.
        """
        mock_file = MagicMock()
        eigvals = [1.0, 0.4, 0.000015, 0.00001] 
        eigvecs = [[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]]
        mock_file.read_ints.side_effect = [
            np.array([1, 4, 1, 1, 0, 0, 1]),  # Header 1
            np.array([1, 0, 0, 0, 0, 0, 0])   # Header 2
        ]
        mock_file.read_reals.side_effect = [np.array([val]+vec) for val,vec in zip(eigvals,eigvecs)]

        with patch('pytomofilt.filter.spio.FortranFile', return_value=mock_file):
            result = self.filt.read_eigenvec_file("dummy")
            npt.assert_array_equal(eigvals[:-1], result[0])
            npt.assert_array_equal(eigvecs[:-1], result[1])


class TestReadWgts(unittest.TestCase):

    def setUp(self):
        
        # Setup a dummy Filter object with required attributes
        self.filt = Filter.__new__(Filter)
        self.filt.ndep = 3
        self.filt.natd = 9

        self.mock_file = MagicMock()
        self.mock_file.read_ints.return_value = [2, 0, 0, 3, 0, 0, 0, 0, 0]
        self.mock_file.read_reals.return_value = np.arange(0, 9, dtype='float32')


    def test_read_wgts_file(self):

        with patch('pytomofilt.filter.spio.FortranFile', return_value=self.mock_file):
            result = self.filt.read_wgts_file("dummy")
            self.assertEqual(len(result), 27)  # 3 depths * 9 = 27
            self.assertEqual(result.dtype, np.float64)
            npt.assert_array_equal(self.mock_file.read_reals.return_value, result[:9])


    def test_wgts_file_wrong_num_depts(self):
        """
        Testing assertion that Filter.ndep = ndepw
        """
        self.filt.ndep = 5  # Should be 3
        with patch('pytomofilt.filter.spio.FortranFile', return_value=self.mock_file):
            with self.assertRaises(AssertionError):
                self.filt.read_wgts_file("dummy")


    def test_wgts_file_wrong_coefs_per_depth(self):
        """
        Testing assertion that Filter.natd = natd = (lmaxw+1)**2
        """
        self.filt.ndep = 5  # Should be 9
        with patch('pytomofilt.filter.spio.FortranFile', return_value=self.mock_file):
            with self.assertRaises(AssertionError):
                self.filt.read_wgts_file("dummy")


class TestApplyFilter(unittest.TestCase):

    @patch.object(Filter, 'read_eigenvec_file')
    @patch.object(Filter, 'read_wgts_file')
    def setUp(self, mock_read_wgts, mock_read_evec):

        # Setup a dummy Filter object with required attributes
        eigvals = np.array([1.0, 0.4, 0.3, 0.2])
        eigvecs = np.array([[1.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0]])
        icrust = 1 
        ismth = 1
        natd = 4
        ndep = 1
        
        mock_read_evec.return_value = (eigvals, eigvecs, icrust,
                                       ismth, natd, ndep)
        mock_read_wgts.return_value = np.arange(1, (natd * ndep) + 1, 
                                                dtype=np.float64)

        self.filt = Filter("dummy_evec", "dummy_wgts", damping=0.1, verbose=False)
    

    def test_apply_filter_size_incorrect(self):
        """
        Test assertion that the input model x has to be larger than 
        the number of eigenvalues and the same dimension as the 
        eigenvectors.
        """
        x_too_small = np.array([0] * 2)
        with self.assertRaises(AssertionError):
            self.filt.apply_filter(x_too_small)

        x_too_large = np.array([0] * 5)
        with self.assertRaises(AssertionError):
            self.filt.apply_filter(x_too_large)


    def test_apply_filter_correctly(self):
        # Create an input vector
        x = np.array([1.0, 2.0, 3.0, 4.0])

        # No model weights or crustal reweighing applied
        self.filt.ismth = 0
        self.filt.icrust = 0
        x_out = self.filt.apply_filter(x)

        self.assertEqual(x_out.shape, x.shape)
        self.assertFalse(np.array_equal(x_out, x))  # Confirm it has been modified


    def test_apply_filter_a_priori_model_weights(self):

        # Create an input vector
        x = np.array([1.0, 2.0, 3.0, 4.0])
    
        # Calculating case with model weight boolean to True
        self.filt.ismth = 1
        self.filt.icrust = 0
        x_model_weights = self.filt.apply_filter(x)

        # Calculating reference case
        self.filt.ismth = 0
        x_out = self.filt.apply_filter(x)

        npt.assert_array_equal(x_model_weights,
                               x_out * self.filt.twts)


    def test_apply_filter_crust_weighting(self):

        # Create an input vector
        x = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Calculating case with crustal weighting
        self.filt.icrust = 1
        self.filt.ismth = 0
        x_crustal = self.filt.apply_filter(x)

        # Calculating reference case
        self.filt.icrust = 0
        x_out = self.filt.apply_filter(x)

        # Check crust weighting has been applied to first natd+1 elements
        npt.assert_array_equal(x_crustal[:self.filt.natd+1],
                               x_out[:self.filt.natd+1] * 1000.0)