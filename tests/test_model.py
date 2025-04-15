import unittest
import unittest
import numpy as np
import tempfile
import os

import pytomofilt.model as mod

class TestDataclass(unittest.TestCase):
    
    def test_dataclass(self):
        """Tests whether RealLayerModel Dataclass saves the correct inputs."""
        # Build a model
        layers = []
        for depth in [10.0, 20.0, 30.0, 50.0, 440.0, 660.0, 1000.0, 2000.0, 2500.0]:
            layers.append(mod.RealLayer(depth,
                                        lats=[90.0, 0.0, 0.0, 0.0, 0.0, -90.0],
                                        lons=[0.0, 0.0, 90.0, 180.0, 270.0, 0.0], 
                                        vals=[10.0, 20.0, 20.0, 20.0, 20.0, 10.0]
                                        ))
            
        layer_model = mod.RealLayerModel(layers)
        self.assertIsInstance(layer_model.layers[1], mod.RealLayer)
        self.assertEqual(layer_model.layers[1].depth, 20.0)
        self.assertEqual(layer_model.layers[1].lats,
                         [90.0, 0.0, 0.0, 0.0, 0.0, -90.0])
        self.assertEqual(layer_model.layers[1].lons,
                         [0.0, 0.0, 90.0, 180.0, 270.0, 0.0])
        self.assertEqual(layer_model.layers[1].vals,
                          [10.0, 20.0, 20.0, 20.0, 20.0, 10.0])


class TestFromFile(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary .sph file for testing."""
        self.test_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        self.test_file.write("5 1111111111111  24 000111111111111111111111\n")  # lmax = 5
        self.test_file.write("0.1\n")  # Sample coefficients
        self.test_file.write("0.2 0.3 0.4\n")
        self.test_file.write("0.5 0.6 0.7 0.8 0.9\n")
        self.test_file.close()
    

    def tearDown(self):
        """Remove the temporary file after tests."""
        os.unlink(self.test_file.name)
    

    def test_from_file_valid(self):
        """Test normal loading from a valid .sph file."""
        obj = mod.RTS_Model.from_file(self.test_file.name)
        self.assertIsInstance(obj, mod.RTS_Model)
        self.assertEqual(obj.lmax, 5)
        self.assertTrue(isinstance(obj.coefs, np.ndarray))
    

    def test_from_file_invalid_format(self):
        """Test loading from an incorrectly formatted file."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as bad_file:
            bad_file.write("Invalid Data\n")
            bad_filename = bad_file.name
        
        with self.assertRaises(ValueError):
            mod.RTS_Model.from_file(bad_filename)
        os.unlink(bad_filename)
    

    def test_from_file_missing_file(self):
        """Test behavior when the file does not exist."""
        with self.assertRaises(FileNotFoundError):
            mod.RTS_Model.from_file("non_existent_file.sph")
    

    def test_from_file_empty_file(self):
        """Test behavior when the file is empty."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as empty_file:
            empty_filename = empty_file.name
        
        with self.assertRaises(StopIteration):  # Next() call on empty file
            mod.RTS_Model.from_file(empty_filename)
        os.unlink(empty_filename)


    def test_too_many_lines(self):
        """Test behavior when the file has too many lines."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as line_file:
            line_file.write("1 1111111111111  24 000111111111111111111111\n")  # lmax = 1
            for i in range(22): # writing for 22 radii instead of 21 (len(knot_radii))
                line_file.write("0.1\n")
                line_file.write("0.2 0.3 0.4\n")
            line_filename = line_file.name
        
        with self.assertRaises(AssertionError):  # Next() call on empty file
            mod.RTS_Model.from_file(line_filename)
        os.unlink(line_filename)


    def test_too_much_data(self):
        """Test behavior when the file has too much data."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as data_file:
            data_file.write("2 1111111111111  24 000111111111111111111111\n")  # lmax = 2
            data_file.write("0.1 0.3\n")
            data_filename = data_file.name
        
        with self.assertRaises(AssertionError):  # Next() call on empty file
            mod.RTS_Model.from_file(data_filename)
        os.unlink(data_filename)


class TestFromDirectory(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory with sample data."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.depth_file = os.path.join(self.test_dir.name, 'depth_layers.dat')
        with open(self.depth_file, 'w') as f:
            f.write("10.0\n20.0\n30.0\n")
        
        for i in range(3):
            with open(os.path.join(self.test_dir.name, f"layer_{i}.dat"), 'w') as f:
                f.write("0.0 90.0 10.0\n0.0 0.0 20.0\n90.0 0.0 20.0\n")
    
    
    def tearDown(self):
        """Remove the temporary directory after tests."""
        self.test_dir.cleanup()
    

    def test_from_directory_valid(self):
        """Test normal loading from a valid directory."""
        obj = mod.RTS_Model.from_directory(self.test_dir.name, lmax=5)
        self.assertIsInstance(obj, mod.RTS_Model)
    

    def test_from_directory_missing_depth_file(self):
        """Test behavior when depth_layers.dat is missing."""
        os.remove(self.depth_file)
        with self.assertRaises(FileNotFoundError):
            mod.RTS_Model.from_directory(self.test_dir.name, lmax=5)
    

    def test_from_directory_no_data_files(self):
        """Test behavior when there are no data files."""
        for i in range(3):
            os.remove(os.path.join(self.test_dir.name, f"layer_{i}.dat"))
        with self.assertRaises(IndexError):
            mod.RTS_Model.from_directory(self.test_dir.name, lmax=5)
            

class TestReparam(unittest.TestCase):

    def setUp(self):
        """Set up a test instance and a mock RealLayerModel."""
        self.test_obj = mod.RTS_Model(lmax=5, rmin=1, rmax=10, knots=None)
        self.test_obj.coefs = np.zeros((3, 2, 6, 6))

        layers = []
        for depth in [10.0, 20.0, 30.0, 2000.0, 2500.0]:
            layers.append(mod.RealLayer(depth,
                                        lats=[90.0, 0.0, 0.0, 0.0, 0.0, -90.0],
                                        lons=[0.0, 0.0, 90.0, 180.0, 270.0, 0.0], 
                                        vals=[10.0, 20.0, 20.0, 20.0, 20.0, 10.0]
                                        ))
        self.layer_model = mod.RealLayerModel(layers)


    def test_not_layermodel(self):
        """"Test whether layer_model input is a RealLayerModel object"""
        with self.assertRaises(AssertionError):
            mod.RTS_Model.reparam('layer_model') 

    
    def test_reparam_valid(self):
        """Test reparam method with valid input."""
        self.test_obj.reparam(self.layer_model)
        self.assertTrue(isinstance(self.test_obj.coefs, np.ndarray))


class TestWrite(unittest.TestCase):
    
    def setUp(self):
        """Set up a test instance of RTS_Model."""
        self.test_obj = mod.RTS_Model(lmax=5, rmin=1, rmax=10, knots=[0.1, 0.2, 0.3])
        self.test_obj.coefs = np.zeros((3, 2, 6, 6))


    def test_write(self):
        """Test the write method by writing and reading a file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            self.test_obj.write(temp_file.name)
            temp_filename = temp_file.name
        
        self.assertTrue(os.path.exists(temp_filename))
        obj = mod.RTS_Model.from_file(temp_filename)
        self.assertEqual(self.test_obj.lmax, obj.lmax)
        self.assertEqual(self.test_obj.coefs, obj.coefs)
        os.remove(temp_filename)


class TestFilterMethods(unittest.TestCase):

    def setUp(self):
        """Set up a test instance of RTS_Model."""
        self.test_obj = mod.RTS_Model(lmax=5, rmin=1, rmax=10, knots=[0.1, 0.2, 0.3])
        self.test_obj.coefs = np.zeros((3, 2, 6, 6))
        self.test_obj.filter_obj = MockFilter()
    

    def test_as_vector(self):
        """Test the as_vector method."""
        vector = self.test_obj.as_vector()
        self.assertEqual(len(vector), (5 + 1) ** 2 * 21)  # Validate vector size
    

    def test_from_vector(self):
        """Test the from_vector method."""
        vector = np.ones((5 + 1) ** 2 * 21)
        self.test_obj.from_vector(vector)
        self.assertTrue(np.all(self.test_obj.coefs == 1))
    

    def test_filter(self):
        """Test the filter method."""
        model = mod.RTS_Model(lmax=5, rmin=1, rmax=10, knots=[0.1, 0.2, 0.3])
        model.coefs = np.random.rand(3, 2, 6, 6)
        self.test_obj.filter_obj = MockFilter()
        filtered_model = self.test_obj.filter(model)
        self.assertIsInstance(filtered_model, mod.RTS_Model)
    

    def test_filter_without_filter_obj(self):
        """Ensure filter raises an error if filter_obj is not set."""
        self.test_obj.filter_obj = None
        model = mod.RTS_Model(lmax=5, rmin=1, rmax=10, knots=[0.1, 0.2, 0.3])
        with self.assertRaises(AssertionError):
            self.test_obj.filter(model)
    

    def test_filter_invalid_input(self):
        """Ensure filter raises an error for invalid input type."""
        with self.assertRaises(AssertionError):
            self.test_obj.filter("invalid_input")
    

class MockFilter:
    def apply_filter(self, vector):
        return vector * 0.5  # Mock filtering behavior