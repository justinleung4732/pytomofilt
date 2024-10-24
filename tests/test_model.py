import unittest
import pytomofilt.model as mod

class TestDataclass(unittest.TestCase):
    
    def test_dataclass(self):
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

class TestFromFile(unittest.Testcase):
    
    def test_wrong_header(self):
        with self.assertRaises(AssertionError):
            # mod.RTS_Model.from_file() ## FIXME will need a dummy file with wrong header
            pass


    def test_too_many_lines(self):
        pass


    def test_too_much_data(self):
        pass


class TestReparam(unittest.Testcase):

    def test_not_layermodel(self):
        with self.assertRaises(AssertionError):
            mod.RTS_Model.from_file('layer_model') 


    def test_from_layer(self):
        pass