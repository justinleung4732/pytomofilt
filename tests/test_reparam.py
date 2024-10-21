import pytomofilt.model as mod

def test_reparam_dataclass():
    # Build a model
    layers = []
    for depth in [10.0, 20.0, 30.0, 50.0, 440.0, 660.0, 1000.0, 2000.0, 2500.0]:
        layers.append(mod.RealLayer(depth,
                                    lats=[90.0, 0.0, 0.0, 0.0, 0.0, -90.0],
                                    lons=[0.0, 0.0, 90.0, 180.0, 270.0, 0.0], 
                                    vals=[10.0, 20.0, 20.0, 20.0, 20.0, 10.0]
                                    ))
        
    layer_model = mod.RealLayerModel(layers)
    print(layer_model)
    print(layer_model.layers[1].depth)
    print(layer_model.layers[1].lats)
    print(layer_model.layers[1].lons)
    print(layer_model.layers[1].vals)