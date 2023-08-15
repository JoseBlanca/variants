import numpy


def create_normal_numpy_array(shape, loc=0.0, scale=1.0):
    return numpy.random.default_rng().normal(loc=loc, scale=scale, size=shape)
