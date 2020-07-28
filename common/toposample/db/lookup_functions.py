import numpy


def smallest_nonzero_value(v):
    v = [_v for _v in v if _v != 0]
    if len(v) > 0:
        return numpy.min(v)
    return 0
