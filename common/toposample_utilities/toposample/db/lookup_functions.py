import numpy


def smallest_nonzero_value(v):
    v = [_v for _v in v if _v != 0]
    if len(v) > 0:
        return numpy.min(v)
    return 0


def smallest_nonzero_absolute_value(v):
    v = numpy.abs(v)
    return smallest_nonzero_value(v)


def difference_between_largest_values(v):
    if len(v) < 2:
        return 0.0
    v = numpy.sort(v)
    return v[-1] - v[-2]


def difference_between_largest_absolute_values(v):
    v = numpy.abs(v)
    return difference_between_largest_values(v)


def largest_value(v):
    if len(v) == 0:
        return 0.0
    return numpy.max(v)


def largest_absolute_value(v):
    v = numpy.abs(v)
    return largest_value(v)
