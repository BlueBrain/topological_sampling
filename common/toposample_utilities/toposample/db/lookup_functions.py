"""
toposampling - Topology-assisted sampling and analysis of activity data
Copyright (C) 2020 Blue Brain Project / EPFL & University of Aberdeen

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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


def nanmean(v):
    return numpy.nanmean(v)
