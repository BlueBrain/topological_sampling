"""
toposampling - Topology-assisted sampling and analysis of activity data
Copyright (C) 2020 Blue Brain Project / EPFL & University of Aberdeen

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from toposample.db import lookup_functions


'''This file provides some functionality to interact with the "topological database file" that is generated in the
"gen_topo_db" step. In particular, the "database" contains in some columns not just single values, but vectors
of values. To reduce that to a single value, either an index has to be specified, or a function (such as max or min).

This file provides three convenience functions that serve this purpose. In all functions, the default behavior is
to just return the value in the specified column. If required, index and/or function are specified through
the corresponding kwargs. The function must be specified as a string that names a function inside the
toposample/db/lookup_functions.py file.
'''


def get_entry_from_row(row, column_name, index=None, function=None):
    v = row[column_name]
    if index is not None:
        if len(v) > index:
            v = v[index]
        else:
            v = 0
    if function is not None:
        v = lookup_functions.__dict__[function](v)
    return v


def get_entry_from_database(db, row_index, column_name, index=None, function=None):
    row = db.loc[row_index]
    return get_entry_from_row(row, column_name, index=index, function=function)


def get_column_from_database(db, column_name, index=None, function=None):
    column = db[column_name].values
    if index is not None:
        column = [_x[index] if len(_x) > index else 0 for _x in column]
    if function is not None:
        function = lookup_functions.__dict__[function]
        column = [function(_x) for _x in column]
    return column
