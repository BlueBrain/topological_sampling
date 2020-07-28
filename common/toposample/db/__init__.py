from toposample.db import lookup_functions


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
