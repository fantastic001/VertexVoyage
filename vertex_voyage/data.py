from typing import Any, Tuple

class TableColumnsMeta(type):
    def __getitem__(cls, columns: Any):
        if not isinstance(columns, tuple):
            columns = (columns,)
        columns = tuple(str(col) for col in columns)
        name = f"{cls.__name__}_[{','.join(columns)}]"
        return type(name, (cls,), {"__columns__": columns})

class TableColumns(metaclass=TableColumnsMeta):
    __columns__: Tuple[str, ...] = ()

import inspect

def discover_required_columns(func):
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        annotation = param.annotation
        if hasattr(annotation, "__columns__"):
            return annotation.__columns__
        
class Product:
    pass 

class NewColumns(Product):
    def __init__(self, columns):
        self.columns = columns

class UpdatedColumns(Product):
    def __init__(self, columns):
        self.columns = columns

class DeletedColumns(Product):
    def __init__(self, columns):
        self.columns = columns

