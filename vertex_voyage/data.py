
import pandas as pd 
import numpy as np 
class DataType:
    pass 

class Data:
    def get_type(self) -> DataType:
        return None

class TableData(DataType):
    def __init__(self, columns) -> None:
        super().__init__()
        self.columns = columns

class PanelData(DataType):
    def __init__(self, columns, index) -> None:
        super().__init__()
        self.columns = columns
        self.index = index

class TimeSeriesData(DataType):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

class GraphData(DataType):
    def __init__(self, node_attrs, edge_attrs) -> None:
        super().__init__()
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs

class ImageData(DataType):
    def __init__(self, width, height) -> None:
        super().__init__()
        self.width = width
        self.height = height

class ArrayData(DataType):
    def __init__(self, shape) -> None:
        super().__init__()
        self.shape = shape



class DataFrameData(Data):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_type(self) -> DataType:
        return TableData(self.data.columns)
    
    def __getattr__(self, name):
        return getattr(self.data, name)
    
    def __getitem__(self, key):
        return self.data[key]

class NumpyArrayData(Data):
    def __init__(self, data: np.ndarray):
        self.data = data

    def get_type(self) -> DataType:
        return ArrayData(self.data.shape)
    
    def __getattr__(self, name):
        return getattr(self.data, name)
    
    def __getitem__(self, key):
        return self.data[key]