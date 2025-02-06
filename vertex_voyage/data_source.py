
from vertex_voyage.config import get_classes_inheriting

class DataType:
    pass 
class DataSource:
    def get_data_type(self) -> DataType:
        return None
    
    def get_data(self):
        return None
    
    def key(self):
        return None
    
    def to_dict(self):
        return {}
    

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



def load_data_source(class_name, params):
    for cls in get_classes_inheriting(DataSource):
        if cls.__name__ == class_name:
            return cls(**params)
    return None

def get_data_source_info(data_source: DataSource):
    return {
        "data_type": data_source.get_data_type(),
        "key": data_source.key(),
        "params": data_source.to_dict(),
        "class": data_source.__class__.__name__
    }

