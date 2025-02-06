
from vertex_voyage.config import get_classes_inheriting
from vertex_voyage.data import * 
from vertex_voyage.model import get_return_type
class DataSource:
    
    def get_data(self) -> Table:
        return None
    
    def get_schema(self) -> Schema:
        return None
    
    def key(self):
        return None
    
    def to_dict(self):
        return {}
    



def load_data_source(class_name, params):
    for cls in get_classes_inheriting(DataSource):
        if cls.__name__ == class_name:
            return cls(**params)
    return None

def get_data_source_info(data_source: DataSource):
    return {
        "schema": data_source.get_schema().columns,
        "key": data_source.key(),
        "params": data_source.to_dict(),
        "class": data_source.__class__.__name__
    }

