
from vertex_voyage.config import get_classes_inheriting
from vertex_voyage.data import * 
class DataSource:
    def get_data_type(self) -> DataType:
        return None
    
    def get_data(self):
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
        "data_type": data_source.get_data_type(),
        "key": data_source.key(),
        "params": data_source.to_dict(),
        "class": data_source.__class__.__name__
    }

