
from vertex_voyage.data import PandasTable, Schema
from vertex_voyage.data import Table
from vertex_voyage.data_source import DataSource
import pandas as pd 

class CSVDataSource(DataSource):
    def __init__(self, filename):
        self.filename = filename
    
    def get_data(self) -> Table:
        return PandasTable(pd.read_csv(self.filename))
    
    def get_schema(self) -> Schema:
        return Schema(self.get_data().columns())

    def to_dict(self):
        return {"filename": self.filename}