import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from typing import Any, Tuple
import inspect
import numpy as np 

class TableColumnsMeta(type):
    def __getitem__(cls, columns: Any):
        if not isinstance(columns, tuple):
            columns = (columns,)
        columns = tuple(str(col) for col in columns)
        name = f"{cls.__name__}_[{','.join(columns)}]"
        return type(name, (cls,), {"__columns__": columns})

class TableColumns(metaclass=TableColumnsMeta):
    __columns__: Tuple[str, ...] = ()




def discover_required_columns(func):
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        annotation = param.annotation
        if hasattr(annotation, "__columns__"):
            return annotation.__columns__
        
class Product(ABC):
    @abstractmethod
    def produce(self, schema: "Schema") -> "Schema":
        pass

class NewColumns(Product):
    def __init__(self, columns: Dict[str, Any]):
        self.columns = columns
    
    def produce(self, schema: "Schema") -> "Schema":
        return schema.with_columns(self.columns)

class UpdatedColumns(Product):
    def __init__(self, columns: List[str]):
        self.columns = columns
    
    def produce(self, schema: "Schema") -> "Schema":
        new_columns = {col: schema.columns[col] for col in schema.columns if col not in self.columns}
        return schema.with_columns(new_columns)

class DeletedColumns(Product):
    def __init__(self, columns: List[str]):
        self.columns = columns
    
    def produce(self, schema: "Schema") -> "Schema":
        new_columns = {col: schema.columns[col] for col in schema.columns if col not in self.columns}
        return Schema(new_columns)


class Table(ABC):
    
    @abstractmethod
    def columns(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def column(self, name: str) -> np.ndarray:
        pass

    def matrix(self, names: List[str]) -> np.ndarray:
        return np.column_stack([self.column(name) for name in names])

    @abstractmethod
    def apply(self, column: str, func: callable) -> "Table":
        pass

    @abstractmethod
    def select(self, *columns: str) -> "Table":
        pass

    @abstractmethod
    def filter(self, func: callable) -> "Table":
        pass

    @abstractmethod
    def update(self, column: str, values: np.ndarray):
        pass

    @abstractmethod
    def concat(self, other: "Table") -> "Table":
        pass

    def query(self, query: "QueryExpression") -> "Table":
        return query.apply(self)

    def __getitem__(self, query) -> "Table":
        if isinstance(query, str):
            return self.column(query)
        if isinstance(query, tuple):
            return self.select(*query)
        if isinstance(query, QueryExpression):
            return self.query(query)
        raise ValueError("Invalid query type")
    
    def __call__(self, func: callable) -> "Table":
        return self.apply(func)
    
    def __or__(self, other: "Table") -> "Table":
        return self.concat(other)
    
    def __setitem__(self, column: str, values: np.ndarray):
        return self.update(column, values)
    



class QueryExpression(ABC):
    @abstractmethod
    def apply(self, table: Table) -> Table:
        pass

        
    def __and__(self, other: "QueryExpression") -> "QueryExpression":
        return And(self.expression, other)
    
    def __or__(self, other: "QueryExpression") -> "QueryExpression":
        return Or(self.expression, other)
    
    def __invert__(self) -> "QueryExpression":
        return Not(self.expression)
    

class Select(QueryExpression):
    def __init__(self, *columns: str):
        self.columns = columns
    
    def apply(self, table: Table) -> Table:
        return table.select(*self.columns)
    
class Equal(QueryExpression):
    def __init__(self, column: str, value: Any):
        self.column = column
        self.value = value
    
    def apply(self, table: Table) -> Table:
        return table.filter(lambda x: x[self.column] == self.value)

class GreaterThan(QueryExpression):
    def __init__(self, column: str, value: Any):
        self.column = column
        self.value = value
    
    def apply(self, table: Table) -> Table:
        return table.filter(lambda x: x[self.column] > self.value)

class LessThan(QueryExpression):
    def __init__(self, column: str, value: Any):
        self.column = column
        self.value = value
    
    def apply(self, table: Table) -> Table:
        return table.filter(lambda x: x[self.column] < self.value)

class And(QueryExpression):
    def __init__(self, *expressions: QueryExpression):
        self.expressions = expressions
    
    def apply(self, table: Table) -> Table:
        for expr in self.expressions:
            table = expr.apply(table)
        return table

class Or(QueryExpression):
    def __init__(self, *expressions: QueryExpression):
        self.expressions = expressions
    
    def apply(self, table: Table) -> Table:
        accumulator = self.expressions[0].apply(table)
        for expr in self.expressions[1:]:
            accumulator = accumulator.concat(expr.apply(table))
        return accumulator
    
class Not(QueryExpression):
    def __init__(self, expression: QueryExpression):
        self.expression = expression
    
    def apply(self, table: Table) -> Table:
        return table.filter(lambda x: not self.expression.apply(x))



class Schema:
    """
    Schema class that holds a mapping of column names to expected Python (or NumPy) types.
    """
    def __init__(self, columns: Dict[str, Any]):
        """
        Initialize Schema with a dict of {column_name: python_type}
        """
        self.columns = dict(columns)  # make a copy

    def with_column(self, col_name: str, col_type: Any) -> 'Schema':
        """
        Returns a new Schema with an additional (or updated) column -> type mapping.
        """
        new_columns = dict(self.columns)
        new_columns[col_name] = col_type
        return Schema(new_columns)
    
    def with_columns(self, columns: Dict[str, Any]) -> 'Schema':
        """
        Returns a new Schema with additional (or updated) columns -> type mappings.
        """
        new_columns = dict(self.columns)
        new_columns.update(columns)
        return Schema(new_columns)

    def check_against(self, table: Table) -> bool:
        """
        Checks that the DataFrame meets the schema requirements:
          1. Contains all columns listed in the schema
          2. The dtypes match the specified python types (simplified check).
        
        Returns True if valid, otherwise raises a ValueError (or return False).
        
        Note: The actual dtype check may need to be more elaborate depending on
        how strictly you need to enforce data types (e.g., int vs. int64).
        """
        for col, expected_type in self.columns.items():
            if col not in table.columns():
                return False
            actual_type = table.columns()[col]
            if not issubclass(actual_type, expected_type):
                return False 
        return True


# ---------------------------
# Abstract checks
# ---------------------------
class SchemaCheck(ABC):
    """
    Abstract base class for schema checks.
    """
    @abstractmethod
    def check(self, table: Table) -> bool:
        """
        Returns True if the DataFrame meets the check criterion, otherwise False.
        """
        pass

    def __and__(self, other: 'SchemaCheck') -> 'SchemaCheck':
        """
        Enable chaining checks with the & (AND) operator.
        """
        return _AndCheck(self, other)

    def __or__(self, other: 'SchemaCheck') -> 'SchemaCheck':
        """
        Enable chaining checks with the | (OR) operator.
        """
        return _OrCheck(self, other)


class _AndCheck(SchemaCheck):
    """
    Helper class to combine two checks with a logical AND.
    """
    def __init__(self, left: SchemaCheck, right: SchemaCheck):
        self.left = left
        self.right = right

    def check(self, df) -> bool:
        return self.left.check(df) and self.right.check(df)


class _OrCheck(SchemaCheck):
    """
    Helper class to combine two checks with a logical OR.
    """
    def __init__(self, left: SchemaCheck, right: SchemaCheck):
        self.left = left
        self.right = right

    def check(self, df) -> bool:
        return self.left.check(df) or self.right.check(df)


class SatisfiesAll(SchemaCheck):
    """
    Checks if the DataFrame satisfies all the given checks.
    """
    def __init__(self, checks: List[SchemaCheck]):
        self.checks = checks

    def check(self, df) -> bool:
        return all(check.check(df) for check in self.checks)
    
class SatisfiesAny(SchemaCheck):
    """
    Checks if the DataFrame satisfies any of the given checks.
    """
    def __init__(self, checks: List[SchemaCheck]):
        self.checks = checks

    def check(self, df) -> bool:
        return any(check.check(df) for check in self.checks)
    

class HasColumn(SchemaCheck):
    """
    Checks if a column exists in the DataFrame.
    """
    def __init__(self, col_name: str):
        self.col_name = col_name

    def check(self, table: Table) -> bool:
        return self.col_name in table.columns()
    
    def with_type(self, col_type: Any) -> 'HasColumnWithType':
        return HasColumnWithType(self.col_name, col_type)


class HasColumnWithType(SchemaCheck):
    """
    Checks if a column with a specific type exists in the DataFrame.
    """
    def __init__(self, col_name: str, col_type: Any):
        self.col_name = col_name
        self.col_type = col_type

    def check(self, df: Table) -> bool:
        if self.col_name not in df.columns():
            return False
        return issubclass(df.columns()[self.col_name], self.col_type)


