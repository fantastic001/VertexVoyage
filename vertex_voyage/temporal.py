
from enum import Enum
from random import random as rand

class EventType(Enum):
    ADD = 1
    REMOVE = 2

class Event:
    def __init__(
            self, 
            src: str | int, 
            dest: str | int, 
            timestamp: int,
            type: EventType = EventType.ADD,
            attrs: dict = None
        ):
        self.src = src
        self.dest = dest
        self.timestamp = timestamp
        self.attrs = attrs
        self.type = type
    
    def __str__(self):
        return f"Event({self.src}, {self.dest}, {self.timestamp}, {self.type}, {self.attrs})"

    def __repr__(self):
        return str(self)

class EventSequence:
    def head(self) -> Event:
        """
        Returns first event in sequence
        """
        pass 
    def tail(self) -> "EventSequence":
        """
        Returns all events except the first
        """
        pass
    def append(self, event: Event) -> "EventSequence":
        """
        Appends event to sequence
        """
        pass

    def empty(self) -> bool:
        """
        Returns True if sequence is empty
        """
        pass

    def __iter__(self):
        return EventStream(self)

class ListEventSequence(EventSequence):
    def __init__(self, events: list[Event] = None):
        self.events = events or []
    
    def head(self) -> Event:
        return self.events[0]
    
    def tail(self) -> "ListEventSequence":
        return ListEventSequence(self.events[1:])
    
    def append(self, event: Event) -> "ListEventSequence":
        return ListEventSequence(self.events + [event])
    
    def empty(self) -> bool:
        return len(self.events) == 0
    
class CSVEventSequence(EventSequence):
    def __init__(self, csv_file: str, row: int = 0):
        self.csv_file = csv_file
        self.file = open(csv_file, "r")
        next(self.file) # Skip header
        for _ in range(row):
            next(self.file) # Skip rows
        self.row = row
    
    def head(self) -> Event:
        line = next(self.file).strip()
        src, dest, timestamp, type, *attrs = line.split(",")
        return Event(src, dest, int(timestamp), EventType[type], dict([attr.split("=") for attr in attrs]))
    
    def tail(self) -> "CSVEventSequence":
        return CSVEventSequence(self.csv_file, self.row + 1)
    
    def append(self, event: Event) -> "CSVEventSequence":
        raise NotImplementedError("Cannot append to CSV file")
    
    def empty(self) -> bool:
        try:
            next(self.file)
            self.file = open(self.file.name, "r")
            next(self.file) # Skip header
            for _ in range(self.row):
                next(self.file) # Skip rows
            return False
        except StopIteration:
            return True
        
    def __del__(self):
        self.file.close()

class FileEventSequence(EventSequence):
    def __init__(self, file: str, fileptr = None):
        if fileptr:
            self.file = fileptr
        else:
            self.file = open(file, "r")
        self.is_empty = False 
        try:
            self.h = next(self.file).strip()
        except StopIteration:
            self.is_empty = True

    def head(self) -> Event:
        line = self.h
        src, dest, timestamp = line.split(" ")
        return Event(src, dest, int(timestamp))
    
    def tail(self) -> "FileEventSequence":
        return FileEventSequence(self.file.name, self.file)
    
    def append(self, event: Event) -> "FileEventSequence":
        raise NotImplementedError("Cannot append to file")
    
    def empty(self) -> bool:
        return self.is_empty

class BarabasiAlbertEventSequence(EventSequence):
    def __init__(self, nodes = None, degree = None, current = None, iter = 0):
        if nodes:
            self.nodes = nodes
        else:
            self.nodes = []
        if degree:
            self.degree = degree
        else:
            self.degree = {} 
        if current:
            self.current = current
        else:
            self.current = None  
        self.iter = iter
    
    def head(self) -> Event:
        if self.current is None:
            if len(self.nodes) == 0:
                mynode = 0
            else:
                mynode = max(self.nodes) + 1
            self.nodes.append(mynode)
            self.degree[mynode] = 0
            self.current = self.nodes[0]
            if sum(self.degree.values()) == 0:
                p = 0.5
            else:
                p = self.degree[self.current] / sum(self.degree.values())
            if rand() < p:
                return Event(
                    self.current, 
                    mynode, 
                    self.iter,
                    EventType.ADD,
                    {}
                )
            else:
               self.current = self.nodes[self.nodes.index(self.current) + 1]
               self.iter += 1
               return self.head()
        else:
            if self.current == self.nodes[-1]:
                self.current = None 
                self.iter += 1
                return self.head()
            else:
                if sum(self.degree.values()) == 0:
                    p = .5
                else:
                    p = self.degree[self.current] / sum(self.degree.values())
                if rand() < p:
                    return Event(
                        self.current,
                        self.nodes[-1],
                        self.iter,
                        EventType.ADD,
                        {}
                    )
                else:
                    self.current = self.nodes[self.nodes.index(self.current) + 1]
                    return self.head()


    def tail(self) -> "BarabasiAlbertEventSequence":
        if self.current == self.nodes[-1]:
            return BarabasiAlbertEventSequence(self.nodes, self.degree, None, self.iter + 1)
        else:
            return BarabasiAlbertEventSequence(self.nodes, self.degree, self.nodes[self.nodes.index(self.current) + 1], self.iter + 1)

    def append(self, event: Event) -> "BarabasiAlbertEventSequence":
        raise NotImplementedError("Cannot append to BarabasiAlbertEventSequence")

    def empty(self) -> bool:
        return False

class EventStream:
    def __init__(self, sequence: EventSequence):
        self.sequence = sequence
    
    def next(self) -> Event:
        return self.sequence.head()
    
    def empty(self) -> bool:
        return self.sequence.empty()
    
    def advance(self) -> "EventStream":
        return EventStream(self.sequence.tail())
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Event:
        if self.empty():
            raise StopIteration
        event = self.next()
        self.sequence = self.advance().sequence
        return event

if __name__ == "__main__":
    sequence = FileEventSequence("data/wiki-talks/wiki.txt")
    i = 0
    for event in sequence:
        if i == 10:
            break
        print(event.timestamp)