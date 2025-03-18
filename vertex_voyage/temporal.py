
from enum import Enum

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
    def __init__(self, file: str, row: int = 0):
        self.file = open(file, "r")
        for _ in range(row):
            next(self.file)
        self.row = row

    def head(self) -> Event:
        line = next(self.file).strip()
        src, dest, timestamp = line.split(" ")
        return Event(src, dest, int(timestamp))
    
    def tail(self) -> "FileEventSequence":
        return FileEventSequence(self.file.name, self.row + 1)
    
    def append(self, event: Event) -> "FileEventSequence":
        raise NotImplementedError("Cannot append to file")
    
    def empty(self) -> bool:
        try:
            next(self.file)
            self.file = open(self.file.name, "r")
            for _ in range(self.row):
                next(self.file)
            return False
        except StopIteration:
            return True
        
    def __del__(self):
        self.file.close()

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