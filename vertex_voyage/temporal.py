
from enum import Enum
from random import choice, random as rand, randint, sample
from random import shuffle
from struct import pack, unpack

class EventType(Enum):
    ADD = 1
    REMOVE = 2

class Event:
    def __init__(
            self, 
            src: str, 
            dest: str, 
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


class ShuffledSequence(EventSequence):
    def __init__(self, sequence: EventSequence, window: int, buffer: list[Event] = None):
        """
        Shuffles a sequence of events. The first window events are shuffled
        and the rest are appended to the buffer. The buffer is shuffled
        and the first window events are returned. The rest of the events
        are returned in the order they were added to the buffer. The
        buffer is shuffled again after each window.
        :param sequence: Sequence of events
        :param window: Window size
        :param buffer: Buffer of events
        """
        if isinstance(sequence, EventStream):
            self.sequence = sequence
        else:
            self.sequence = EventStream(sequence)
        self.window = window
        if buffer is None:
            self.buffer = []
        else:
            self.buffer = list(buffer).copy() 
        shuffle(self.buffer)
        if len(self.buffer) < self.window:
            i = len(self.buffer)
            for event in self.sequence:
                self.buffer.append(event)
                i += 1
                if i == self.window:
                    break
        earliest_event_index = self.buffer.index(min(self.buffer, key=lambda x: x.timestamp))
        earliest_timestamp = self.buffer[earliest_event_index].timestamp
        first_timestamp = self.buffer[0].timestamp
        if earliest_timestamp != first_timestamp:
            self.buffer[0].timestamp = earliest_timestamp
            self.buffer[earliest_event_index].timestamp = first_timestamp
    
    def head(self) -> Event:
        return self.buffer[0]
        
    
    def tail(self) -> "ShuffledSequence":
        return ShuffledSequence(self.sequence, self.window, self.buffer[1:])
    
    def append(self, event: Event) -> "ShuffledSequence":
        raise NotImplementedError("Cannot append to ShuffledSequence")
    
    def empty(self) -> bool:
        return self.sequence.empty() and len(self.buffer) == 0
    

class FromIterable(EventSequence):

    def __init__(self, iterable):
        self.iterable = iterable
        self._head = next(iterable, None)

    def head(self) -> Event:
        return self._head
    
    def tail(self) -> "FromIterable":
        return FromIterable(self.iterable)
    
    def append(self, event: Event) -> "FromIterable":
        raise NotImplementedError("Cannot append to FromIterable")
    
    def empty(self) -> bool:
        return self._head is None

class ForestFireEventSequence(FromIterable):
    def __forest_fire_iter(self):
        """
        Generates a forest fire event sequence. The probability of
        generating an edge is p. The probability of generating a
        node is 1 - p.
        """
        vertices = [0]
        i = 1 
        time = 0
        while True:
            vertices.append(i)
            ambassador = choice(vertices[:-1])
            visited = set()
            to_visit = [ambassador]
            while to_visit:
                current = to_visit.pop()
                if current in visited:
                    continue
                visited.add(current)
                yield Event(
                    current,
                    i,
                    time,
                    EventType.ADD,
                    {}
                )
                time += 1
                burn_fwd = sample(vertices, int(self.p * len(vertices)))
                to_visit.extend(burn_fwd)
            i += 1
    def __init__(self, p: float):
        """
        Generates a forest fire event sequence. The probability of
        generating an edge is p. The probability of generating a
        node is 1 - p.
        
        :param p: Probability of generating an edge
        """
        self.p = p
        super().__init__(self.__forest_fire_iter())
class SBMSequence(FromIterable):
    def __init__(self, community_probs: list[float], p: list[list[float]]):
        self.community_probs = community_probs
        self.p = p
        assert len(p) == len(community_probs)
        for row in p:
            assert len(row) == len(community_probs)
            assert all([0 <= p_ij <= 1 for p_ij in row])
        self.cum_probs = [community_probs[0]]
        for i in range(1, len(community_probs)):
            self.cum_probs.append(self.cum_probs[-1] + community_probs[i])
        self.node_membership = {}
        self.total_nodes = 0
        super().__init__(self._sbm_iter())
        
    
    def _node_membership(self, node: int) -> int:
        if node in self.node_membership:
            return self.node_membership[node]
        else:
            r = rand()
            for i in range(len(self.cum_probs)):
                if r <= self.cum_probs[i]:
                    self.node_membership[node] = i
                    return i
    

    def _sbm_iter(self):
        iter = 0
        while True:
                i = self.total_nodes
                for j in range(self.total_nodes):
                    bi = self._node_membership(i)
                    bj = self._node_membership(j)
                    if rand() < self.p[bi][bj]:
                        iter += 1
                        yield Event(i, j, iter, EventType.ADD, {
                            "community_src": bi,
                            "community_dst": bj
                        })
                self.total_nodes += 1

class BinaryEventSequence(EventSequence):
    """
    Binary event sequence in a file. Same as FileEventSequence, but
    stores events in binary format. Each event is stored as a tuple of
    (src, dest, timestamp) in a binary file. All elements of tuple are
    stored as 4 byte integers. The file is opened in binary mode.

    """
    def __init__(self, path: str, fileptr = None):
        self.path = path
        if fileptr:
            self.file = fileptr
        else:
            self.file = open(path, "rb")
        self.is_empty = False
        self.h = self.file.read(12)
        if len(self.h) == 0:
            self.is_empty = True
        elif len(self.h) != 12:
            raise ValueError("Invalid event header. Event header must be 12 bytes long")
        else:
            self.h = unpack("iii", self.h)
            if len(self.h) != 3:
                raise ValueError("Invalid event header. Event header must be 12 bytes long")
            
    def head(self) -> Event:
        src, dest, timestamp = self.h
        return Event(src, dest, timestamp, EventType.ADD, {})
    def tail(self) -> "BinaryEventSequence":
        return BinaryEventSequence(self.path, self.file)
    def append(self, event: Event) -> "BinaryEventSequence":
        raise NotImplementedError("Cannot append to BinaryEventSequence")
    def empty(self) -> bool:
        return self.is_empty
    
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


class FirstN(FromIterable):
    def _draw_first(self, sequence: EventSequence, n: int):
        i = 0
        for event in sequence:
            if i == n:
                break
            i += 1
            yield event

    def __init__(self, sequence: EventSequence, n: int):
        self.sequence = sequence
        self.n = n
        super().__init__(self._draw_first(sequence, n))

def to_nx_graph(tg: EventSequence):
    """
    Returns a networkx graph from an event sequence
    """
    import networkx as nx
    G = nx.Graph()
    for event in tg:
        if event.type == EventType.ADD:
            G.add_edge(event.src, event.dest)
        elif event.type == EventType.REMOVE:
            G.remove_edge(event.src, event.dest)
    return G


def draw_graph(tg: EventSequence):
    """
    Draws a graph from an event sequence
    """
    import networkx as nx
    G = to_nx_graph(tg)
    nx.draw(G)

def animate_graph(tg: EventSequence):
    """
    Animates a graph from an event sequence
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.Graph()
    fig, ax = plt.subplots()
    for event in tg:
        if event.type == EventType.ADD:
            G.add_edge(event.src, event.dest)
            nx.draw(G, ax=ax)
            plt.pause(0.1)
            ax.clear()


def write_to_file(sequence: EventSequence, file: str):
    """
    Writes an event sequence to a file
    """
    with open(file, "w") as f:
        first_line = True
        for event in sequence:
            if first_line:
                first_line = False
            else:
                f.write("\n")
            f.write("%s %s %d" % (event.src, event.dest, event.timestamp))

def write_to_binary_file(sequence: EventSequence, file: str):
    """
    Writes an event sequence to a binary file
    """
    with open(file, "wb") as f:
        for event in sequence:
            f.write(pack("iii", int(event.src), int(event.dest), event.timestamp))
            
if __name__ == "__main__":
    # sequence = ShuffledSequence(FileEventSequence("data/wiki-talks/wiki.txt"), 2)
    # sequence = ShuffledSequence(BarabasiAlbertEventSequence(), 10)
    # i = 0
    # for event in sequence:
    #     i += 1
    #     if i == 10:
    #         break
    #     print(event)
    
    def my_iter():
        for i in range(10):
            yield Event("a", "b", i)
    print("FromIterable")
    sequence = FromIterable(my_iter())
    for event in sequence:
        print(event)
    
    print("SBMSequence")
    sequence = SBMSequence([.5, .5], [[.5, .1], [.1, .5]])
    for event in FirstN(sequence, 10):
        print(event)
    sequence = SBMSequence([.5, .5], [[.5, .1], [.1, .5]])
    write_to_binary_file(FirstN(sequence, 10), "test.bin")

    print("BinaryEventSequence")
    for event in BinaryEventSequence("test.bin"):
        print(event)

    # import matplotlib.pyplot as plt
    # print("Drawing graph")
    # animate_graph(SBMSequence([.5, .5], [[.5, .1], [.1, .5]]))
    # draw_graph(FirstN(SBMSequence([.5, .5], [[.5, .1], [.1, .5]]), 10))
    # plt.show()

    # write_to_file(FirstN(SBMSequence([.5, .5], [
    #     [.5, .1], 
    #     [.1, .5]
    # ]), 10**10), "test.txt")
    # graph = to_nx_graph(FileEventSequence("test.txt"))
    # print("Graph of %d nodes" % graph.number_of_nodes())

