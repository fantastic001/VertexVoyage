
import random
import unittest
from collections import Counter

from vertex_voyage.temporal import Event, FromIterable
from vertex_voyage.temporal_ordering import BFSOrdering, DFSOrdering, RandomOrdering


def event_key(event: Event):
    return (event.src, event.dest, event.timestamp)


class TestBFSOrdering(unittest.TestCase):
    def test_bfs_visits_breadth_first(self):
        events = [
            Event(1, 2, 0),
            Event(1, 3, 1),
            Event(2, 4, 2),
            Event(1, 5, 3),
        ]
        ordering = BFSOrdering(FromIterable(iter(events)))
        result = [(e.src, e.dest) for e in ordering]
        self.assertEqual(result, [(1, 2), (1, 3), (1, 5), (2, 4)])

    def test_bfs_preserves_all_events(self):
        events = [
            Event(1, 2, 0),
            Event(1, 3, 1),
            Event(2, 4, 2),
            Event(3, 4, 3),
        ]
        ordering = BFSOrdering(FromIterable(iter(events)))
        self.assertEqual(
            Counter(event_key(e) for e in ordering),
            Counter(event_key(e) for e in events),
        )

    def test_bfs_disconnected_components_ordered_by_first_seen(self):
        events = [
            Event(1, 2, 0),
            Event(3, 4, 1),
        ]
        ordering = BFSOrdering(FromIterable(iter(events)))
        result = [(e.src, e.dest) for e in ordering]
        self.assertEqual(result, [(1, 2), (3, 4)])

    def test_bfs_empty_sequence(self):
        ordering = BFSOrdering(FromIterable(iter([])))
        self.assertEqual(list(ordering), [])


class TestDFSOrdering(unittest.TestCase):
    def test_dfs_visits_depth_first(self):
        events = [
            Event(1, 2, 0),
            Event(1, 3, 1),
            Event(2, 4, 2),
            Event(1, 5, 3),
        ]
        ordering = DFSOrdering(FromIterable(iter(events)))
        result = [(e.src, e.dest) for e in ordering]
        self.assertEqual(result, [(1, 2), (2, 4), (1, 3), (1, 5)])

    def test_dfs_preserves_all_events(self):
        events = [
            Event(1, 2, 0),
            Event(1, 3, 1),
            Event(2, 4, 2),
            Event(3, 4, 3),
        ]
        ordering = DFSOrdering(FromIterable(iter(events)))
        self.assertEqual(
            Counter(event_key(e) for e in ordering),
            Counter(event_key(e) for e in events),
        )

    def test_dfs_disconnected_components_ordered_by_first_seen(self):
        events = [
            Event(1, 2, 0),
            Event(3, 4, 1),
        ]
        ordering = DFSOrdering(FromIterable(iter(events)))
        result = [(e.src, e.dest) for e in ordering]
        self.assertEqual(result, [(1, 2), (3, 4)])

    def test_dfs_empty_sequence(self):
        ordering = DFSOrdering(FromIterable(iter([])))
        self.assertEqual(list(ordering), [])


class TestRandomOrdering(unittest.TestCase):
    def test_random_preserves_all_events(self):
        events = [Event(i, i + 1, i) for i in range(20)]
        ordering = RandomOrdering(FromIterable(iter(events)))
        self.assertEqual(
            Counter(event_key(e) for e in ordering),
            Counter(event_key(e) for e in events),
        )

    def test_random_changes_order(self):
        events = [Event(i, i + 1, i) for i in range(20)]

        random.seed(0)
        ordering = RandomOrdering(FromIterable(iter(events)))
        result = [event_key(e) for e in ordering]

        original = [event_key(e) for e in events]
        self.assertNotEqual(result, original)

    def test_random_empty_sequence(self):
        ordering = RandomOrdering(FromIterable(iter([])))
        self.assertEqual(list(ordering), [])


if __name__ == "__main__":
    unittest.main()
