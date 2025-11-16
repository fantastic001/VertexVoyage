
import unittest

from vertex_voyage.dynnode2vec import DynNode2Vec
from vertex_voyage.temporal import FirstN, ForestFireEventSequence, Event 
from unittest.mock import MagicMock

class TestDynNode2Vec(unittest.TestCase):
    def test_construction(self):
        model = DynNode2Vec()
        self.assertIsInstance(model, DynNode2Vec)
    
    def test_event_sequence_integration(self):
        event_sequence = FirstN(ForestFireEventSequence(.1), 5)
        model = DynNode2Vec()
        nodes = set()
        event: Event
        for event in event_sequence:
            nodes.add(event.src)
            nodes.add(event.dest)
            model.update(event)
        embeddings = model.embed_nodes(list(nodes))
        self.assertEqual(len(embeddings), len(nodes))

    def test_update_method_calls_random_walks(self):
        model = DynNode2Vec()
        model._random_walk = MagicMock(side_effect=model._random_walk)
        events = [
            Event(src=1, dest=2, timestamp=0),
            Event(src=2, dest=3, timestamp=1),
            Event(src=3, dest=4, timestamp=2)
        ]
        for event in events:
            model.update(event)
            # for src and dest, we should have called _random_walk n_walks times each
            expected_calls = model.n_walks * 2
            self.assertEqual(model._random_walk.call_count, expected_calls)
            model._random_walk.reset_mock()

if __name__ == "__main__":
    unittest.main()