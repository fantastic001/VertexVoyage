
from functools import wraps
import unittest
from unittest import mock

from vertex_voyage import model
from vertex_voyage.dynnode2vec import DynNode2Vec
from vertex_voyage.temporal import FirstN, ForestFireEventSequence, Event 
from unittest.mock import MagicMock
from vertex_voyage.word2vec import word2vec

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
    
    @mock.patch('vertex_voyage.word2vec.word2vec')
    def test_if_word2vec_called_on_previous_models(self, mock_word2vec):
        model = DynNode2Vec()
        mock_word2vec.side_effect = word2vec
        model.update_model = MagicMock(side_effect=model.update_model)
        events = [
            Event(src=1, dest=2, timestamp=0),
            Event(src=2, dest=3, timestamp=1),
            Event(src=3, dest=4, timestamp=2)
        ]
        for i, event in enumerate(events):
            model.update(event)
            mock_word2vec.assert_called_once()
            if i > 0:
                model.update_model.assert_called_once()
                # word2vec should have been called with old_model not None
                args, kwargs = mock_word2vec.call_args
                self.assertIsNotNone(kwargs.get('old_model', None))
            else:
                model.update_model.assert_called_once_with(
                    mock.ANY, 
                    None
                )
                # word2vec should have been called where old_model is None
                args, kwargs = mock_word2vec.call_args
                self.assertIsNone(kwargs.get('old_model', None))
            model.update_model.reset_mock()
            mock_word2vec.reset_mock()

if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()