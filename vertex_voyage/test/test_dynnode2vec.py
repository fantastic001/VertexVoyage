
from functools import wraps
import unittest
from unittest import mock

import gensim
from networkx import karate_club_graph

from vertex_voyage.dynnode2vec import DynNode2Vec
from vertex_voyage.reconstruction import reconstruct, get_f1_score
from vertex_voyage.temporal import (
    FirstN, 
    ForestFireEventSequence, 
    Event, 
    buffered
)
from unittest.mock import MagicMock
from vertex_voyage.word2vec import word2vec

import numpy as np

class TestDynNode2Vec(unittest.TestCase):
    def test_construction(self):
        model = DynNode2Vec()
        self.assertIsInstance(model, DynNode2Vec)
    
    def test_event_sequence_integration(self):
        event_sequence = FirstN(ForestFireEventSequence(.1), 5)
        model = DynNode2Vec()
        nodes = set()
        event: Event
        for buffer in buffered(event_sequence, buffer_size=2):
            for event in buffer:
                nodes.add(event.src)
                nodes.add(event.dest)
            model.update(buffer)
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
        for buf in buffered(events, buffer_size=1):
            model.update(buf)
            # for src and dest, we should have called _random_walk n_walks times each
            expected_calls = model.n_walks * 2 if len(model.G.nodes()) >= model.retrain_threshold else len(model.G.nodes()) * model.n_walks
            self.assertEqual(model._random_walk.call_count, expected_calls)
            model._random_walk.reset_mock()
    
    @mock.patch('vertex_voyage.word2vec.word2vec')
    @mock.patch('gensim.models.Word2Vec')
    def test_if_word2vec_called_on_previous_models(self, mock_gensim_word2vec, mock_word2vec):
        model = DynNode2Vec()
        mock_word2vec.side_effect = word2vec
        M = gensim.models.Word2Vec(vector_size=model.dim, min_count=1)
        M.build_vocab([[0, 1, 2], [1, 2, 3]])
        M.train([[0, 1, 2], [1, 2, 3]], total_examples=2, epochs=1, min_count=0, null_word=-1)

        mock_gensim_word2vec.return_value = M
        # We called gensim to create artificial return value. We do not 
        # want it to be counted as a call to gensim in the first update, so we reset it here.
        mock_gensim_word2vec.reset_mock()
        model.update_model = MagicMock(side_effect=model.update_model)
        events = [
            Event(src=1, dest=2, timestamp=0),
            Event(src=2, dest=3, timestamp=1),
            Event(src=3, dest=4, timestamp=2)
        ]
        model.embed_node = MagicMock(return_value=np.zeros(model.dim))
        for i, buf in enumerate(buffered(events, buffer_size=1)):
            model.update(buf)
            # either gensim Word2Vec or our custom word2vec should have been called
            retrain = len(model.G.nodes()) < model.retrain_threshold or i == 0
            self.assertTrue(
                (not retrain and not mock_gensim_word2vec.called) or mock_word2vec.called or 
                (retrain and mock_gensim_word2vec.called))
            if not retrain:
                model.update_model.assert_called_once()
                # word2vec should have been called with old_model not None
                if mock_word2vec.called:
                    args, kwargs = mock_word2vec.call_args
                    self.assertIsNotNone(kwargs.get('old_model', None))
            else:
                model.update_model.assert_called_once_with(
                    mock.ANY, 
                    None
                )
                # word2vec should have been called where old_model is None
                if mock_word2vec.called:
                    args, kwargs = mock_word2vec.call_args
                    self.assertIsNone(kwargs.get('old_model', None))
                else:
                    mock_gensim_word2vec.assert_called_once_with(
                        vector_size=model.dim,
                        window=model.window_size,
                        min_count=0,
                        sg=1,
                        null_word = -1,
                        negative=model.negative_sample_num,
                        alpha=model.learning_rate,
                        epochs=model.epochs,
                        seed=mock.ANY,
                        shrink_windows=False,
                        workers=mock.ANY
                    )
            model.update_model.reset_mock()
            mock_word2vec.reset_mock()
            mock_gensim_word2vec.reset_mock()
    
    def test_zacharys_karate_club(self):
        G = karate_club_graph()
        model = DynNode2Vec(
            dim=100, 
            walk_size=80, 
            n_walks=10, 
            window_size=10,
            epochs=10, 
            p = .25,
            q = 4,
            negative_sample_num=1, # in practice, should be 500
            seed=42,
            learning_rate=0.01,
            use_threads=False
        )
        t = 0
        for u, v in G.edges():
            event = Event(src=u, dest=v, timestamp=t)
            model.update([event])
            t += 1
        embeddings = model.embed_nodes(list(G.nodes()))
        self.assertEqual(len(embeddings), len(G.nodes()))
        reconstructed = reconstruct(
            G.number_of_edges(), 
            embeddings, 
            list(G.nodes())
        )
        f1 = get_f1_score(G, reconstructed)
        self.assertGreater(f1, 0.5)
    
    def test_zacharys_karate_club_buffered(self):
        G = karate_club_graph()
        model = DynNode2Vec(
            dim=100, 
            walk_size=80, 
            n_walks=10, 
            window_size=10,
            epochs=10, 
            p = .25,
            q = 4,
            negative_sample_num=1, # in practice, should be 500
            seed=42,
            learning_rate=0.01,
            use_threads=False
        )
        t = 0
        events = []
        for u, v in G.edges():
            event = Event(src=u, dest=v, timestamp=t)
            events.append(event)
            t += 1
        for buffer in buffered(events, buffer_size=5):
            model.update(buffer)
        embeddings = model.embed_nodes(list(G.nodes()))
        self.assertEqual(len(embeddings), len(G.nodes()))
        reconstructed = reconstruct(
            G.number_of_edges(), 
            embeddings, 
            list(G.nodes())
        )
        f1 = get_f1_score(G, reconstructed)
        self.assertGreater(f1, 0.5)

if __name__ == "__main__":
    unittest.main()

