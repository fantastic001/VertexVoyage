
import unittest 
from vertex_voyage.grid_search import grid_search
class TestGridSearchCV(unittest.TestCase):
    def test_simple_case(self):
        f = lambda x: x**2
        plus = lambda x, y: x + y
        identity = lambda x: x
        result = grid_search(f, identity, plus, param_ranges={'x': [1, 2]})
        self.assertEqual(result, 5)
    
    def test_intermediate_callback(self):
        f = lambda x: x**2
        plus = lambda x, y: x + y
        identity = lambda x: x
        calls = []
        def callback(result, **params):
            calls.append((result, params['x']))
        result = grid_search(f, identity, plus, param_ranges={'x': [1, 2]}, intermediate_callback=callback)
        self.assertEqual(result, 5)
        self.assertEqual(calls, [(1, 1), (4, 2)])

    def test_fixed_params(self):
        f = lambda x, y: x + y
        plus = lambda x, y: x + y
        identity = lambda x: x
        result = grid_search(f, identity, plus, param_ranges={'x': [1, 2]}, fixed_params={'y': 10})
        self.assertEqual(result, 23)

if __name__ == '__main__':
    unittest.main()