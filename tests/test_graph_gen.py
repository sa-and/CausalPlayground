import unittest
import numpy as np
from CausalPlayground.generators import CausalGraphGenerator, CausalGraphSetGenerator
import networkx as nx
import os


class TestGraphGeneration(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGraphGeneration, self).__init__(*args, **kwargs)
        self.sizes = [3, 4, 10, 15]
        self.generators = {n: CausalGraphGenerator(n_endo=n, n_exo=n-3) for n in self.sizes}
        self.unconfounded_test_graph = CausalGraphGenerator(n_endo=5,
                                                            n_exo=10,
                                                            allow_exo_confounders=False).generate_random_graph()[0]
        self.confounded_test_graph = CausalGraphGenerator(n_endo=10,
                                                          n_exo=100,
                                                          allow_exo_confounders=True).generate_random_graph()[0]
        
    def test_type(self):
        self.assertEqual(type(self.unconfounded_test_graph), nx.DiGraph)

    def test_n_nodes(self):
        [self.assertEqual(len(self.generators[n].generate_random_graph()[0].nodes), 2*n-3) for n in self.sizes]

    def test_acyclyc(self):
        [self.assertTrue(nx.is_directed_acyclic_graph(self.generators[n].generate_random_graph()[0])) for n in self.sizes]

    def test_endo_exo(self):
        for node in self.unconfounded_test_graph.nodes:
            self.assertIn(self.confounded_test_graph.nodes[node]['type'], ['endo', 'exo'])

    def test_exo_root(self):
        '''
        Test whether exogenous variables are roots of the graph
        '''
        for graph in [self.confounded_test_graph, self.unconfounded_test_graph]:
            for node in graph.nodes:
                if graph.nodes[node]['type'] == 'exo':
                    self.assertIn(len([i for i in graph.predecessors(node)]), [0, 1])

    def test_confounding(self):
        for node in self.unconfounded_test_graph.nodes:
            if self.unconfounded_test_graph.nodes[node]['type'] == 'exo':
                self.assertIn(len([n for n in self.unconfounded_test_graph.successors(node)]), [0, 1])

    def test_n_generated_graphs(self):
        set_generator = CausalGraphSetGenerator(n_endo=5, n_exo=10, allow_exo_confounders=False)
        set_generator.generate(100)
        self.assertEqual(len(set_generator.graphs), 99)
        
    def test_seeding(self):
        set_generator1 = CausalGraphSetGenerator(n_endo=5, n_exo=5, allow_exo_confounders=False, seed=1)
        set_generator2 = CausalGraphSetGenerator(n_endo=5, n_exo=5, allow_exo_confounders=False, seed=50)
        set_generator3 = CausalGraphSetGenerator(n_endo=5, n_exo=5, allow_exo_confounders=False, seed=1)
        set_generator1.generate(50)
        set_generator2.generate(50)
        set_generator3.generate(50)
        
        for i in range(49):
            self.assertTrue(nx.is_isomorphic(set_generator1.graphs[i], set_generator3.graphs[i]))
            self.assertFalse(nx.is_isomorphic(set_generator1.graphs[i], set_generator2.graphs[i]))  # holds just for
            # these specific seeds and 50 samples

    def test_graphset_save_load(self):
        graph_set = CausalGraphSetGenerator(n_endo=5, n_exo=10, allow_exo_confounders=False)
        graph_set.generate(20)
        graph_set.save('delme.pkl')
        graph_set.load('delme.pkl')
        self.assertTrue(len(graph_set.graphs), 20)
        os.remove('delme.pkl')
        
    def test_unique_graph_set(self):
        graph_set = CausalGraphSetGenerator(n_endo=4, n_exo=0, allow_exo_confounders=False)
        graph_set.generate(300)
        edges = [list(g.edges()) for g in graph_set.graphs]
        self.assertTrue(all([len(e) == len(set(e)) for e in edges]))
        graph_set.generate(300, method='ER', p=0.3)
        edges = [list(g.edges()) for g in graph_set.graphs]
        self.assertTrue(all([len(e) == len(set(e)) for e in edges]))

    def test_save_load_graph_set(self):
        graph_set = CausalGraphSetGenerator(n_endo=4, n_exo=0, allow_exo_confounders=False)
        graph_set.generate(300)
        graph_set.save('delme.pkl')
        graph_set.load('delme.pkl')
        np.array([1, 2]).dump('delme.pkl')
        with self.assertRaises(TypeError):
            graph_set.load('delme.pkl')
        

if __name__ == '__main__':
    unittest.main()
