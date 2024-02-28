import CausalPlayground
from CausalPlayground import StructuralCausalModel
from CausalPlayground import SCMGenerator
import unittest
from functions import *
import random


class TestSCM(unittest.TestCase):
    def __init__(self, *args,  **kwargs):
        super(TestSCM, self).__init__(*args, **kwargs)
        self.scm = SCMGenerator(all_functions={'linear': f_linear}, seed=42).create_random(possible_functions=["linear"], n_endo=20, n_exo=0)[0]
        self.test_scm = StructuralCausalModel()
        self.test_scm.add_endogenous_var('A', lambda noise: noise + 5, {'noise': 'U'})
        self.test_scm.add_exogenous_var('U', random.randint, {'a': 3, 'b': 8})
        self.test_scm.add_endogenous_var('Effect', lambda x: x * 2, {'x': 'A'})

    def test_creation(self):
        sample = self.test_scm.get_next_sample()
        u = sample[1]['U']
        a = sample[0]['A']
        effect = sample[0]['EFFECT']
        self.assertIn(u, [3, 4, 5, 6, 7, 8])
        self.assertEqual(a, u+5)
        self.assertEqual(effect, a*2)

    def test_create_from_graph(self):
        graph = self.test_scm.create_graph()
        generator = SCMGenerator(all_functions={'linear': f_linear}, seed=1)
        scm = generator.create_scm_from_graph(graph=graph, possible_functions=['linear'],
                                              exo_distribution=random.randint, exo_distribution_kwargs={'a': 2, 'b': 5})
        self.assertTrue(scm.functions['A'][1] == {'U': 'U'})
        sample = scm.get_next_sample()
        u = sample[1]['U']
        a = sample[0]['A']
        effect = sample[0]['EFFECT']
        self.assertIn(u, [2, 3, 4, 5])
        self.assertTrue(effect <= 2*a)
    
    def test_intervention(self):
        # do an intervention and compare before and after
        x0 = self.scm.get_next_sample()[0]['X0']
        self.scm.do_interventions([("X0", (lambda: 5, {})), ("X1", (lambda x0: x0+1, {'x0': 'X0'}))])
        x0_do = self.scm.get_next_sample()[0]['X0']
        x1_do = self.scm.get_next_sample()[0]['X1']
        self.assertTrue(x0_do == 5)
        self.assertTrue(x1_do == 6)
        self.assertFalse(x0 == x0_do)

        # test graphical implications
        graph = self.scm.create_graph()
        # X1 should only have 1 parent
        self.assertEqual(len([e for e in graph.edges if e[1] == 'X1']), 1)
        # X0 should have no parents
        self.assertEqual(len([e for e in graph.edges if e[1] == 'X0']), 0)

        # test None - intervention
        self.scm.do_interventions([None, ("X1", (lambda x0: x0 + 1, {'x0': 'X0'}))])
        self.scm.undo_interventions()

    def test_undo_intervention(self):
        self.scm.do_interventions([("X0", (lambda: 5, {})), ("X1", (lambda x0: x0 + 1, {'x0': 'X0'}))])
        x1_do = self.scm.get_next_sample()[0]['X1']
        self.scm.undo_interventions()
        x1 = self.scm.get_next_sample()[0]['X1']
        self.assertFalse(x1_do == x1)

    def test_intervention_targets(self):
        """Are the correct intervention targets returned"""
        self.scm.do_interventions([("X0", (lambda: 5, {})), ("X1", (lambda x0: x0 + 1, {'x0': 'X0'}))])
        self.assertTrue(self.scm.get_intervention_targets() == ["X0", "X1"])
        self.scm.undo_interventions()
        self.assertTrue(self.scm.get_intervention_targets() == [])


class TestRandNN(unittest.TestCase):
    def __init__(self, *args,  **kwargs):
        super(TestRandNN, self).__init__(*args, **kwargs)
        self.scm1 = SCMGenerator(all_functions={'linear': f_linear}, seed=42).create_random(possible_functions=["linear"], n_endo=10, n_exo=0)[0]
        # self.scm2 = SCMGenerator(seed=42).create_random(possible_functions=["NN"], n_endo=6, n_exo=8,
        #                                                 exo_distribution=random.random, exo_distribution_kwargs={})[0]

    def test_sampling_types(self):
        vals = self.scm1.get_next_sample()[0]
        self.assertTrue(isinstance(vals['X0'], float))
        self.assertTrue(len(vals) == 10)

        # endo, exo = self.scm2.get_next_sample()
        # self.assertTrue(isinstance(endo[0], float))
        # self.assertTrue(len(endo) == 6)
        # self.assertTrue(isinstance(exo[0], float))
        # self.assertTrue(len(exo) == 8)

    def test_determinism(self):
        vals1 = self.scm1.get_next_sample()[0]
        vals2 = self.scm1.get_next_sample()[0]
        self.assertEqual(vals1, vals2)

        # vals1 = self.scm2.get_next_sample()[0]
        # vals2 = self.scm2.get_next_sample()[0]
        # self.assertNotEqual(vals1, vals2)


class TestExogenous(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestExogenous, self).__init__(*args, **kwargs)

        self.scm_confounded_controlled = StructuralCausalModel()

        self.generator = SCMGenerator(all_functions={'linear': f_linear}, seed=42)
        self.scm_unconfounded = self.generator.create_random(possible_functions=["linear"], n_endo=5, n_exo=4,
                                                             exo_distribution=random.random, exo_distribution_kwargs={},
                                                             allow_exo_confounders=False)[0]
        self.scm_confounded = self.generator.create_random(possible_functions=["linear"], n_endo=5, n_exo=7,
                                                           exo_distribution=random.random, exo_distribution_kwargs={},
                                                           allow_exo_confounders=True)[0]

    def test_confounded_create_graph(self):
        "does the graph generated from scm.create_graph for a scm with exo vars contain the exo vars"
        # unconfounded case
        graph_unconfounded = self.scm_unconfounded.create_graph()
        exo_edges = [e for e in graph_unconfounded.edges if e[0] in self.scm_unconfounded.exogenous_vars]
        self.assertFalse(len(exo_edges) == 0)
        self.assertTrue(len(set([e[0] for e in exo_edges])) == len([e[0] for e in exo_edges]))

        # confounded case
        graph_confounded = self.scm_confounded.create_graph()
        exo_edges = [e for e in graph_confounded.edges if e[0] in self.scm_confounded.exogenous_vars]
        self.assertFalse(len(exo_edges) == 0)
        self.assertFalse(len(set([e[0] for e in exo_edges])) == len([e[0] for e in exo_edges]))

    def test_sampling(self):
        "does the sampling work with exo vars"
        sample1 = self.scm_unconfounded.get_next_sample()
        sample2 = self.scm_unconfounded.get_next_sample()
        self.assertNotEqual(list(sample1[0].values()), list(sample2[0].values()))  # endogenous vars
        self.assertNotEqual(list(sample1[1].values()), list(sample2[1].values()))  # exoogenous vars


if __name__ == '__main__':
    unittest.main()
