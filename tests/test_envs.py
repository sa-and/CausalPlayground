import unittest
from CausalPlayground import SCMEnvironment
from CausalPlayground import StructuralCausalModel
import random


class TestSCMEnvironment(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSCMEnvironment, self).__init__(*args, **kwargs)
        self.test_scm = StructuralCausalModel()
        self.test_scm.add_exogenous_var('U', random.randint, {'a': 3, 'b': 8})
        self.test_scm.add_endogenous_var('A', lambda noise: noise + 5, {'noise': 'U'})
        self.test_scm.add_endogenous_var('B', lambda: 1, {})
        self.test_scm.add_endogenous_var('Effect', lambda x, y: x * 2 + y, {'x': 'A', 'y': 'B'})

        self.env = SCMEnvironment(scm=self.test_scm,
                                  possible_interventions=[("A", (lambda: 5, {})),
                                                          ("B", (lambda x0: x0 + 1, {'x0': 'A'}))],
                                  render_mode='human')

    def test_possible_interventions(self):
        self.assertTrue(self.env.action_space.feature_space.n == 3)
        mapped_sample = [self.env.get_intervention_from_action(a) for a in self.env.action_space.sample((5, None))]
        self.assertTrue(all([s in self.env.possible_interventions for s in mapped_sample]))

    def test_steps(self):
        for step in range(5):
            action = self.env.action_space.sample()
            print(action)
            self.env.step(action)

    def test_buffers(self):
        """Test wether the buffers for the histories of samples are correct"""
        self.env.update_values_from_scm_sample()
        self.env.clear_samples()
        self.env.clear_intervention_targets()
        self.assertTrue(len(self.env.samples_so_far) == 0)
        self.assertTrue(len(self.env.targets_so_far) == 0)

        self.env.scm.do_interventions([("A", (lambda: 5, {})), ("B", (lambda x0: x0 + 1, {'x0': 'A'}))])
        self.env.update_values_from_scm_sample()
        self.assertTrue(len(self.env.samples_so_far) == 1)
        self.assertTrue(len(self.env.targets_so_far) == 1)
        self.assertDictEqual(self.env.targets_so_far.iloc[-1].to_dict(), {'A': 1, 'B': 1, 'EFFECT': 0})

        self.env.scm.undo_interventions()
        self.env.update_values_from_scm_sample()
        self.assertDictEqual(self.env.targets_so_far.iloc[-1].to_dict(), {'A': 0, 'B': 0, 'EFFECT': 0})

    def test_rendering(self):
        for _ in range(3):
            self.env.scm.get_next_sample()
            self.env.render()
            self.env.render_mode = 'terminal'
            self.env.render()

    def test_add_intervention(self):
        self.assertEquals(len(self.env.possible_interventions), 3)
        self.env.add_possible_intervention(('EFFECT', (lambda x: 3*x, {'x': 'U'})))
        self.assertEquals(len(self.env.possible_interventions), 4)
        with self.assertRaises(AssertionError):
            self.env.add_possible_intervention(('X', (lambda x: 3 * x, {'x': 'U'})))  # X is not part of the SCM
        self.assertTrue(any([i[0] == 'EFFECT' for i in self.env.possible_interventions[1:]]))  # Ignor None intervention


if __name__ == '__main__':
    unittest.main()
