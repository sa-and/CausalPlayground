from CausalPlayground import SCMEnvironment, StructuralCausalModel
import random


class NoIntervEnv(SCMEnvironment):
    def __init__(self, scm, possible_interventions):
        super(NoIntervEnv, self).__init__(scm, possible_interventions)

    def determine_reward(self):
        if len(self.scm.get_intervention_targets()) > 0:
            return -1
        else:
            return 0


if __name__ == '__main__':
    scm = StructuralCausalModel()
    scm.add_endogenous_var('A', lambda noise: noise + 5, {'noise': 'U'})
    scm.add_exogenous_var('U', random.randint, {'a': 3, 'b': 8})
    scm.add_endogenous_var('EFFECT', lambda x: x * 2, {'x': 'A'})
    env = NoIntervEnv(scm, possible_interventions=[("A", (lambda: 5, {})), ("EFFECT", (lambda x0: x0+1, {'x0': 'A'}))])

    for s in range(10):
        action_sample = env.action_space.sample()
        obs, rew, _, _, _ = env.step(action_sample)
        print(action_sample, obs, rew)