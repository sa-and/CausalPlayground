"""Defines the classes for the Gym environments that use an SCM"""

from typing import List, Callable, Tuple, NoReturn, Dict, Any, SupportsFloat
import networkx as nx
from gymnasium import Env
import numpy as np
from CausalPlayground import StructuralCausalModel
from pandas import DataFrame
from gymnasium import spaces


class SCMEnvironment(Env):
    """
    Defines a GYM environment over an SCM. Inherited from gymnasium.Env, please refer to
    https://gymnasium.farama.org/api/env/#gymnasium.Env for further documentation.
    """

    render_mode: str | None = None
    """Mode in which to render the environment. E.g.: 'human', 'terminal', None"""
    steps_this_episode: int
    """Number of steps that have been performed in the current episode."""
    current_action_indicies: List[int]
    """List of indices in the discrete action space that indicate the intervention in 'possible_interventions'."""
    current_interventions: List[Tuple[str, Tuple[Callable, Dict]]]
    """List of indices in the interventions corresponding to the current action indicies."""
    possible_interventions: List[Tuple[str, Tuple[Callable, Dict]]]
    """List of all possible interventions. Each intervention is represented by a tuple with the intervention target
    as first element and the callable + a dictionary mapping the parameters of the callable to the variables names as
    second element. The empty intervention $do(\emptyset)$ is always part of the possible interventions."""
    scm: StructuralCausalModel
    """SCM object that that represents the the data-generating process of the environment."""
    samples_so_far: DataFrame
    """Data frame containing all the samples that have been collected by this environment. The rows correspond to
    the targets in 'targets_so_far'."""
    targets_so_far: DataFrame
    """Data frame containing all intervention targets of the environment so far. The targets are encoded one-hot where
    1 means that the variable has been intervened on and 0 means it has not been intervened on. The rows correspond to
    the samples in 'samples_so_far'."""
    observation_space: spaces.Box
    """Observation space of the environment for a vector containing the values of the variables of the SCM."""
    action_space: spaces.Sequence
    """Action space of the environment for a list of discrete actions representing the indicies of the interventions 
    that are performed on the SCM"""

    def __init__(self, scm: StructuralCausalModel,
                 possible_interventions: List[Tuple[str, Tuple[Callable, Dict]]],
                 render_mode: str | None = None,
                 seed: int | None = None) -> None:
        """
        Constructor for SCM Gymnasium environment.

        :param scm: The SCM for the data-generating process of the environment
        :param possible_interventions: List of tuples where every tuple contains the name of the variable to intervene
        on and a callable that represents the new causal function for this variable given its parents as a dict. The
        dict maps the parameters of the callable (key) to the names of the parents in the SCM (value).
        :param render_mode: what way to render the environment. 'human' plots the causal graph + current values,
        'terminal' prints the current values of the variables.
        :param seed: seed for the random number generator.
        """
        super(SCMEnvironment, self).__init__()
        self.render_mode = render_mode
        self.steps_this_episode = 0
        self.current_action_indicies = []
        self.current_interventions = []
        self.scm = scm
        self.possible_interventions = [None]
        [self.add_possible_intervention(i) for i in possible_interventions]

        # create sample buffer that collects all samples
        self.samples_so_far = DataFrame(columns=list(self.scm.endogenous_vars.keys()) +
                                                list(self.scm.exogenous_vars.keys()))
        # create buffer that collects all intervention targets. 1 if intervened 0 if not
        self.targets_so_far = DataFrame(columns=list(self.scm.endogenous_vars.keys()))

        self.endogenous_vars = self.exogenous_vars = None

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(self.scm.endogenous_vars)+len(self.scm.exogenous_vars),))
        self.action_space = spaces.Sequence(spaces.Discrete(len(self.possible_interventions)))

        self.reset(seed=seed)

    def step(self, action: list[int]) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Performs an action in the environment and returns the effects. This corresponds to simultaneously applying a
        list of interventions to the SCM. New values are sampled from the SCM and reward, terminated and truncated flag
        are returned.

        :param action: List of intervention indicies according to the possible_interventions defined in the constructor
        :return: see https://gymnasium.farama.org/api/env/#gymnasium.Env
        """
        # Map action indicies to interventions
        self.current_interventions = [self.get_intervention_from_action(a) for a in action]

        # apply intervention
        self.scm.do_interventions(self.current_interventions)

        # sample the environment's SCM
        self.update_values_from_scm_sample()

        self.render()

        self.steps_this_episode += 1

        # task-specific quantities
        observation = self.get_observation()
        reward = self.determine_reward()
        truncated = self.determine_truncated()
        terminated = self.determine_terminated()

        self.scm.undo_interventions()

        if terminated or truncated:
            # reset environment if episode is done
            self.reset()

        return observation, reward, terminated, truncated, {}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) \
            -> tuple[np.ndarray, dict[str, Any]]:
        """
        Defines the reset operation to be done after each episode.

        :param seed: seed for the random number generator
        :param options: Additional information to specify how the environment is reset
        :return: The current observation and additional information.
        """
        super().reset(seed=seed, options=options)
        self.steps_this_episode = 0

        self.update_values_from_scm_sample()

        return self.get_observation(), {}

    def render(self) -> NoReturn:
        """
        Renders the environment depending on the mode specified at initialization. When 'render_mode' is set to 'human',
        the causal graph is drawn with corresponding values. When 'render_mode' is set to 'terminal', the current values
        of the SCM's variables are printed.
        """
        if self.render_mode == 'human':
            self.scm.draw_graph()

        elif self.render_mode == 'terminal':
            print(self.scm.endogenous_vars, self.scm.exogenous_vars)

    def update_values_from_scm_sample(self) -> NoReturn:
        """Samples the next sample from the SCM and stores it in the appropriate buffers."""
        # get the next sample
        self.endogenous_vars, self.exogenous_vars = self.scm.get_next_sample()
        # add it to the history of samples
        self.samples_so_far.loc[len(self.samples_so_far)] = dict(self.endogenous_vars, **self.exogenous_vars)
        # determine the intervention targets and one-hot encode them
        targets = self.scm.get_intervention_targets()
        targets_row = {k: 1 if k in targets else 0 for k in self.endogenous_vars.keys()}
        self.targets_so_far.loc[len(self.targets_so_far)] = targets_row

    def clear_samples(self) -> NoReturn:
        """Clears buffer with samples collected so far."""
        self.samples_so_far = DataFrame(columns=list(self.scm.endogenous_vars.keys()) +
                                                list(self.scm.exogenous_vars.keys()))

    def clear_intervention_targets(self) -> NoReturn:
        """Clears buffer with intervention targets collected so far."""
        self.targets_so_far = DataFrame(columns=list(self.scm.endogenous_vars.keys()))

    def get_intervention_from_action(self, action: int) -> Tuple[str, Tuple[Callable, Dict]]:
        """Returns the intervention given the index in the discrete action space."""
        return self.possible_interventions[action]

    def get_causal_structure(self) -> nx.DiGraph:
        """
        Returns the causal graph of the SCM.

        :return: nx.DiGraph that represents the causal graph.
        """
        return self.scm.create_graph()

    def add_possible_intervention(self, intervention: Tuple[str, Tuple[Callable, Dict]]) -> NoReturn:
        """
        Adds an intervention to the action space. Caution: this operation implies adding actions to the action space.

        :param intervention: the intervention that should be added to the possible interventions. The first element of
        the tuple is the intervention target and the second element is the callable function representing the causal
        assignment together with a dictionary mapping the parameters of the callable to the names of the variables.
        """
        assert intervention[0] in self.scm.endogenous_vars, (f"The intervention target {intervention[0]} is not an "
                                                             f"endogenous variable in the SCM.")
        self.possible_interventions.append(intervention)
        self.action_space = spaces.Sequence(spaces.Discrete(len(self.possible_interventions)))

    # Task-specific methods. Override in inheriting class to adapt behaviour
    def get_observation(self) -> np.ndarray:
        """
        Determines the observation of the current step and returns it.

        :return: the observation as a vector.
        """
        return np.array(dict(self.endogenous_vars, **self.exogenous_vars).values())

    def determine_reward(self) -> float:
        """
        Calculates the reward for the current step and returns it.

        :return: the reward of the current step
        """
        return 0.0

    def determine_truncated(self) -> bool:
        """
        Determines whether the current epsisode ends because of a condition outside of the MDP definition.

        :return: whether the current episode ends.
        """
        return False

    def determine_terminated(self) -> bool:
        """
        Determines whether a terminal state was reached within the MDP definition.

        :return: whether the current episode ends.
        """
        return False


class SCMReservoirEnvironment(Env):
    '''
    Defines a gymnasium environment over a set of SCMs.
    '''

    def __init__(self, scms: List[StructuralCausalModel],
                 possible_interventions: List[Tuple[str, Tuple[Callable, Dict]]],
                 render_mode: str | None = None,
                 seed: int | None = None) -> None):

        self.envs = []
        self.test_set = test_set
        self.train_set = train_set
        self.eval_func_type = eval_func_type
        self.agent_type = agent_type
        self.episode_length = episode_length
        self.n_vars = n_vars
        self.possible_functions = possible_functions
        self.gen = SCMGenerator()
        self.intervention_value = intervention_value
        self.mode = mode

        self.current_env = self.get_next_env(new_env=True)
        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

    def reset(self):
        # reset the current environment
        self.current_env.reset()

        # choose a random next environment and reset it
        self.current_env = self.get_next_env(new_env=True)
        return self.current_env.reset()

    def get_observation(self):
        return self.current_env.observation

    def get_causal_structure(self):
        return self.current_env.get_causal_structure()

    def set_reward(self, reward: float):
        self.current_env.set_reward(reward)

    def step(self, action):
        return self.current_env.step(action)

    def render(self, mode='human'):
        self.current_env.render(mode)

    def get_next_env(self, new_env: bool = True):
        """
        Samples a new environment in the reservoir if new_env is True. The environment is only returned if its graph
        structure is not in the test.

        :param new_env: If True, a new environment is samples. If False, the environment of the previous episode
        is returned
        """
        if not self.train_set is None:  # sample new environment from the train set if exists
            graph = random.choice(self.train_set)
            scm = self.gen.create_scm_from_graph(graph, possible_functions=[k for k in self.possible_functions])

        elif not self.test_set is None and self.train_set is None:  # if only test set is provided, sample a scm that
            # is not in the test set

            while True:  # resample scm until there is one that is not in the test set
                scm = self.gen.create_random(possible_functions=[k for k in self.possible_functions],
                                             n_endo=self.n_vars, n_exo=0)[0]
                graph = scm.create_graph()

                in_testset = False
                for i in range(len(self.test_set)):
                    edit_distance = nx.graph_edit_distance(self.test_set[i], graph)
                    if edit_distance == 0:
                        in_testset = True
                        break
                if not in_testset:
                    break

        else:  # self.test_set is None and self.train_set is None:  sample a completely new SCM with random structure
            scm = self.gen.create_random(possible_functions=[k for k in self.possible_functions],
                                             n_endo=self.n_vars, n_exo=0)[0]

        agent, _ = self._build_agent_eval_func()
        return SCMEnvironment(agent, self.episode_length, scm, mode=self.mode)