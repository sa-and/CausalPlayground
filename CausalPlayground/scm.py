"""
This file defines all functionality for Structural Causal Models.
"""

from typing import Tuple, Any, Callable, Dict, List
import networkx as nx
import matplotlib.pyplot as plt
import random


class StructuralCausalModel:
    """
    Class defining a Structural Causal Model. We define an SCM as a tuple
    .. math::
        \mathcal{M} = (\mathcal{X}, \mathcal{F}, \mathcal{U}, \mathcal{P})
    where :math:`\mathcal{X}` is the set of endogenous variables, :math:`\mathcal{F}` is the set of assignment
    functions, :math:`\mathcal{U}` the set of exogenous variables, and :math:`\mathcal{P}` is the set of probability
    distributions associated with the exogenous variables.
    """

    endogenous_vars: Dict[str, Any]
    """Dictionary storing the value of each endogenoous variable."""
    functions: Dict[str, Tuple[Callable, dict]]
    """Functional assignments of the endogenous variables. for each endogenous variables, the functional assignments are
    stored as well as a dictionary mapping the parameters of the callable (key) to the name of the causes (values)."""
    exogenous_vars: Dict[str, Any]
    """Dictionary storing the values of the exogenous variables."""
    exogenous_distributions: Dict[str, Tuple[Callable, dict]]
    """Dictionary storing the distribution for each exogenous variable. The values of the dict contain the callable 
    representing the distribution and it's kwargs as a tuple."""
    saved_functions: Dict[str, Tuple[Callable, dict]]
    """Contains a backup of the function of each endogenous variable to be able to restore them after intervention."""
    def __init__(self):
        self.endogenous_vars = {}
        self.exogenous_vars = {}
        self.functions = {}
        self.exogenous_distributions = {}
        self.saved_functions = {}

    def add_endogenous_var(self, name: str, function: Callable, param_varnames: dict):
        """
        Adds an endogenous variable to the SCM.

        :param name: name of the endogenous variable
        :param function: callable that returns a value given the causes of the endogenous variables.
        :param param_varnames: dict that maps names of the parameters in the function to the name of the parent node.

        Example:
        >>> scm.add_endogenous_var('A', lambda noise: noise + 5, {'noise': 'U'})
        """
        # all names are uppercase
        name = name.upper()
        # ensure unique names
        assert name not in self.exogenous_vars.keys(), 'Variable already exists'
        assert name not in self.endogenous_vars.keys(), 'Variable already exists in endogenous vars'

        self.endogenous_vars[name] = None
        self.functions[name] = (function, param_varnames)

    def add_endogenous_vars(self, vars: List[Tuple[str, Callable, dict]]):
        """
        Adds a list of endogenous variables to the SCM.

        :param vars: list of endogenous variables definitions as defined in the 'add_endogenous_var' function.
        """
        [self.add_endogenous_var(v[0], v[1], v[2]) for v in vars]

    def add_exogenous_var(self, name: str, distribution: Callable, distribution_kwargs: dict):
        """
        Add an exogenous variable to the SCM.

        :param name: name of the exogenous variable.
        :param distribution: distribution of the exogenous variable.
        :param distribution_kwargs: kwargs for the distribution

        Example:
        >>> scm.add_exogenous_var('U', random.randint, {'a': 3, 'b': 8})
        """
        # all names are uppercase
        name = name.upper()
        # ensure unique names
        assert name not in self.exogenous_vars.keys(), 'Variable already exists'
        assert name not in self.endogenous_vars.keys(), 'Variable already exists in endogenous vars'

        self.exogenous_vars[name] = None
        self.exogenous_distributions[name] = (distribution, distribution_kwargs)

    def add_exogenous_vars(self, vars: List[Tuple[str, Any, Callable, dict]]):
        """
        Adds a list of exogenous variables to the SCM.

        :param vars: list of exogenous variables definitions as defined in the 'add_exogenous_var' function.
        """
        [self.add_exogenous_var(v[0], v[1], v[2]) for v in vars]

    def get_next_sample(self) -> Tuple[Dict, Dict]:
        """
        Generates an ancestral sample from the joint distribution of endogenous variables by generating a sample of the
        exogenous variables
        .. math::
            u \sim P(\mathcal{U})
        and determining the value of variables :math:`\mathcal{X}` by applying the functions in :math:`\mathcal{F}`
        according to the topological ordering in the causal graph resulting in a sample of the endogenous variables
        .. math::
            x \sim P(\mathcal{X})


        :return: a sample of endogenous :math:`x` and exogenous variables :math:`u`.
        """
        random.seed()
        # update exogenous vars
        for key in self.exogenous_vars:
            dist = self.exogenous_distributions[key]
            res = dist[0](**dist[1])
            self.exogenous_vars[key] = res

        # update endogenous vars
        structure_model = self.create_graph()
        node_order = [n for n in nx.topological_sort(structure_model)]
        # remove exogenous vars since they are root nodes and already have a value
        [node_order.remove(n) for n in self.exogenous_vars]
        # propagate causal effects along the topological ordering
        for node in node_order:
            # get the values for the parameters needed in the functions
            params = {}
            for param in self.functions[node][1]:  # parameters of functions
                if self.functions[node][1][param] in self.endogenous_vars.keys():
                    params[param] = self.endogenous_vars[self.functions[node][1][param]]
                else:
                    params[param] = self.exogenous_vars[self.functions[node][1][param]]

            # Update variable according to its function and parameters
            self.endogenous_vars[node] = self.functions[node][0](**params)
        return dict(self.endogenous_vars), dict(self.exogenous_vars)

    def do_interventions(self, interventions: List[Tuple[str, Tuple[Callable, dict]]]):
        """
        Replaces the functions of the scm with the given interventions per endogenous variable. E.g. the intervention
        :math:`do(X_0 = 5, X_1 = X_0+1)` can be implemented with
        >>> scm.do_interventions([("X0", (lambda: 5, {})), ("X1", (lambda x0: x0+1, {'X0':'x0'})])

        :param interventions: List of tuples where every tuple contains the name of the variable to intervene on and a
        callable that represents the new causal function for this variable given its parents as a dict. The dict maps
        the parameters of the callable (key) to the names of the parents in the SCM (value).
        """
        random.seed()
        self.saved_functions = {}
        for interv in interventions:
            if interv:  # interv not None
                self.saved_functions[interv[0]] = self.functions[interv[0]]
                self.functions[interv[0]] = interv[1]

    def undo_interventions(self):
        """
        Restores all functional relations that were deleted in the previous call of `do_interventions`.
        """
        for key, value in self.saved_functions.items():
            self.functions[key] = value
        self.saved_functions.clear()

    def get_intervention_targets(self) -> List[str]:
        """
        Returns a list containing the names of the variables that are currently being intervened on.

        :return: List of intervention targets.
        """
        return list(self.saved_functions.keys())

    def create_graph(self) -> nx.DiGraph:
        """
        Returns the DAG that corresponds to the functional structure of this SCM.

        :return: A causal graph.
        """
        graph = nx.DiGraph()

        # create nodes
        [graph.add_node(var.upper(), type='endo') for var in self.endogenous_vars]
        [graph.add_node(var.upper(), type='exo') for var in self.exogenous_vars]

        for var in self.functions:
            for parent in self.functions[var][1].values():
                if parent.lower() in self.endogenous_vars or parent.upper() in self.endogenous_vars\
                        or parent.lower() in self.exogenous_vars or parent.upper() in self.exogenous_vars:
                    graph.add_edge(parent.upper(), var.upper())

        return graph

    def draw_graph(self):
        """
        Draws the causal graph.
        """
        graph = self.create_graph()
        values = dict(self.endogenous_vars), dict(self.exogenous_vars)
        values = dict(values[0], **values[1])
        nx.draw(graph, arrowsize=20, with_labels=True, node_size=3000, font_size=10,
                labels={key: str(key) + ':\n' + str(values[key]) for key in values}, pos=nx.planar_layout(graph))
        plt.show()

