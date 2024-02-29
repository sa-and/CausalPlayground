"""
# Overview
The [CausalPlayground](https://github.com/sa-and/CausalPlayground) library serves as a tool for causality research,
focusing on the interactive exploration of structural
causal models (SCMs). It provides extensive functionality for creating, manipulating and sampling SCMs, seamlessly
integrating them with the Gymnasium framework. Users have complete control over SCMs, enabling precise manipulation and
interaction with causal mechanisms. Additionally, CausalPlayground offers a range of useful helper functions for generating
diverse instances of SCMs and DAGs, facilitating quantitative experimentation and evaluation. Notably, the library is
designed for easy integration with reinforcement learning methods, enhancing its utility in active inference and
learning settings. This documentation presents the complete API documentation and a quickstart guide [here](https://sa-and.github.io/CausalPlayground/).

# Installation guide (TODO)
Either clone and 'pip install -r requirements.txt'

or 'pip install CausalPlayground'

# Structural Causal Models (SCM)
SCMs are a powerful model to express a data-generating process that is governed by clear causal relations represented
as functions. More formally, an SCM models the causal relationship between endogenous variables, describing the main
random variables of intrerest, and exogenous variables that, roughly speaking, model the context of the process. The
causal effects are described as functional relations between variables, and the exogenous variables are modeled through
distributions. Taken together, an SCM defines the joint probability distribution over its variables. CausalPlayground
facilitates the creation, manipulation, and sampling of SCM as described below.

## Defining an SCM
Define the SCM $\mathcal{M} = (\mathcal{X}, \mathcal{U}, \mathcal{F}, \mathcal{P})$ with
$\mathcal{X}= \\{ A, Effect \\}$, $\mathcal{U}=\\{U\\}$, $\mathcal{P}=\\{Uniform(3, 8)\\}$, and
$\mathcal{F} = \begin{cases}A\leftarrow 5+U\\\Effect \leftarrow A*2\end{cases}$

```Python
>>> scm = StructuralCausalModel()
>>> scm.add_endogenous_var('A', lambda noise: noise+5, {'noise': 'U'})
>>> scm.add_exogenous_var('U', random.randint, {'a': 3, 'b': 8})
>>> scm.add_endogenous_var('Effect', lambda x: x*2, {'x': 'A'})
```

The functions can be defined as lambda functions. Alternatively, any callable function is admissible. The given
dictionary maps the parameters in the callable function to the names of the variables. Note that the variable names will
always be converted to upper-case.

## Sampling the SCM
```Python
>>> x, u = scm.get_next_sample()
>>> x
{'A': 10, 'EFFECT': 20}
>>> u
{'U': 5}
```

## Intervening in an SCM
To perform an intervention $do(X_0 \leftarrow 5, X_1 \leftarrow X_0+1)$, the following can be implemented:
```Python
    >>> scm.do_interventions([("X0", (lambda: 5, {})),
                              ("X1", (lambda x0: x0+1, {'x0':'X0'})])
```
As before, the new function describing the causal relations of a node to its parents, is represented by anny callable
function and the dictionary maps the parameters of that function to the names of the variables. Calling
```Python
    >>> scm.undo_interventions()
```
will return the SCM to its original state with $do(\emptyset)$.

# SCM Environments
## Defining an SCM Environment
When building agents, that interact with an SCM you can use the `CausalPlayground.SCMEnvironment` class.
```Python
>>> from CausalPlayground import SCMEnvironment
>>> env = SCMEnvironment(scm, possible_interventions=[("X0", (lambda: 5, {})), ("X1", (lambda x0: x0+1, {'x0':'X0'}))])
```
The actions in this environment correspond to the interventions defined in the `possible_interventions`.

Calling the `step(action)` function applies the interventions defined in `action` (see explaination below), samples the
intervened SCM, determines the new observation, termination flag, truncated flag and reward, and, finally undoes the
interventions.

To enable multiple interventions simultaneously, a list of interventions can be passed as action. The elements in this
list, correspond to the discrete index of that intervention in the list `possible_interventions`. For example,
simultaneously performing both possible interventions in one step can be done as follows:
```Python
>>> env.step([1, 2])
```
whereas performing a step without intervention can be done by either invoking the first (empty) intervention, or no
intervention as follows:
```Python
>>> env.step([0])
>>> env.step([])
```

## Generating data from the SCM
In its basic implementation, `SCMEnvironment` always returns the current values of the variables as observation and 0
reward regardless of the action. Furthermore, the `terminated` and `truncated` flag are always False, leading to an
episode never terminating.

Consider the following SCM with given possible interventions:
```Python
>>> scm = StructuralCausalModel()
>>> scm.add_endogenous_var('A', lambda noise: noise + 5, {'noise': 'U'})
>>> scm.add_exogenous_var('U', random.randint, {'a': 3, 'b': 8})
>>> scm.add_endogenous_var('EFFECT', lambda x: x * 2, {'x': 'A'})
```
The basic implementation can be used to generate samples (e.g. 1000) from an SCM either with no interventions
(observational data):
```Python
>>> for i in range(1000):
>>>     env.step([0])
```
with some fixed interventions:
```Python
>>> for i in range(1000):
>>>     env.step([1, 2])
```
or even with random interventions:
```Python
>>> for i in range(1000):
>>>     env.step(env.action_space.sample())
```

The collected data can be found in `env.samples_so_far` and the intervention targets in `env.targets_so_far`. These
buffers can be flushed on demand with `clear_samples()` and `clear_intervention_targets()`, respectively.

## Extending SCMEnvironment
While the basic implementation is useful for collecting data generated by an SCM, it holds potential for more sophisticated applications to
reinforcement learning by inheriting from `SCMEnvironment`. To adapt the environment to a specific task, simply override
the neccessary functions determining observation, reward, truncated, and terminated. (Note that the interventions in a
step are undone *after* these quantities are determined.)

Here an example for an environment that rewards interventions:
```Python
>>> from CausalPlayground import SCMEnvironment

>>> class NoIntervEnv(SCMEnvironment):
        def __init__(self, scm, possible_interventions):
            super(NoIntervEnv, self).__init__(scm, possible_interventions)

        def determine_reward(self):
            if len(self.scm.get_intervention_targets()) > 0:
                return 1
            else:
                return 0
>>> env = NoIntervEnv(scm, possible_interventions=[("A", (lambda: 5, {})), ("EFFECT", (lambda x0: x0+1, {'x0': 'A'}))])
```

# Generators
In some scenarios it might be useful to automatically generate graphs and SCMs. To this end, we implemented some helper
classes.

## Graph generation
You can use the `CausalPlayground.CausalGraphGenerator` class to quickly generate directed acyclic graphs. To generate a random
graph with 7 endogenous and 5 exogenous variables, you can run the following:
```Python
>>> from CausalPlayground import CausalGraphGenerator
>>> gen = CausalGraphGenerator(7, 5)
>>> graph = gen.generate_random_graph()
```
This creates graphs with no exogenous confounders. For allowing exogenous confounders, instantiate the generator as
```Python
>>> gen = CausalGraphGenerator(7, 5, allow_exo_confounders=True)
```

With the `CausalPlayground.CausalGraphSetGenerator` class, you can generate many DAGs at once. Additionally, this class makes
sure that each graph is contained only ones in the list of generated graphs.
```Python
>>> from CausalPlayground import CausalGraphSetGenerator
>>> gen = CausalGraphSetGenerator(7, 5)
>>> gen.generate(1000)
```
generates 1000 distinct DAGs with 7 endogenous nodes and 5 exogenous nodes that can be accessed through `gen.graphs`.

## SCM generation
The `CausalPlayground.SCMGenerator` class also let's you generate SCMs automatically, for example with linear additive
causal relations and a random causal structure.
```Python
>>> from tests.functions import f_linear
>>> from CausalPlayground import SCMGenerator

>>> gen = SCMGenerator(all_functions={'linear': f_linear})
>>> scm_unconfounded = gen.create_random(possible_functions=["linear"], n_endo=5, n_exo=4,
                                     exo_distribution=random.random, exo_distribution_kwargs={},
                                     allow_exo_confounders=False)[0]
>>> scm_confounded = gen.create_random(possible_functions=["linear"], n_endo=5, n_exo=7,
                                   exo_distribution=random.random, exo_distribution_kwargs={},
                                   allow_exo_confounders=True)[0]
```
To generate an SCM based on a given causal structure:
```Python
>>> generator.create_scm_from_graph(graph=GRAPH, possible_functions=['linear'],
                                exo_distribution=random.randint, exo_distribution_kwargs={'a': 2, 'b': 5})
```

`./envs/generation/functions.py` defines the functions that can be used when defining your SCM. You can either use the
pre-defined functions, or define the function that is tailored to your need.

The interface to define your own data-generating function is as follows: define an (outer) python function that takes as
input a list of strings (the parent's variable names) and outputs the (inner) data-generation function. Giving the inner
function **kwargs as parameters, allows you to access the parent variable's values. The remaining behaviour of the outer
and inner function can be arbitrarily defined. An example is given below:

```Python
def f_linear(parents: List[str]):
    weights = {p: random.uniform(0.5, 2.0) for p in parents}
    default_value = 0.0

    def f(**kwargs):
        if len(kwargs) == 0:
            mu = default_value
        else:
            mu = 0.0

        for p in parents:
            mu += weights[p] * kwargs[p]
        return mu

    return f
```
This function can then be passed to the generator as described above.

"""
from .scm import StructuralCausalModel
from .generators import SCMGenerator, CausalGraphGenerator, CausalGraphSetGenerator
from .scm_environment import SCMEnvironment
