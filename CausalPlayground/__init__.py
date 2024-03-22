"""
# Overview
The CausalPlayground library serves as a tool for causality research,
focusing on the interactive exploration of structural
causal models (SCMs). It provides extensive functionality for creating, manipulating and sampling SCMs, seamlessly
integrating them with the Gymnasium framework. Users have complete control over SCMs, enabling precise manipulation and
interaction with causal mechanisms. Additionally, CausalPlayground offers a range of useful helper functions for generating
diverse instances of SCMs and DAGs, facilitating quantitative experimentation and evaluation. Notably, the library is
optimized for (but not limited to) easy integration with reinforcement learning methods, enhancing its utility in active inference and
learning settings. This documentation presents the complete API documentation and a quickstart guide. The GitHub
repository can be found [here](https://github.com/sa-and/CausalPlayground).

# Installation guide
In your python environment `pip install causal-playground`.

# Structural Causal Models (SCM)
SCMs are a powerful model to express a data-generating process that is governed by clear causal relations represented
as functions. More formally, an SCM models the causal relationship between endogenous variables, describing the main
random variables of intrerest, and exogenous variables that, roughly speaking, model the context of the process. The
causal effects are described as functional relations between variables, and the exogenous variables are modeled through
distributions. Taken together, an SCM defines the joint probability distribution over its variables.

## Defining an SCM
Creating a Structural Causal Model (SCM) with CausalPlayground is straightforward. First, instantiate an object of the
`StructuralCausalModel` class. Then, add endogenous and exogenous variables to the SCM using the `add_endogenous_var()`
 and `add_exogenous_var()` methods, respectively. For each endogenous variable, specify its name, a function that
 determines its value based on its dependencies, and a dictionary mapping the function's arguments to the names of the
 variables it depends on. Similarly, for each exogenous variable, provide its name, a function/distribution for
 generating its values, and a dictionary of keyword arguments to pass to the function. The functions can be defined as
 lambda functions. Alternatively, any callable function is admissible. Note that the variable names will
always be converted to upper-case.

An example for creating the SCM $\mathcal{M} = (\mathcal{X}, \mathcal{U}, \mathcal{F}, \mathcal{P})$ with
$\mathcal{X}= \\\{ A, Effect \\\}$, $\mathcal{U}=\\\{U\\\}$, $\mathcal{P}=\\\{Uniform(3, 8)\\\}$, and
$\mathcal{F} = \\\{A\leftarrow 5+U, Effect \leftarrow A*2\\\}$

```Python
>>> scm = StructuralCausalModel()
>>> scm.add_endogenous_var('A', lambda noise: noise+5, {'noise': 'U'})
>>> scm.add_exogenous_var('U', random.randint, {'a': 3, 'b': 8})
>>> scm.add_endogenous_var('Effect', lambda x: x*2, {'x': 'A'})
```

## Sampling the SCM
To sample from this SCM, use the `get_next_sample()` method of the `StructuralCausalModel` object. This method returns a
 tuple $(x, u)$, where $x$ is a dictionary containing the sampled values of the endogenous variables, and $u$ is a
 dictionary containing the sampled values of the exogenous variables. For example:
```Python
>>> x, u = scm.get_next_sample()
>>> x
{'A': 10, 'EFFECT': 20}
>>> u
{'U': 5}
```

## Intervening in an SCM
Interventions on an SCM can be performed using the `do_interventions()` method of the `StructuralCausalModel` object.
This method takes a list of tuples, where each tuple represents an intervention on a specific variable. The first
element of the tuple is the name of the variable to intervene on, and the second element is a tuple containing the
intervention function and a dictionary mapping the function's arguments to the names of the variables it depends on. To
perform an intervention $do(X_0 \leftarrow 5, X_1 \leftarrow X_0+1)$, the following can be implemented:
```Python
    >>> scm.do_interventions([("X0", (lambda: 5, {})),
                              ("X1", (lambda x0: x0+1, {'x0':'X0'})])
```
The `do_interventions()` method is called with a list of two interventions. The first intervention sets the value of the
variable `X0` to a constant value of 5, using a lambda function with no arguments. The second intervention sets the
value of the variable `X1` to the value of `X0+1`, using a lambda function that takes `x0` as an argument and a
dictionary that maps  `x0` to the variable name `X0`. After  applying these interventions, the SCM will use the new
causal mechanisms for the intervened variables when sampling. To undo all interventions
that are currently applied to the SCM, you can call `scm.undo_interventions()`, which will return the SCM to its
original state with $do(\emptyset)$.

# SCM Environments
## Creating Interactive Environments
To create an interactive environment for working with an SCM using the Gymnasium framework, use the `SCMEnvironment`
class from the `CausalPlayground` module. Instantiate an `SCMEnvironment` object by passing the `StructuralCausalModel`
object and a list of possible interventions. These interventions correspond to the actions in this environment. The list
 of possible interventions follows the same format as the `do_interventions()` method, where each intervention is
 represented by a tuple containing the variable name, the intervention function, and a dictionary mapping the function's
  arguments to variable names. In the example code snippet:
```Python
>>> env = SCMEnvironment(scm, possible_interventions=[  ("X0", (lambda: 5, {})),
                                                        ("X1", (lambda x0: x0+1, {'x0':'X0'}))])
```
The first intervention sets the value of `X0` to a constant value of 5, while the second intervention sets the value of
`X1` to the value of `X0+1`. The resulting `env` object can be used to interact with the SCM using the standard
Gymnasium environment interface, allowing for interactive exploration and experimentation with the causal model.

## Interacting with SCMs
Calling the `step(action)` function applies the interventions defined in `action`, samples the
intervened SCM, determines the new observation, termination flag, truncated flag and reward, and, finally undoes the
interventions.

The action is a list of indices corresponding to the indices of `possible_interventions` defined upon the initialization
 of the environment. This allows for multiple interventions simultaneously. For example, simultaneously performing both
 possible interventions in one step can be done with `env.step([1, 2])`. Whereas performing a step without intervention
 can be done by either invoking the first (empty) intervention `env.step([0])`, or no intervention `env.step([])`.

## Generating data from the SCM
In its basic implementation, `SCMEnvironment` always returns the current values of the variables as observation and 0
reward regardless of the action. Furthermore, the terminated and truncated flag are always `False`, leading to an
episode never terminating. The basic implementation can be used to generate samples (e.g. 1000) from an SCM either with
no interventions (observational data):
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

The collected data can be found in `env.samples_so_far` and the corresponding intervention targets in `env.targets_so_far`. These
buffers can be flushed on demand with `clear_samples()` and `clear_intervention_targets()`, respectively.

## Extending SCMEnvironment
While the basic implementation is useful for collecting data generated by an SCM, it holds potential for more
sophisticated applications to reinforcement learning by inheriting from `SCMEnvironment`. To adapt the environment to a
specific task, override the neccessary functions determining observation, reward, truncated, and terminated. (Note that
the interventions in a step are undone *after* these quantities are determined). Below we provide an example for an
environment that rewards interventions:
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
>>> env = NoIntervEnv(scm, possible_interventions=[ ("A", (lambda: 5, {})),
                                                    ("EFFECT", (lambda x0: x0+1, {'x0': 'A'}))])
```

# Generators
In some scenarios it might be useful to automatically generate graphs and SCMs. To this end, we implemented some helper
classes.

## Generating Graphs
The `CausalGraphGenerator` class allows you to quickly generate random DAGs with a specified number of endogenous and
exogenous variables. To create a random graph with 7 endogenous and 5 exogenous variables, you can use the following
code:
```Python
>>> gen = CausalGraphGenerator(7, 5)
>>> graph = gen.generate_random_graph()
```
By default, the generated graphs do not include exogenous confounders. However, you can allow for the presence of
exogenous confounders by setting the  `allow_exo_confounders` parameter to `True` when instantiating the
`CausalGraphGenerator`:
```Python
>>> gen = CausalGraphGenerator(7, 5, allow_exo_confounders=True)
```

If you need to generate a large set of distinct DAGs, you can use the `CausalGraphSetGenerator` class. This class
ensures that each generated graph is unique within the set. To generate 1000 distinct DAGs with 7 endogenous nodes and 5
 exogenous nodes, you can use the following code:
```Python
>>> gen = CausalGraphSetGenerator(7, 5)
>>> gen.generate(1000)
```
the generated DAGs can be accessed through the `gen.graphs` attribute.

## Generating SCMs
The  `CausalPlayground.SCMGenerator` class provides a convenient way to generate random SCMs with specified properties,
such as the number of endogenous and exogenous variables, causal relationships, and exogenous variable distributions.
This feature allows you to quickly create datasets of SCMs for testing and experimentation. This is exemplified by the
following code:
```Python
>>> from tests.functions import f_linear

>>> gen = SCMGenerator(all_functions={'linear': f_linear})
>>> scm_unconfounded = gen.create_random(possible_functions=["linear"], n_endo=5, n_exo=4,
                                         exo_distribution=random.random,
                                         exo_distribution_kwargs={},
                                         allow_exo_confounders=False)[0]
```
In this example, the  `SCMGenerator` is instantiated with a dictionary of functions, where the key `'linear'` is
associated with the `f_linear` function that returns a linear combination of inputs with random weights. The
`create_random` method is then used to generate an SCM with 5 endogenous variables and 4 exogenous variables, using the
 `f_linear` function for the causal relationships and `random.random` for the exogenous variable distribution. The
`allow_exo_confounders` parameter determines whether the generated SCM allows for exogenous confounders.

You can also generate a random SCM based on a given causal structure by using the `create_scm_from_graph` method. The
code snipped below shows and example for generating and SCM based on a given GRAPH structure, the `'linear'` function
defined above for endogenous variables, and a uniform distribution between 2 and 5 for the exogenous variables:
```Python
>>> generator.create_scm_from_graph(graph=GRAPH, possible_functions=['linear'],
                                    exo_distribution=random.randint,
                                    exo_distribution_kwargs={'a': 2, 'b': 5})
```

## Custom Causal Relationships
To define your own data-generating function, you need to create an outer function that takes a list of strings (the
parent variable names) as input and returns an inner function that determines the value of the associated variable. The
inner function should take `**kwargs` as parameters to access the parent variable values. An example of a custom
data-generating function, `f_linear`, is provided in the code snippet below.

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
By using the `SCMGenerator` class and custom data-generating functions, you can easily create datasets of SCMs with
desired properties, enabling you to test and evaluate various causal inference algorithms and techniques.

"""
from .scm import StructuralCausalModel
from .generators import SCMGenerator, CausalGraphGenerator, CausalGraphSetGenerator
from .scm_environment import SCMEnvironment
