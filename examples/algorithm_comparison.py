"""Script that uses the CausalPlayground library to compare causal discovery algorithms."""
import warnings
from CausalPlayground import *
from tests.functions import f_linear
from castle.metrics import MetricsDAG
from castle.algorithms import PC, GES, Notears
import random
import networkx as nx
import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Hyperparameters
n_scms = 20  # how many SCMs should be evaluated
algos = {'PC': PC, 'GES': GES, 'NOTEARS': Notears}  # which algorithms should be compared

# generate distinct confounded and unconfounded causal graphs
generator = CausalGraphSetGenerator(n_endo=4, n_exo=4, allow_exo_confounders=False)
generator.generate(n_scms)
graphs_unconfounded = generator.graphs
generator = CausalGraphSetGenerator(n_endo=4, n_exo=4, allow_exo_confounders=True)
generator.generate(n_scms)
graphs_confounded = generator.graphs

# generate the SCMs for evaluation
generator = SCMGenerator(all_functions={"linear": f_linear}, seed=42)
scms_unconfounded = [generator.create_scm_from_graph(  graph,
                                                       possible_functions=['linear'],
                                                       exo_distribution=random.gauss,
                                                       exo_distribution_kwargs={'mu': 0, 'sigma': 1})
                               for graph in graphs_unconfounded]
scms_confounded = [generator.create_scm_from_graph(graph,
                                                     possible_functions=['linear'],
                                                     exo_distribution=random.gauss,
                                                     exo_distribution_kwargs={'mu': 0, 'sigma': 1})
                               for graph in graphs_confounded]

# initialize the results
metrics = ['fdr', 'tpr', 'fpr', 'shd', 'nnz', 'precision', 'recall', 'F1', 'gscore']
results = {algo: {'uconf': pd.DataFrame(columns=metrics), 'conf': pd.DataFrame(columns=metrics)} for algo in algos.keys()}
# Evaluate algorithms
for scm_uconf, scm_conf in zip(scms_unconfounded, scms_confounded):
    # Define the Gymnasium environment based on the SCM with no interventions
    env_uconf = SCMEnvironment(scm_uconf, possible_interventions=[])
    env_conf = SCMEnvironment(scm_conf, possible_interventions=[])
    # Collect 1000 samples from the SCM
    [env_uconf.step([0]) for _ in range(500)]
    [env_conf.step([0]) for _ in range(500)]
    samples_uconf = env_uconf.samples_so_far.to_numpy()
    samples_conf = env_conf.samples_so_far.to_numpy()
    for a in algos.keys():
        algo = algos[a]()
        # learn causal graph in unconfounded case
        try:
            algo.learn(samples_uconf)
        except LinAlgError or ValueError:  # discard SCMs that create errors
            continue
        # compute and store metrics
        mt = MetricsDAG(algo.causal_matrix, nx.to_numpy_array(scm_uconf.create_graph()))
        results[a]['uconf'].loc[len(results[a]['uconf'])] = mt.metrics

        # learn causal graph in unconfounded case
        try:
            algo.learn(samples_conf)
        except LinAlgError or ValueError:  # discard SCMs that create errors
            continue
        # compute and store metrics
        mt = MetricsDAG(algo.causal_matrix, nx.to_numpy_array(scm_conf.create_graph()))
        results[a]['conf'].loc[len(results[a]['conf'])] = mt.metrics

# plot results
# Extract the required metrics from the results
algos_list = list(algos.keys())
metrics_to_plot = ['shd', 'F1', 'fdr', 'tpr']

# Prepare the data for plotting
plot_data = {metric: {'uconf': [], 'conf': []} for metric in metrics_to_plot}
for algo in algos_list:
    for metric in metrics_to_plot:
        plot_data[metric]['uconf'].append(results[algo]['uconf'][metric].mean())
        plot_data[metric]['conf'].append(results[algo]['conf'][metric].mean())

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Set the width of each bar and the spacing between groups
bar_width = 0.08
group_spacing = 0.1

# Calculate the positions of the bars on the x-axis
x = np.arange(len(algos_list))

# Create the bars for each metric and configuration
for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i * (bar_width + group_spacing), plot_data[metric]['uconf'], width=bar_width, label=f'{metric} (uconf)')
    ax.bar(x + i * (bar_width + group_spacing) + bar_width, plot_data[metric]['conf'], width=bar_width, label=f'{metric} (conf)')

# Set the x-axis labels and tick positions
ax.set_xticks(x + (len(metrics_to_plot) - 1) * (bar_width + group_spacing) / 2)
ax.set_xticklabels(algos_list)

# Add labels and title
ax.set_ylabel('Metric Value')
ax.set_xlabel('Algorithms')
ax.set_title('Comparison of Algorithms and Configurations')

# Add a legend
ax.legend()

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()