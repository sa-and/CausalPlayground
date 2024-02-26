"""Script ot generate and save a dataset of DAGs and resulting SCMs"""

import argparse
import os
from CausalPlayground.generators import CausalGraphSetGenerator
from CausalPlayground import SCMGenerator
import dill
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-graphs', type=int, default=5,
                        help='Amount of graphs to generate.')
    parser.add_argument('--scms-per-graph', type=int, default=5,
                        help='Amount of SCMs to generate.')
    parser.add_argument('--save-dir', type=str, help='Filepath of where to save the data.')
    parser.add_argument('--functions', nargs='+',
                        default=["linear"],
                        help='List of functions that should be included in the SCM. Must be subset of '
                             '["linear", "cubic", "even", "quadratic", "greaterThanSix", "mix", NN]')
    parser.add_argument('--n-endo', type=int, default=3,
                        help='Amount of endogenous variables.')
    parser.add_argument('--n-exo', type=int, default=0,
                        help='Amount of exogenous variables.')
    parser.add_argument('--confounders', action='store_true', default=False,
                        help='Whether to allow for exogenous confounders or not.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')

    args = parser.parse_args()
    os.mkdir(args.save_dir)

    # Generate the graphs
    graph_set_gen = CausalGraphSetGenerator(n_endo=args.n_endo, n_exo=args.n_exo, allow_exo_confounders=args.confounders, seed=args.seed)
    print('Generating Graphs...')
    graph_set_gen.generate(args.n_graphs)
    graph_set_gen.save(args.save_dir+'graphs.pkl', 'wb')
    print('Graphs saved to ', args.save_dir+'graphs.pkl')

    # generate n SCMs per graph
    scm_gen = SCMGenerator(seed=args.seed)
    print('Generating SCMs...')
    scms = []
    [[scms.append(scm_gen.create_scm_from_graph(graph, args.functions)) for i in range(args.scms_per_graph)] for graph
     in tqdm(graph_set_gen.graphs)]

    with open(args.save_dir+'scms.pkl', 'wb') as f:
        dill.dump(scms, f)

    print('Scms saved to ', args.save_dir+'scms.pkl')