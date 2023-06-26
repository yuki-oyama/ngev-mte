from core.graph import Graph
from core.loading_models import NGEVMCA, NGEVDial
from core.define_sue import *
from utils.comparator import Comparator
from utils.network_generator import NetworkGenerator
import pandas as pd
import numpy as np
import argparse
import os
import json
import time
import gc
from icecream import ic
from copy import copy

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
  return v.lower() in ('true', '1')

# Data parameters
data_arg = add_argument_group('Data')
data_arg.add_argument('--network', type=str, default='cross', help='network')
data_arg.add_argument('--reference', type=str2bool, default=False, help='reference data')
data_arg.add_argument('--models', nargs='+', type=str, default=['PL', 'AGDBT'], help='compared models')
data_arg.add_argument('--x_axis', type=str, default='CPU time', help='network')
data_arg.add_argument('--network_size', nargs='+', type=int, default=[4], help='network size')
data_arg.add_argument('--network_type', type=str, default='grid', help='network type')
data_arg.add_argument('--bidirection', type=str2bool, default=True, help='bidirection')
data_arg.add_argument('--capacity', type=float, default=10000, help='network capacity')
data_arg.add_argument('--demand', nargs='+', type=float, default=[10000], help='od demand')

# Model parameters
model_arg = add_argument_group('Model')
model_arg.add_argument('--model_name', type=str, default='MCA', help='NGEV parameter settings')
model_arg.add_argument('--syntax', type=str, default='0.5', help='NGEV parameter settings')
model_arg.add_argument('--beta', nargs='+', type=float, default=[0.15,4.], help='params for link-cost performance functions')
model_arg.add_argument('--congestion', type=float, default=1., help='congestion level which multiplies od flows')
model_arg.add_argument('--optimizer', type=str, default='AGD', help='optimizer')
model_arg.add_argument('--m_min', type=int, default=50, help='minimum number of iterations')
model_arg.add_argument('--m_max', type=int, default=500, help='maximum number of iterations')
model_arg.add_argument('--threshold', nargs='+', type=float, default=[1e-3, 1e-3, 1e-3, 1e-5], help='threshold for convergence: 0=x,1=c,2=grad,3=obj')
model_arg.add_argument('--print_step', type=int, default=10, help='every step to log results')
model_arg.add_argument('--tolerance', nargs='+', type=float, default=[1e-20, 1e-10], help='tolerance for MCA')
model_arg.add_argument('--parallel', type=str2bool, default=False, help='parallel computing')

# AGD parameters
agd_arg = add_argument_group('AGD')
agd_arg.add_argument('--step_size', type=float, default=1e-4, help='step size')
agd_arg.add_argument('--k_min', type=int, default=50, help='minimum number for adaptive restart')
agd_arg.add_argument('--with_BT', type=str2bool, default=True, help='with backtracking')
agd_arg.add_argument('--eta', type=float, default=0.25, help='param for backtracking')
agd_arg.add_argument('--min_s', type=float, default=1e-8, help='lower bound of step size')

# # Adam parameters
# adam_arg = add_argument_group('Adam')
# adam_arg.add_argument('--adam_params', nargs='+', type=float, default=[0.01,0.9,0.999,1e-8], help='default params for adam')

# Line Search parameters
ls_arg = add_argument_group('Line Search')
ls_arg.add_argument('--line_search', type=str, default='PL', help='solution method for primal')
ls_arg.add_argument('--ls_tolerance', type=float, default=1e-3, help='tolerance for golden section')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed

if __name__ == '__main__':
    config, _ = get_config()
    network_ = 'results/grid_network/'
    # reference_dir = network_ + '_reference/cross/'
    save_dir = network_ + time.strftime("%Y%m%dT%H%M%S")
    os.makedirs(save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(vars(config), f, indent=True)
    save_dir += '/'
    print(save_dir)

    for demand in config.demand:
        # if config.reference:
        #     ref_Z = pd.read_csv(reference_dir+'reference_q'+str(int(demand))+'.csv', index_col=0)
        for network_size in config.network_size:
            # generate network data
            n_generator = NetworkGenerator(
                network_size, config.capacity, demand,
                network_type=config.network_type, bidirection=config.bidirection
            )
            node_data, link_data, od_data = n_generator.generate_network()
            n_generator._print_size()

            # build graph
            graph = Graph()
            graph.build_graph(node_data, link_data)
            graph.set_od_flows(od_data, congestion=config.congestion)
            graph.compute_shortest_paths()

            # define theta and loading model
            if config.syntax == 'Inputs':
                theta_data = pd.read_csv(network_+'theta.csv')
                theta = list(theta_data['theta'].values)
                theta = np.array(theta, dtype=np.float)
            else:
                theta = None

            model_class = {
                'MCA': NGEVMCA,
                'Dial': NGEVDial
            }.get(config.model_name, None)
            model = model_class(graph, syntax=config.syntax, theta=theta, alpha=None, parallel=config.parallel, tolerance=copy(config.tolerance))

            # SUE assignments
            for model_name in config.models:

                if config.reference:
                    ref_key = 'AGDBT' if model_name == 'AGDBT' else 'PL'
                    ref_data = pd.read_csv(reference_dir+ref_key+'_q'+str(int(demand))+'_'+str(network_size)+'_linkvars.csv', sep='\t')
                    optZ = float(ref_Z.loc[network_size,ref_key])
                    ic(optZ)
                else:
                    ref_data = None
                    optZ = None

                sue_class = {
                    'PL': get_pl,
                    'MSA': get_msa,
                    'GD': get_gd,
                    'AGDBT': get_agd_BT,
                    'AGD': get_agd,
                }.get(model_name, None)
                sue, dir_ = sue_class(config, graph, ref_data=ref_data, ref_opt=optZ)
                model.torelance = copy(config.tolerance)
                sue.solve(model)
                graph.write_link_vars(
                    [sue.X[sue.m], sue.c[sue.m], sue.inv_c[sue.m], sue.grad_c[sue.m]],
                    ['flow', 'cost', 'invC', 'gradC'],
                    save_dir+dir_+'_'+config.model_name+'_'+str(network_size)+'_'+str(int(demand))+'q_linkvars.csv'
                )
                sue.write_results(save_dir+dir_+'_'+config.model_name+'_'+str(network_size)+'_'+str(int(demand))+'q_metric.csv')
                sue.write_objectives(model, save_dir+dir_+'_'+config.model_name+'_'+str(network_size)+'_'+str(int(demand))+'q_obj_vals.csv')

                # release memory
                del sue
                gc.collect()
