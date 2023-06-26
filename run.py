from core.graph import Graph
import core.loading_models as loading
from core.define_sue import *
from utils.comparator import Comparator
import pandas as pd
import numpy as np
import argparse
import os
import json
import time

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
data_arg.add_argument('--network', type=str, default='SiouxFalls', help='network')
data_arg.add_argument('--reference', type=str2bool, default=False, help='reference data')
data_arg.add_argument('--x_axis', type=str, default='Iteration', help='network')


# Model parameters
model_arg = add_argument_group('Model')
model_arg.add_argument('--model_name', type=str, default='NGEVMCA', help='network loading model')
data_arg.add_argument('--optimizers', nargs='+', type=str, default=['PL', 'AGDBT'], help='compared models')
model_arg.add_argument('--syntax', type=str, default='0.5', help='NGEV parameter settings')
model_arg.add_argument('--beta', nargs='+', type=float, default=[0.15,4.], help='params for link-cost performance functions')
model_arg.add_argument('--congestion', type=float, default=1., help='congestion level which multiplies od flows')
model_arg.add_argument('--m_min', type=int, default=100, help='minimum number of iterations')
model_arg.add_argument('--m_max', type=int, default=500, help='maximum number of iterations')
model_arg.add_argument('--threshold', nargs='+', type=float, default=[1e-3, 1e-3, 1e-3, 1e-3], help='threshold for convergence: 0=var,1=obj,2=grad')
model_arg.add_argument('--print_step', type=int, default=50, help='every step to log results')
model_arg.add_argument('--parallel', type=str2bool, default=False, help='parallel computing')
model_arg.add_argument('--tolerance', nargs='+', type=float, default=[1e-20, 1e-10], help='tolerance for MCA')
model_arg.add_argument('--n_draws', type=int, default=1000, help='variance parameter')
model_arg.add_argument('--seed', type=int, default=123, help='variance parameter')

# AGD parameters
agd_arg = add_argument_group('AGD')
agd_arg.add_argument('--step_size', type=float, default=1e-5, help='step size')
agd_arg.add_argument('--k_min', type=int, default=50, help='minimum number for adaptive restart')
agd_arg.add_argument('--with_BT', type=str2bool, default=True, help='with backtracking')
agd_arg.add_argument('--eta', type=float, default=0.95, help='param for backtracking')
agd_arg.add_argument('--min_s', type=float, default=1e-8, help='lower bound of step size')

# FW parameters
fw_arg = add_argument_group('FW')
fw_arg.add_argument('--lr', type=float, default=1e-4, help='learning rate')

# Adam parameters
adam_arg = add_argument_group('Adam')
adam_arg.add_argument('--adam_params', nargs='+', type=float, default=[0.01,0.9,0.999,1e-8], help='default params for adam')

# Line Search parameters
ls_arg = add_argument_group('Line Search')
ls_arg.add_argument('--line_search', type=str, default='PL', help='solution method for primal')
ls_arg.add_argument('--ls_tolerance', type=float, default=1e-3, help='tolerance for golden section')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed

if __name__ == '__main__':
    config, _ = get_config()
    network_ = 'dataset/' + config.network + '/'
    save_dir = network_ + time.strftime("%Y%m%dT%H%M%S")
    os.makedirs(save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(vars(config), f, indent=True)
    save_dir += '/'
    print(save_dir)

    # input files
    node_data = pd.read_csv(network_+'node.csv')
    link_data = pd.read_csv(network_+'link.csv')
    od_data = pd.read_csv(network_+'od.csv')
    ref_link_vars = pd.read_table(network_+'link_vars_ref_'+str(config.congestion)+'q_'+str(config.syntax)+'_AGPBT.csv') if config.reference else None
    ref_metric = pd.read_csv(network_+'metric_ref_'+str(config.congestion)+'q_'+str(config.syntax)+'.csv') if config.reference else None

    if config.model_name == 'Probit':
        ref_link_vars = pd.read_table(network_+'link_vars_ref_'+str(config.congestion)+'q_probit10000.csv') if config.reference else None
        ref_metric = None

    # build graph
    graph = Graph()
    graph.build_graph(node_data, link_data)
    graph.set_od_flows(od_data, congestion=config.congestion)
    graph.compute_shortest_paths()

    # define comparator
    comp = Comparator(ref_val=ref_metric)

    # define theta and loading model
    if config.syntax == 'Inputs':
        theta_data = pd.read_csv(network_+'theta.csv')
        theta = list(theta_data['theta'].values)
        theta = np.array(theta, dtype=np.float)
    else:
        theta = None

    if config.model_name != 'Probit':
        model = {
            'NGEVMCA': loading.NGEVMCA(graph, syntax=config.syntax, theta=theta, alpha=None, parallel=config.parallel, tolerance=config.tolerance),
            'NGEVDial': loading.NGEVDial(graph, syntax=config.syntax, theta=theta, alpha=None, parallel=config.parallel),
            'LogitMCA': loading.LogitMCA(graph, syntax='Logit', theta=float(config.syntax), parallel=config.parallel),
            'LogitDial': loading.LogitDial(graph, syntax='Logit', theta=float(config.syntax), parallel=config.parallel),
        }.get(config.model_name, None)
    else:
        model = loading.Probit(graph, theta=0.5, n_draws=config.n_draws, seed_=config.seed, parallel=config.parallel)

    # SUE assignments
    for optimizer in config.optimizers:
        sue_class = {
            'PL': get_pl,
            'MSA': get_msa,
            'GD': get_gd,
            'AGDBT': get_agd_BT,
            'AGD': get_agd,
            'Probit': get_probit
        }.get(optimizer, None)
        sue, dir_ = sue_class(config, graph, ref_link_vars, None)
        sue.solve(model)
        graph.write_link_vars(
            [sue.X[sue.m], sue.c[sue.m], sue.inv_c[sue.m], sue.grad_c[sue.m]],
            ['flow', 'cost', 'invC', 'gradC'],
            save_dir+dir_+config.model_name+'_'+config.syntax+'_'+str(config.congestion)+'q_linkvars.csv'
        )
        sue.write_results(save_dir+dir_+config.model_name+'_'+config.syntax+'_'+str(config.congestion)+'q_metric.csv')
        sue.write_objectives(model, save_dir+dir_+config.model_name+'_'+config.syntax+'_'+str(config.congestion)+'q_obj_vals.csv')
        comp.add_model(sue, optimizer)

    # comparison
    comp.visualize_results(file_dir_=save_dir+'comparison', x_axis=config.x_axis)
