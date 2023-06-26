import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from copy import copy
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

class Graph(object):

    def __init__(self):
        pass

    def build_graph(self, node_data, link_data):
        """
        Arguments:
            node_data: pandas dataframe ('node_id', 'x', 'y')
            link_data: pandas dataframe ('link_id', 'from_', 'to_', 'cost', 'capacity')
        """

        self.nodes = [] # list of node indexes
        self.pos = []
        self.forward_stars = {}
        self.backward_stars = {}
        self.links = []
        self.senders = []
        self.receivers = []
        self.c0 = []
        self.mu = []
        self.num_ods = 0
        self.node_idxs = {}

        for i in range(len(node_data)):
            node = node_data.iloc[i]
            node_id, x, y = node['node_id':'y']
            self.nodes.append(i)
            self.forward_stars[i] = []
            self.backward_stars[i] = []
            self.node_idxs[int(node_id)] = i
            self.pos.append([float(x), float(y)])

        for a in range(len(link_data)):
            link = link_data.iloc[a]
            link_id, from_node, to_node, cost, mu = link['link_id':'capacity']
            i, j = self.node_idxs[int(from_node)], self.node_idxs[int(to_node)]
            self.links.append(a)
            self.senders.append(i)
            self.receivers.append(j)
            self.forward_stars[i].append(j)
            self.backward_stars[j].append(i)
            self.c0.append(float(cost))
            self.mu.append(float(mu))

        self.senders = np.array(self.senders, dtype=np.int)
        self.receivers = np.array(self.receivers, dtype=np.int)
        self.pos = np.array(self.pos, dtype=np.float)
        self.c0 = np.array(self.c0, dtype=np.float)
        self.mu = np.array(self.mu, dtype=np.float)
        self.link_idxs = {(s, r): a for a, s, r in zip(self.links, self.senders, self.receivers)}

    def set_od_flows(self, od_data, congestion=1.):
        """
        Arguments:
            od_flow: pd dataframe ('origin', 'destination', 'flow')
        """
        self.dests = []
        self.od_flows = {}

        for i in tqdm(range(len(od_data))):
            od = od_data.iloc[i]
            o = self.node_idxs[int(od['origin'])]
            d = self.node_idxs[int(od['destination'])]
            flow = float(od['flow'])
            if d not in self.dests:
                self.dests.append(d)
                self.od_flows[d] = {}
            self.od_flows[d][o] = flow * congestion
            self.num_ods += 1

        self.origins = {d: list(v.keys()) for d, v in self.od_flows.items()}

    def compute_shortest_paths(self):

        self.sp_costs = {}
        num_nodes = len(self.nodes)

        for d_idx, d in enumerate(self.dests):
            W = csr_matrix((self.c0, (self.receivers, self.senders)), shape=(num_nodes,num_nodes))
            sp_cost = shortest_path(W, method='D', directed=True, indices=d, return_predecessors=False)
            sp_cost = np.array(sp_cost)
            # rev_graph = nx.DiGraph()
            # rev_graph.add_nodes_from(self.nodes)
            # rev_links = [(r, s, c) for s, r, c in zip(self.senders, self.receivers, self.c0)]
            # rev_graph.add_weighted_edges_from(rev_links)
            # cost_to_d = nx.single_source_dijkstra_path_length(rev_graph, d)
            # sp_cost = np.array(
            #         [cost_to_d[node] for node in self.nodes], dtype=np.float
            #     )
            self.sp_costs[d] = sp_cost

    def visualize(self, weights, alpha=None, point_scale=50,
                    figsize=(10,8), annotate=False,
                    graph_title=None, file_path=None):
        """
        Arguments:
            inputs: vector of link variables (flow, cost etc.)
        """
        # figure
        plt.figure(figsize=figsize)

        # plot nodes
        plt.scatter(self.pos[:,0], self.pos[:,1], s=point_scale, c='gray')
        if annotate:
            for i in range(self.pos.shape[0]):
                plt.annotate(i, (self.pos[i,0], self.pos[i,1]))

        if alpha is None:
            alpha = np.ones(shape=weights.shape)

        # edges
        for a, i, j in zip(self.links, self.senders, self.receivers):
            color = 'b' if weights[a] > 0 else 'g'
            plt.plot(
                [self.pos[i,0], self.pos[j,0]],
                [self.pos[i,1], self.pos[j,1]],
                color=color,
                alpha=alpha[a],
                linewidth=weights[a]
            )

        plt.xlabel('x')
        plt.ylabel('y')
        # plt.xticks(np.arange(np.min(self.pos[:,0]), np.max(self.pos[:,0])+1))
        # plt.yticks(np.arange(np.min(self.pos[:,1]), np.max(self.pos[:,1])+1))
        if graph_title:
            plt.title(graph_title)

        if file_path:
            plt.savefig(file_path)
        else:
            plt.show()

    def write_link_vars(self, var_list, names, file_path):

        vars = np.array(var_list, dtype=np.float)

        with open(file_path, 'w') as f:
            header = 'from_ \t to_'
            for name in names:
                header += '\t' + name
            header += '\n'
            f.write(header)

            for a, i, j in zip(self.links, self.senders, self.receivers):
                txt = str(i) + '\t' + str(j)
                for v in range(len(var_list)):
                    txt += '\t' + str(vars[v,a])
                txt += '\n'
                f.write(txt)
