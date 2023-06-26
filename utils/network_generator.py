import numpy as np
import pandas as pd
from collections import namedtuple

Node = namedtuple('Node', 'node_id x y')
Link = namedtuple('Link', 'link_id, from_, to_, cost, capacity')
OD = namedtuple('OD', 'origin destination flow')

class NetworkGenerator:

    def __init__(self, size, capacity, demand, network_type='grid', bidirection=False, randomness=False, unit_len=1):
        self.size = size
        self.cap = capacity
        self.demand = demand
        self.type = network_type
        self.bidirection = bidirection
        self.rand = randomness
        self.unit_len = unit_len

    def draw_cap(self):
        if self.rand:
            return np.clip(np.random.normal(loc=self.cap,scale=(0.5*self.cap)), 0.5*self.cap, 1.5*self.cap)
        else:
            return self.cap

    def draw_cost(self, loc):
        if self.rand:
            return np.clip(np.random.normal(loc=loc,scale=(0.5*loc)), 0.25*loc, 2*loc)
        else:
            return loc

    def generate_network(self):
        # self.size is the number of links on edge
        N = self.size + 1 # number of nodes on edge

        self.node_data = []
        self.link_data = []
        self.od_data = []

        if self.type == 'grid':
            num_nodes = int(N ** 2)
            self.node_deg = {i:0 for i in range(num_nodes)}
            for i in range(num_nodes):
                x = self.unit_len * (i % N)
                y = self.unit_len * (i // N)
                self.node_data.append(Node(i, x, y))

                # right-bottom corner
                if i == (num_nodes-1):
                    pass
                # right edge
                elif (i+1) % N == 0:
                    self._add_link(i, i+N)
                # bottom edge
                elif i // N == (N-1):
                    self._add_link(i, i+1)
                # otherwise
                else:
                    self._add_link(i, i+N)
                    self._add_link(i, i+1)

                if self.bidirection:
                    # left-up corner
                    if i == 0:
                        pass
                    # left edge
                    elif (i+1) % N == 1:
                        self._add_link(i,i-N)
                    # bottom edge
                    elif i // N == 0:
                        self._add_link(i, i-1)
                    # otherwise
                    else:
                        self._add_link(i,i-N)
                        self._add_link(i,i-1)

            if not self.bidirection:
                self.od_data.append(OD(0, num_nodes-1, self.demand))
            else:
                strategy = 'gravity'
                od_unit = 2
                n_units = self.size // od_unit
                if strategy == 'triangle':
                    o_set = {od_unit*l for l in range(n_units+1)} | {N*od_unit*k for k in range(1,n_units)}
                    d_set = {self.size*N + od_unit*l for l in range(n_units+1)} | {N*od_unit*k + self.size for k in range(1,n_units)}
                    for o in o_set:
                        for d in d_set:
                            if o == 0 or o == self.size: #left and right up corners
                                self.od_data.append(OD(o, d, self.demand*(2/3)/len(d_set)))
                            else:
                                self.od_data.append(OD(o, d, self.demand/len(d_set)))
                elif strategy == 'ladder':
                    o_set = [[od_unit*l for l in range(1,n_units)], [N*od_unit*k for k in range(1,n_units)]]
                    d_set = [[self.size*N + od_unit*l for l in range(1,n_units)], [N*od_unit*k + self.size for k in range(1,n_units)]]
                    for o_nodes, d_nodes in zip(o_set, d_set):
                        for o in o_nodes:
                            for d in d_nodes:
                                self.od_data.append(OD(o, d, self.demand/len(d_nodes)))
                elif strategy == 'gravity':
                    location = 'cross'
                    gamma = 0.1
                    if location == 'grid':
                        od_set = {od_unit*k*N + od_unit*l: od_unit*np.array([k, l]) for k in range(n_units+1) for l in range(n_units+1)}
                    elif location == 'cross':
                        n_units = n_units // 2
                        od_set = {2*od_unit*k*N + 2*od_unit*l: 2*od_unit*np.array([k, l]) for k in range(n_units+1) for l in range(n_units+1)}
                        od_set.update({od_unit*(2*k+1)*N + od_unit*(2*l+1): od_unit*np.array([2*k+1, 2*l+1]) for k in range(n_units) for l in range(n_units)})
                        self.d_nodes = od_set
                    c = {o:{d: np.sum(np.abs(od_set[d] - od_set[o])) for d in od_set.keys()} for o in od_set.keys()}
                    for o, c_o in c.items():
                        c_o.pop(o)
                        deno = np.sum(np.exp(-gamma * np.array(list(c_o.values()))))
                        gen_flow = self.demand * self.node_deg[o]
                        for d in od_set.keys():
                            if d != o:
                                q_od = gen_flow * np.exp(-gamma * c_o[d])/deno
                                self.od_data.append(OD(o, d, q_od))

        node_df = pd.DataFrame(self.node_data, columns=self.node_data[0]._fields)
        link_df = pd.DataFrame(self.link_data, columns=self.link_data[0]._fields)
        od_df = pd.DataFrame(self.od_data, columns=self.od_data[0]._fields)

        return node_df, link_df, od_df

    def _add_link(self, from_, to_):
        self.link_data.append(
            Link(len(self.link_data)+1, from_, to_, self.draw_cost(self.unit_len), self.draw_cap())
        )
        self.node_deg[from_] += 1

    def _get_size(self):
        return self.size, len(self.node_data), len(self.link_data), len(self.od_data), len(self.d_nodes)

    def _print_size(self):
        print(
            'network size: n = {}, num_node = {}, num_link = {}, num_od = {}, num_d = {}'.format(
                self.size, len(self.node_data), len(self.link_data), len(self.od_data), len(self.d_nodes))
        )

if __name__ == '__main__':
    generator = NetworkGenerator(24, 100, 300, 'grid', bidirection=True)
    node_df, link_df, od_df = generator.generate_network()
    #print(node_df)
    #print(link_df)
    #print(od_df)
