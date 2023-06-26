import numpy as np
import networkx as nx
from collections import OrderedDict
from copy import *
import multiprocessing as mp
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from icecream import ic
from utils.utils import Timer
from tqdm import tqdm

class Loader(object):

    def __init__(self,
                graph,
                syntax='1/BS',
                theta=None,
                alpha=None,
                parallel=False
                ):

        self.graph = graph
        self.theta = theta
        self.alpha = alpha
        self.syntax = syntax
        self.parallel = parallel
        self.eps = 1e-8
        self.inf = 1e+10
        self.c0 = graph.c0
        self.mu = graph.mu
        self.p = {}
        self.p_edge = {}

        # initialize parameter for the first loading
        self.define_params()

    def set_theta(self):
        self.theta = np.ones(
                shape=(len(self.graph.dests), len(self.graph.nodes)), dtype=np.float
            )
        if self.syntax == 'Logit':
            pass
        # backward_stars-based
        elif self.syntax == '1/BS':
            for i, node in enumerate(self.graph.nodes):
                BS = self.graph.backward_stars[node]
                if len(BS) == 0:
                    self.theta[:,i] = 1.
                else:
                    self.theta[:,i] = 1./len(BS)
        # forward_stars-based
        elif self.syntax == '1/FS':
            for i, node in enumerate(self.graph.nodes):
                FS = self.graph.forward_stars[node]
                if len(FS) == 0:
                    self.theta[:,i] = 1.
                else:
                    self.theta[:,i] = 1./len(FS)
        # shortest-path-cost-based
        elif self.syntax == 'SP':
            for d, dnode in enumerate(self.graph.dests):
                sp_cost = self.graph.sp_costs[dnode]
                self.theta[d] = sp_cost[0]/(sp_cost + self.eps)
                self.theta[d] = np.clip(self.theta[d], 0., 10.)
            #print(self.theta)
        else: # Papola and Marzano (2013)
            xi = float(self.syntax)
            for d, dnode in enumerate(self.graph.dests):
                sp_cost = self.graph.sp_costs[dnode]
                self.theta[d] = np.pi / np.sqrt(6.*xi*(sp_cost+self.eps))
                self.theta[d] = np.clip(self.theta[d], 0., 10.)
            #print(self.theta)

    def set_alpha(self):
        self.alpha = np.ones(len(self.graph.links), dtype=np.float)
        if self.syntax == 'Logit':
            pass
        else: # backward_stars-based
            for a, j in zip(self.graph.links, self.graph.receivers):
                self.alpha[a] = 1./np.sum(self.graph.receivers == j)

    def define_params(self):
        # define theta at first time
        if self.theta is None:
            self.set_theta()
        elif type(self.theta) != float and len(self.theta.shape) == 1:
            self.theta = np.tile(self.theta[np.newaxis,:], [len(self.graph.dests),1])
        # define alpha at first time
        if self.alpha is None:
            self.set_alpha()
        pass

    def load_flow(self, cost=None, ret_S=False):

        # cost used in this loading
        if cost is not None:
            c = cost.copy()
        else:
            c = self.c0.copy()

        # outputs
        num_nodes = len(self.graph.nodes)
        num_links = len(self.graph.links)
        num_dests = len(self.graph.dests)
        self.x = np.zeros(shape=(num_dests, num_links), dtype=np.float)
        self.total_cost = np.zeros(num_dests, dtype=np.float)
        self.entropy = np.zeros(num_dests, dtype=np.float)
        self.emc = np.zeros(num_dests, dtype=np.float)

        if self.parallel:
            # Assign flows (multiprocessing)
            n_cpu = mp.cpu_count()
            n_threads = n_cpu // 2 if num_dests > (n_cpu // 2) else num_dests
            batch_size = num_dests // n_threads
            p = mp.Pool(n_threads)
            d_arange = np.arange(num_dests)
            particles = [(num_dests + i) // n_threads for i in range(n_threads)]
            cum = np.cumsum(particles)
            paramsList = [
                [d_arange[cum[r-1]:cum[r]], c, ret_S] if r > 0 else [d_arange[:cum[r]], c, ret_S]
                for r in range(n_threads)
                ]
            results = p.map(self._load_parallel, paramsList)
            p.close()
            for r, params in enumerate(paramsList):
                self.x[params[0]] = results[r][0]
                self.total_cost[params[0]] = results[r][1]
                self.entropy[params[0]] = results[r][2]
                self.emc[params[0]] = results[r][3]
        else:
            for d_idx in range(num_dests):
                self.x[d_idx], self.total_cost[d_idx], self.entropy[d_idx], self.emc[d_idx] =\
                    self._load_flow_d((d_idx, c, ret_S))

        if ret_S:
            return self.emc
        else:
            #self.X = np.sum(self.x, axis=0)
            return self.x, self.total_cost, self.entropy, self.emc

    def _load_parallel(self, params):
        d_idxs, c, ret_S = params
        D, N, L = len(d_idxs), len(self.graph.nodes), len(c)
        x = np.zeros(shape=(D, L), dtype=np.float)
        total_cost = np.zeros(D, dtype=np.float)
        entropy = np.zeros(D, dtype=np.float)
        emc = np.zeros(D, dtype=np.float)
        for i, d_idx in enumerate(d_idxs):
            x[i], total_cost[i], entropy[i], emc[i] = self._load_flow_d([d_idx, c, ret_S])
        return x, total_cost, entropy, emc

    def sample_path(self, d_idx, o, fix_len=False):
        alts = np.arange(len(self.graph.nodes))
        p, d = self.p[d_idx], self.graph.dests[d_idx]
        curr_node = o
        node_seq, link_seq = [curr_node], []
        # sampling
        while True:
            next_node = np.random.choice(alts, p=p[curr_node].toarray()[0])
            node_seq.append(next_node)
            link_seq.append(self.graph.link_idxs[(curr_node, next_node)])
            if next_node == d:
                break
            else:
                curr_node = next_node
        # return
        if not fix_len:
            return node_seq, link_seq
        else:
            nodes, links = [[-1 for _ in range(fix_len)] for __ in range(2)]
            nodes[:len(node_seq)] = node_seq
            links[:len(link_seq)] = link_seq
            return nodes, links

class NGEVMCA(Loader):

    def __init__(self, graph, syntax='NGEV', theta=None, alpha=None, parallel=False, tolerance=[1e-100, 1e-50]):
        super().__init__(graph, syntax, theta, alpha, parallel)
        self.tolerance = tolerance
        self._load_flow_d = self._load_flow_d_sle

    def _load_flow_d_vi(self, params):
        # input
        d_idx, c, ret_S = params
        num_nodes = len(self.graph.nodes)
        num_links = len(self.graph.links)
        # output
        x = np.array(num_links, dtype=np.float)
        total_cost, entropy, emc = 0., 0., 0.

        e = np.ones(num_nodes, dtype=np.float)
        d = self.graph.dests[d_idx]
        od_flows = self.graph.od_flows[d]

        # STEP1: COMPUTE WEIGHT MATRIX
        W = self.alpha * np.exp(- self.theta[d_idx, self.graph.senders] * c) # (n_links,)
        W *= (self.graph.senders != d)

        # variables
        z = np.zeros(num_nodes, dtype=np.float) # (n_nodes,)
        b = np.zeros(num_nodes, dtype=np.float) # (n_nodes,)
        z[d] = 1.
        b[d] = 1.

        # solve by iterations
        m_1 = 0
        while True:
            # vector of size (n_links,)
            # Xz(ij) = z(j) ** (theta[i]/theta[j])
            Xz = z[self.graph.receivers] ** (
                    self.theta[d_idx, self.graph.senders]/self.theta[d_idx, self.graph.receivers])

            # update value functions
            zm = np.zeros(num_nodes, dtype=np.float) # (n_nodes,)
            np.add.at(zm, self.graph.senders, W * Xz) # scatter_add (add_to, index, add_from)
            zm += b

            # convergence
            dif = np.sum(np.abs(zm - z))
            m_1 += 1
            z = zm
            ########################################################################
            #*--- tolerance should be 1e-100 for strict primal-dual comparison ---*#
            ########################################################################
            if (dif < self.tolerance[0] and z[self.graph.origins[d]].all() > 0.) or m_1 > 100:
                # print(d_idx, dif, m_1, z)
                break
        assert np.min(z[self.graph.origins[d]]) > 0., 'z includes zeros!!: d={}, z={}, dif={}, m={}'.format(d, z[self.graph.origins[d]], dif, m_1)

        # STEP4: COMPUTE FLOWS
        q0 = np.zeros(num_nodes, dtype=np.float)
        for o, flow in od_flows.items():
            if o != d:
                q0[o] = flow
                # expected minimum cost
                emc += flow * ((-1/self.theta[d_idx,o]) * np.log(z[o]))
        q = q0.copy()

        # end if only S is of interest
        if not ret_S:
            # STEP3: COMPUTE PROBABILITY
            p = (W * Xz) / z[self.graph.senders]

            # solve by iterations
            m_2 = 0
            while True:
                qm = np.zeros(num_nodes, dtype=np.float)
                np.add.at(qm, self.graph.receivers, q[self.graph.senders] * p) # scatter_add (add_to, index, add_from)
                qm += q0

                # convergence check
                dif = np.sum(np.abs(qm - q))
                m_2 += 1
                q = qm
                #######################################################################
                #*--- tolerance should be 1e-50 for strict primal-dual comparison ---*#
                #######################################################################
                if dif < self.tolerance[1] or m_2 > 50:
                    # print(d_idx, dif, m_2, q)
                    break
            # print(q)

            # output
            x = p * q[self.graph.senders]
            total_cost = np.sum(c * x)
            p_up = np.clip(p, self.eps, None)
            entropy = np.sum(
                - (1/self.theta[d_idx, self.graph.senders]) * q[self.graph.senders] *\
                    p_up * (np.log(p_up) - np.log(self.alpha))
            )

        return x, total_cost, entropy, emc

    def _load_flow_d_sle(self, params):
        # input
        d_idx, c, ret_S = params
        num_nodes = len(self.graph.nodes)
        num_links = len(self.graph.links)
        d = self.graph.dests[d_idx]
        od_flows = self.graph.od_flows[d]
        # output
        x = np.array(num_links, dtype=np.float)
        total_cost, entropy, emc = 0., 0., 0.

        # solve by value iteration
        w_edge = self.alpha * np.exp(- self.theta[d_idx, self.graph.senders] * c) # (n_links,)
        w_edge *= (self.graph.senders != d)
        z = np.zeros(num_nodes, dtype=np.float) # (n_nodes,)
        b = np.zeros(num_nodes, dtype=np.float) # (n_nodes,)
        z[d] = 1.
        b[d] = 1.

        m_1 = 0
        while True:
            # vector of size (n_links,)
            # Xz(ij) = z(j) ** (theta[i]/theta[j])
            Xz = z[self.graph.receivers] ** (
                    self.theta[d_idx, self.graph.senders]/self.theta[d_idx, self.graph.receivers])

            # update value functions
            zm = np.zeros(num_nodes, dtype=np.float) # (n_nodes,)
            np.add.at(zm, self.graph.senders, w_edge * Xz) # scatter_add (add_to, index, add_from)
            zm += b

            # convergence
            dif = np.linalg.norm(zm - z)
            # dif = np.linalg.norm(np.log(zm) - np.log(z))
            m_1 += 1
            z = zm
            ########################################################################
            #*--- tolerance should be 1e-100 for strict primal-dual comparison ---*#
            ########################################################################
            if (dif < self.tolerance[0] and z[self.graph.origins[d]].all() > 0.) or (m_1 > 50 and z[self.graph.origins[d]].all() > 0.) or m_1 > 1000:
                # print(d_idx, dif, m_1)
                break
        assert np.min(z[self.graph.origins[d]]) > 0., 'z includes zeros!!: d={}, z={}, dif={}, m={}'.format(d, z[self.graph.origins[d]], dif, m_1)
        if m_1 > 100: print(f'iteration over 100!!; m={m_1} for d={d}')

        # forward computation for link flows
        origins = list(od_flows.keys())
        flows = list(od_flows.values())
        # q = np.zeros(num_nodes, dtype=np.float)
        # q[origins] = flows
        emc = np.sum(flows * ((-1/self.theta[d_idx,origins]) * np.log(z[origins])))

        # end if only S is of interest
        if not ret_S:
            # STEP3: COMPUTE PROBABILITY
            p_edge = (w_edge * Xz) / z[self.graph.senders]
            p = csr_matrix((p_edge, (self.graph.senders, self.graph.receivers)), shape=(num_nodes,num_nodes))
            q0 = csr_matrix((flows, (origins, np.zeros(len(od_flows)))), shape=(num_nodes,1))
            I = sp.identity(num_nodes)
            q = sp.linalg.spsolve((I - p.T), q0)

            # output
            x = p_edge * q[self.graph.senders]
            total_cost = np.sum(c * x)
            p_nonzero = p_edge > 0
            p_up = np.clip(p_edge, self.eps, None)
            entropy = np.sum(
                - (1/self.theta[d_idx, self.graph.senders]) * q[self.graph.senders] *\
                    p_up * (np.log(p_up) - np.log(self.alpha)) * p_nonzero
            )

        return x, total_cost, entropy, emc

    def eval_prob(self, c, d_idx):
        # input
        num_nodes = len(self.graph.nodes)
        num_links = len(self.graph.links)
        d = self.graph.dests[d_idx]

        # STEP1: COMPUTE WEIGHT MATRIX
        W = self.alpha * np.exp(- self.theta[d_idx, self.graph.senders] * c) # (n_links,)
        W *= (self.graph.senders != d)

        # variables
        z = np.zeros(num_nodes, dtype=np.float) # (n_nodes,)
        b = np.zeros(num_nodes, dtype=np.float) # (n_nodes,)
        z[d] = 1.
        b[d] = 1.

        # solve by iterations
        m_1 = 0
        while True:
            # vector of size (n_links,)
            # Xz(ij) = z(j) ** (theta[i]/theta[j])
            Xz = z[self.graph.receivers] ** (
                    self.theta[d_idx, self.graph.senders]/self.theta[d_idx, self.graph.receivers])

            # update value functions
            zm = np.zeros(num_nodes, dtype=np.float) # (n_nodes,)
            np.add.at(zm, self.graph.senders, W * Xz) # scatter_add (add_to, index, add_from)
            zm += b

            # convergence
            dif = np.sum(np.abs(zm - z))
            m_1 += 1
            z = zm

            if (dif < self.tolerance[0] and z.all() > 0.) or m_1 > 100:
                # print(d_idx, dif, m_1, z)
                break
        assert np.min(z) > 0., 'z includes zeros!!: d={}, z={}, dif={}, m={}'.format(d, z, dif, m_1)

        # STEP3: COMPUTE PROBABILITY
        self.p_edge[d_idx] = (W * Xz) / z[self.graph.senders]
        self.p[d_idx] = csr_matrix((self.p_edge[d_idx], (self.graph.senders, self.graph.receivers)), shape=(num_nodes,num_nodes))

class LogitMCA(Loader):

    def __init__(self, graph, syntax='Logit', theta=1.0, parallel=False):
        super().__init__(graph, 'Logit', theta=theta, alpha=1.0, parallel=parallel)

    def _load_flow_d(self, params):
        # input
        d_idx, c, ret_S = params
        num_nodes = len(self.graph.nodes)
        num_links = len(self.graph.links)
        d = self.graph.dests[d_idx]
        od_flows = self.graph.od_flows[d]
        # output
        x = np.array(num_links, dtype=np.float)
        total_cost, entropy, emc = 0., 0., 0.

        # weight matrix of size N x N
        exp_c = np.exp(- self.theta * c) * (self.graph.senders != d)
        W = csr_matrix((exp_c, (self.graph.senders, self.graph.receivers)), shape=(num_nodes,num_nodes))
        I = sp.identity(num_nodes)
        b = csr_matrix(([1.], ([d], [0])), shape=(num_nodes, 1))

        # solve the system of linear equations
        z = sp.linalg.spsolve((I - W), b)
        Xz = z[self.graph.receivers]
        assert np.min(z[self.graph.origins[d]]) > 0., 'z includes zeros!!: d={}, z={}'.format(d, z[self.graph.origins[d]])

        # forward computation for link flows
        origins = list(od_flows.keys())
        flows = list(od_flows.values())
        # q = np.zeros(num_nodes, dtype=np.float)
        # q[origins] = flows
        emc = np.sum(flows * ((-1/self.theta) * np.log(z[origins])))

        # end if only S is of interest
        if not ret_S:
            # STEP3: COMPUTE PROBABILITY
            p_edge = (exp_c * Xz) / z[self.graph.senders]
            p = csr_matrix((p_edge, (self.graph.senders, self.graph.receivers)), shape=(num_nodes,num_nodes))
            q0 = csr_matrix((flows, (origins, np.zeros(len(od_flows)))), shape=(num_nodes,1))
            q = sp.linalg.spsolve((I - p.T), q0)

            # output
            x = p_edge * q[self.graph.senders]
            total_cost = np.sum(c * x)
            p_nonzero = p_edge > 0
            p_up = np.clip(p_edge, self.eps, None)
            entropy = np.sum(
                - (1/self.theta) * q[self.graph.senders] * p_up * np.log(p_up) * p_nonzero
            )

        return x, total_cost, entropy, emc

    def eval_prob(self, c, d_idx):
        # input
        num_nodes = len(self.graph.nodes)
        num_links = len(self.graph.links)
        d = self.graph.dests[d_idx]

        # weight matrix of size N x N
        exp_c = np.exp(- self.theta * c) * (self.graph.senders != d)
        W = csr_matrix((exp_c, (self.graph.senders, self.graph.receivers)), shape=(num_nodes,num_nodes))
        I = sp.identity(num_nodes)
        b = csr_matrix(([1.], ([d], [0])), shape=(num_nodes, 1))

        # solve the system of linear equations
        z = sp.linalg.spsolve((I - W), b)
        Xz = z[self.graph.receivers]
        assert np.min(z) > 0., 'z includes zeros!!: d={}, z={}'.format(d, z)

        # STEP3: COMPUTE PROBABILITY
        self.p_edge[d_idx] = (exp_c * Xz) / z[self.graph.senders]
        self.p[d_idx] = csr_matrix((self.p_edge[d_idx], (self.graph.senders, self.graph.receivers)), shape=(num_nodes,num_nodes))

class NGEVDial(Loader):

    def __init__(self, graph, syntax='NGEV', theta=None, alpha=None, parallel=False):
        super().__init__(graph, syntax, theta, alpha, parallel)

    def _load_flow_d(self, params):
        # input
        d_idx, c, ret_S = params
        num_links = len(self.graph.links)
        num_nodes = len(self.graph.nodes)
        d = self.graph.dests[d_idx]
        od_flows = self.graph.od_flows[d]

        # output
        x = np.zeros(num_links, dtype=np.float)
        total_cost, entropy, emc = 0., 0., 0.

        # shortest path computing
        w_rev = csr_matrix((c, (self.graph.receivers, self.graph.senders)), shape=(num_nodes,num_nodes))
        dist_to_d, successor = shortest_path(w_rev, method='D', directed=True, indices=d, return_predecessors=True)
        dist_to_d = np.array(dist_to_d)
        backward = np.argsort(dist_to_d) # from the nearest to the the farthest to d
        forward = np.flipud(backward) # from the farthest to the nearest to d

        # link likelihood
        likelihoods = self.alpha * np.exp(-self.theta[d_idx, self.graph.senders] *\
                                (c + dist_to_d[self.graph.receivers] - dist_to_d[self.graph.senders]))
        likelihoods *= (dist_to_d[self.graph.senders] > dist_to_d[self.graph.receivers])

        # backward computation for link weights
        link_weights = np.zeros(num_links, dtype=np.float) # link weights
        node_weights = np.zeros(num_nodes, dtype=np.float) # node weights
        node_weights[d] = 1

        for i in backward:
            for j in self.graph.forward_stars[i]:
                a = self.graph.link_idxs[(i, j)]
                node_weights[i] += link_weights[a]

            for h in self.graph.backward_stars[i]:
                a = self.graph.link_idxs[(h, i)]
                if i == d:
                    link_weights[a] = likelihoods[a]
                else:
                    link_weights[a] = likelihoods[a] * (node_weights[i]**(self.theta[d_idx,h]/self.theta[d_idx,i]))

        # forward computation for link flows
        origins = list(od_flows.keys())
        flows = list(od_flows.values())
        q = np.zeros(num_nodes, dtype=np.float)
        q[origins] = flows
        emc = np.sum(flows * ((-1/self.theta[d_idx,origins]) * np.log(node_weights[origins])))
        node_flows = np.zeros(num_nodes, dtype=np.float)

        if not ret_S:
            p = link_weights/node_weights[self.graph.senders]
            p_nonzeros = (p > 0)
            p_up = np.clip(p, self.eps, None)

            for i in forward:
                if i == d:
                    continue
                node_flows[i] = q[i]
                for h in self.graph.backward_stars[i]:
                    a = self.graph.link_idxs[(h, i)]
                    node_flows[i] += x[a]
                for j in self.graph.forward_stars[i]:
                    a = self.graph.link_idxs[(i, j)]
                    x[a] = node_flows[i] * p[a]

            total_cost = np.sum(c * x)
            entropy = np.sum( - (1/self.theta[d_idx, self.graph.senders]) * x * \
                                (np.log(p_up) - np.log(self.alpha)) * p_nonzeros)

        return x, total_cost, entropy, emc

class LogitDial(Loader):

    def __init__(self, graph, syntax='Logit', theta=1.0, parallel=False):
        super().__init__(graph, 'Logit', theta=theta, alpha=1.0, parallel=parallel)

    def _load_flow_d(self, params):
        # input
        d_idx, c, ret_S = params
        num_links = len(self.graph.links)
        num_nodes = len(self.graph.nodes)
        d = self.graph.dests[d_idx]
        od_flows = self.graph.od_flows[d]

        # output
        x = np.zeros(num_links, dtype=np.float)
        total_cost, entropy, emc = 0., 0., 0.

        # shortest path computing
        w_rev = csr_matrix((c, (self.graph.receivers, self.graph.senders)), shape=(num_nodes,num_nodes))
        dist_to_d, successor = shortest_path(w_rev, method='D', directed=True, indices=d, return_predecessors=True)
        dist_to_d = np.array(dist_to_d)
        backward = np.argsort(dist_to_d) # from the nearest to the the farthest to d
        forward = np.flipud(backward) # from the farthest to the nearest to d

        # link likelihood
        likelihoods = np.exp(-self.theta * (c + dist_to_d[self.graph.receivers] - dist_to_d[self.graph.senders]))
        likelihoods *= (dist_to_d[self.graph.senders] > dist_to_d[self.graph.receivers])

        # backward computation for link weights
        link_weights = np.zeros(num_links, dtype=np.float) # link weights
        node_weights = np.zeros(num_nodes, dtype=np.float) # node weights
        node_weights[d] = 1

        for i in backward:
            for j in self.graph.forward_stars[i]:
                a = self.graph.link_idxs[(i, j)]
                node_weights[i] += link_weights[a]

            for h in self.graph.backward_stars[i]:
                a = self.graph.link_idxs[(h, i)]
                if i == d:
                    link_weights[a] = likelihoods[a]
                else:
                    link_weights[a] = likelihoods[a] * node_weights[i]

        # forward computation for link flows
        origins = list(od_flows.keys())
        flows = list(od_flows.values())
        emc = np.sum(flows * ((-1/self.theta) * np.log(node_weights[origins])))

        q = np.zeros(num_nodes, dtype=np.float)
        q[origins] = flows
        node_flows = np.zeros(num_nodes, dtype=np.float)

        if not ret_S:
            p = link_weights/node_weights[self.graph.senders]
            p_nonzeros = (p > 0)
            p_up = np.clip(p, self.eps, None)

            for i in forward:
                if i == d:
                    continue
                node_flows[i] = q[i]
                for h in self.graph.backward_stars[i]:
                    a = self.graph.link_idxs[(h, i)]
                    node_flows[i] += x[a]
                for j in self.graph.forward_stars[i]:
                    a = self.graph.link_idxs[(i, j)]
                    x[a] = node_flows[i] * p[a]

            total_cost = np.sum(c * x)
            entropy = np.sum( - (1/self.theta) * x * np.log(p_up) * p_nonzeros)

        return x, total_cost, entropy, emc

class Probit(Loader):

    def __init__(self, graph, theta=1.0, n_draws=1000, seed_=111, parallel=False):
        super().__init__(graph, theta=theta, parallel=parallel)
        self.n_draws = n_draws
        np.random.seed(seed_)

    def load_flow(self, cost=None, ret_S=False, error_mode=False, x_ref=None):

        num_nodes = len(self.graph.nodes)
        num_links = len(self.graph.links)
        num_dests = len(self.graph.dests)

        # cost used in this loading
        if cost is not None:
            c = cost.copy()
        else:
            c = self.c0.copy()

        # draw link costs
        sigma = np.sqrt(self.theta * c)
        sampled_costs = np.random.normal(c, sigma, size=(self.n_draws, num_links))
        sampled_costs = np.clip(sampled_costs, 0.1, None)

        # outputs
        self.x = np.zeros(shape=(num_dests, num_links), dtype=np.float)

        if error_mode: # mode to analyze the error
            print('run in error analysis mode')
            timer = Timer()
            enorm, max_erel, max_stddev, sum_sbx, max_sbx, runtime = np.zeros((6,self.n_draws), dtype=np.float)
            nonzero = np.nonzero(x_ref)
            x_mean, x_var = np.zeros((2,num_links), dtype=np.float)
            for r in tqdm(range(self.n_draws)):
                x = np.zeros((num_dests,num_links), dtype=np.float)
                w_rev = csr_matrix((sampled_costs[r], (self.graph.receivers, self.graph.senders)), shape=(num_nodes,num_nodes))
                n_cpu = mp.cpu_count()
                if self.parallel and num_dests > n_cpu:
                    n_threads = n_cpu // 2
                    batch_size = num_dests // n_threads
                    p = mp.Pool(n_threads)
                    d_arange = np.arange(num_dests)
                    particles = [(num_dests + i) // n_threads for i in range(n_threads)]
                    cum = np.cumsum(particles)
                    paramsList = [
                        [d_arange[cum[r-1]:cum[r]], w_rev] if r > 0 else [d_arange[:cum[r]], w_rev]
                        for r in range(n_threads)
                        ]
                    results = p.map(self._load_flow_d, paramsList)
                    p.close()
                    for d_idx, xd in enumerate(results):
                        x[d_idx] = xd
                else:
                    for d_idx in range(num_dests):
                        x[d_idx] = self._load_flow_d((d_idx, w_rev))
                runtime[r] = timer.stop()
                x = np.sum(x, axis=0)
                if r == 0:
                    x_mean += x
                else:
                    x_nonzero = np.nonzero(x)
                    x_mean_new = (r * x_mean + x) / (r + 1)
                    # x_var = (r * (x_var + x_mean**2) + x**2)/(r+1) - x_mean_new**2
                    x_var = (1 - (1/r)) * x_var + (r+1) * (x_mean_new - x_mean)**2
                    stddev = np.sqrt(x_var/(r+1))
                    x_mean = x_mean_new
                    enorm[r] = np.linalg.norm(x_mean - x_ref)
                    max_erel[r] = np.max(np.abs(x_mean[nonzero] - x_ref[nonzero])/x_ref[nonzero])
                    max_stddev[r] = np.max(stddev)
                    sum_sbx[r] = np.sum(stddev)/np.sum(x)
                    max_sbx[r] = np.max(stddev[x_nonzero]/x[x_nonzero])

            return enorm, max_erel, max_stddev, sum_sbx, max_sbx, runtime

        if self.parallel:
            print('run in parallel mode')
            # Assign flows (multiprocessing)
            n_cpu = mp.cpu_count()
            n_threads = n_cpu
            ic(n_threads)
            p = mp.Pool(n_threads)
            batch_size = self.n_draws // n_threads
            particles = [(self.n_draws + i) // n_threads for i in range(n_threads)]
            cum = np.cumsum(particles)
            paramsList = [sampled_costs[cum[r-1]:cum[r]] if r > 0 else sampled_costs[:cum[r]] for r in range(n_threads)]
            # paramsList = [sampled_costs[r] for r in range(self.n_draws)]
            results = p.map(self._load_parallel, paramsList)
            p.close()
            # self.x = np.mean(np.concatenate(x_list, axis=0), axis=0)
            cum_R = 0
            for x, R in results:
                self.x = (cum_R * self.x + R * x) / (cum_R + R)
                cum_R += R
        else:
            # print('run in non-parallel mode')
            for r in range(self.n_draws):
                w_rev = csr_matrix((sampled_costs[r], (self.graph.receivers, self.graph.senders)), shape=(num_nodes,num_nodes))
                for d_idx in range(num_dests):
                    self.x[d_idx] += self._load_flow_d((d_idx, w_rev)) #sampled_costs[r],
            self.x /= self.n_draws

        return self.x, None, None, None

    def _load_parallel(self, c):
        # inputs
        num_nodes = len(self.graph.nodes)
        num_dests = len(self.graph.dests)
        R, num_links = c.shape

        x = np.zeros(shape=(num_dests, num_links), dtype=np.float)
        for r in range(R):
            w_rev = csr_matrix((c[r], (self.graph.receivers, self.graph.senders)), shape=(num_nodes,num_nodes))
            for d_idx in range(num_dests):
                x[d_idx] += self._load_flow_d((d_idx, w_rev))
        x /= R
        return x, R

    def _load_flow_d(self, params):
        # input
        d_idx, w_rev = params #c,
        num_links = len(self.graph.links)
        num_nodes = len(self.graph.nodes)
        d = self.graph.dests[d_idx]
        od_flows = self.graph.od_flows[d]

        # output
        x = np.zeros(num_links, dtype=np.float)

        ##### scipy.sparse.shortest path #####
        _, successor = shortest_path(w_rev, method='D', directed=True, indices=d, return_predecessors=True)
        for o, flow in od_flows.items():
            if o == d:
                continue
            i = o
            while True:
                j = successor[i]
                a = self.graph.link_idxs[(i, j)]
                x[a] += flow
                if j == d:
                    break
                i = j
        return x
