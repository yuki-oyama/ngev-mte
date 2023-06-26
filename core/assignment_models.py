import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time
from icecream import ic

class NGEVSUE(object):

    def __init__(self,
                graph,
                beta=[0.15, 4.],
                threshold=[1e-3, 1e-3, 1e-3, 1e-3],
                m_min=100,
                m_max=5000,
                ref_data=None,
                ref_opt=None,
                print_step=50,
                link_performance='polynomial',
                dynamic_accuracy=False
                ):

        # network
        self.graph = graph
        self.c0 = graph.c0
        self.mu = graph.mu

        # params
        self.eps = 1e-8
        self.inf = 1e+10
        self.beta = beta
        self.threshold = {
            'x': threshold[0] if threshold[0] is not None else 1.,
            'c': threshold[1] if threshold[1] is not None else 1.,
            'grad_c': threshold[2] if threshold[2] is not None else 1.,
            'Z': threshold[3] if threshold[3] is not None else 1.
        }
        self.m_min = m_min
        self.m_max = m_max
        self.print_step = print_step
        self.dynamic_accuracy = dynamic_accuracy

        # reference
        self.ref_x = np.array(ref_data['flow']) if ref_data is not None else None
        self.ref_c = np.array(ref_data['cost']) if ref_data is not None else None
        self.ref_Z = ref_opt

        # functions
        self.fn_type = link_performance
        self.c_fn = self.set_c_fn()
        self.inv_c_fn = self.set_inv_c_fn()

        # record time
        self.cpu_time = {}
        self.time_limit = 36*60*60 # n hours

        # variables
        self.m = 0
        self.x = {}
        self.X = {}
        self.c = {0: self.c0}
        self.inv_c = {}
        self.grad_c = {}
        self.rel_dif_x = {}
        self.rel_dif_c = {}
        self.rel_grads = {}
        self.rel_dif_Z = {}

        # objectives
        self.Z = {}
        self.C = {}
        self.int_C = {}
        self.H = {}
        self.S = {}

    def set_c_fn(self):
        if self.fn_type == 'polynomial':
            return lambda x: self.c0 * (1 + self.beta[0] * ((x/self.mu)**self.beta[1]))
        elif self.fn_type == 'linear':
            return lambda x: self.c0 + self.beta[0] * x

    def set_inv_c_fn(self):
        if self.fn_type == 'polynomial':
            return lambda c: self.mu * ( ((1/self.beta[0]) * ((c / self.c0) - 1))**(1/self.beta[1]) )
        elif self.fn_type == 'linear':
            return lambda c: (c - self.c0) / self.beta[0]

    def update_cost(self, x):
        return self.c_fn(x)

    def compute_inv_c(self, c):
        return self.inv_c_fn(c)

    def integral_inv_c(self, c):
        C_inv = np.sum(
            (self.beta[1] / (self.beta[1] + 1)) * \
                self.beta[0] * self.c0 * self.mu * \
                (((c - self.c0) / (self.beta[0] * self.c0))**((self.beta[1]+1)/self.beta[1]))
        )
        return C_inv

    def integral_c(self, x):
        C = np.sum(
            self.c0 * (
                x + (self.beta[0]*self.mu/(self.beta[1]+1)) * ((x/self.mu)**(self.beta[1]+1))
            )
        )
        return C

    def set_entropy_fnc(self, model):
        if type(model.theta) == float:
            self.get_entropy = self.get_entropy_logit
        else:
            self.get_entropy = self.get_entropy_ngev

    def get_entropy_ngev(self, model, x):
        H = 0
        x_nonzero = (x > 0)
        x_up = np.clip(x, self.eps, None)
        for d_idx, d in enumerate(self.graph.dests):
            # link entropy
            H += np.sum(
                - (1/model.theta[d_idx, self.graph.senders] * x_up[d_idx] * \
                    (np.log(x_up[d_idx]) - np.log(model.alpha))) * x_nonzero[d_idx]
            )
            # node entropy
            z = np.zeros(len(self.graph.nodes), dtype=np.float)
            np.add.at(z, self.graph.senders, x[d_idx])
            z_nonzero = z > 0
            z_up = np.clip(z, self.eps, None)
            H += np.sum(
                (1/model.theta[d_idx]) * z_up * np.log(z_up) * z_nonzero
            )
        return H

    def get_entropy_logit(self, model, x):
        H = 0
        x_nonzero = (x > 0)
        x_up = np.clip(x, self.eps, None)
        for d_idx, d in enumerate(self.graph.dests):
            # link entropy
            H += np.sum(
                - (1/model.theta * x_up[d_idx] * np.log(x_up[d_idx])) * x_nonzero[d_idx]
            )
            # node entropy
            z = np.zeros(len(self.graph.nodes), dtype=np.float)
            np.add.at(z, self.graph.senders, x[d_idx])
            z_nonzero = z > 0
            z_up = np.clip(z, self.eps, None)
            H += np.sum(
                (1/model.theta) * z_up * np.log(z_up) * z_nonzero
            )
        return H

    def get_total_cost(self, x, c):
        return np.dot(x, c)

    def update_accuracy(self, model, decay=None):
        model.tolerance[0] *= decay

    def write_results(self, file_path):
        # this will be updated for each of dual and primal
        pass

    def convergence_test(self):
        conv_x, conv_c, conv_gradc, conv_Z = True, True, True, True
        m = self.m

        # relative diffrence
        ref_x = self.X[m] if self.ref_x is None else self.ref_x
        ref_c = self.c[m] if self.ref_c is None else self.ref_c
        self.rel_dif_x[m] = np.max(
            np.abs( (self.X[m+1] - ref_x)/np.clip(ref_x, 1., None) )
            ) if self.threshold['x'] < 1. else 0.
        self.rel_dif_c[m] = np.max(
            np.abs( (self.c[m+1] - ref_c)/np.clip(ref_c, 1., None) )
            ) if self.threshold['c'] < 1. else 0.
        idx_ = np.nonzero(self.X[m])
        self.rel_grads[m] = np.mean(
            np.abs(self.grad_c[m][idx_])/self.X[m][idx_]
            ) if self.threshold['grad_c'] < 1. else 0.
        # self.rel_dif_Z[m] = 0. if self.ref_Z is None else np.abs(self.ref_Z - self.Z[m+1])/self.ref_Z
        if m == 0:
            self.rel_dif_Z[m] = 0.
        elif self.ref_Z is None:
            self.rel_dif_Z[m] = np.abs(self.Z[m] - self.Z[m+1])/self.Z[m]
        else:
            self.rel_dif_Z[m] = np.abs(self.ref_Z - self.Z[m+1])/self.ref_Z

        # convergence
        conv_x = self.rel_dif_x[m] < self.threshold['x']
        conv_c = self.rel_dif_c[m] < self.threshold['c']
        conv_gradc = self.rel_grads[m] < self.threshold['grad_c']
        conv_Z = self.rel_dif_Z[m] < self.threshold['Z']

        # record time
        self.cpu_time[m] = time.perf_counter() - self.t0
        time_over = self.cpu_time[m] > self.time_limit

        convergence = (m > self.m_max) or time_over or\
            ( m > self.m_min and (conv_x and conv_c and conv_gradc and conv_Z) )
        return convergence

    def write_objectives(self, model, file_path):
        self.set_entropy_fnc(model)
        m = self.m
        H = self.get_entropy(model, self.x[m])
        intC = self.integral_c(self.X[m])
        invC = self.integral_inv_c(self.c[m])
        C = self.get_total_cost(self.X[m], self.c[m])
        x, Cd, Hd, Sd = model.load_flow(cost=self.c[m])
        C2 = np.sum(Cd)
        H2 = np.sum(Hd)
        Zd = np.sum(Cd - Hd) - invC
        Zp = intC - H

        with open(file_path, 'w') as f:
            header = 'entropy,total_cost,entropy_load,total_cost_loading,int_C,int_inv_C,emc,Zd,Zp \n'
            f.write(header)
            txt = str(H) + ',' + str(C) + ',' + str(H2) + ',' + str(C2) + ',' + str(intC) + ',' + str(invC) +\
                    ',' + str(np.sum(Sd)) + ',' + str(Zd) + ',' + str(Zp) + '\n'
            f.write(txt)

    def print_record(self, m, grads=False):
        print('m:{} \t wrap:{}s \t dif:{} \t grad:{} \t difZ:{} \t Z:{} \t c:{} \t x:{}'.format(
            m,
            self.cpu_time[m] - self.cpu_time[m-1] if m > 0 else self.cpu_time[m],
            self.rel_dif_c[m] if grads else self.rel_dif_x[m],
            self.rel_grads[m] if grads else 'None',
            self.rel_dif_Z[m] if self.rel_dif_Z[m] < self.inf else 'None',
            self.Z[m+1],
            self.c[m][:5],
            self.X[m][:5]
            )
        )

    def print_variables(self, var_list):
        for a, link in enumerate(self.graph.links):
            txt = 'link:{}'.format(link)
            for v in var_list:
                txt += '\t' + str(v[a])
            print(txt)

class Dual(NGEVSUE):

    def __init__(self,
                graph,
                optimizer,
                beta=[1.,4.],
                threshold=[None, 1e-3, 1e-3, None],
                m_min=1000,
                m_max=10000,
                ref_data=None,
                ref_opt=None,
                print_step=50
                ):

        # NGEV assignment class
        super().__init__(graph,
                        beta=beta,
                        threshold=[None, threshold[1], threshold[2], threshold[3]],
                        m_min=m_min, m_max=m_max, ref_data=ref_data, ref_opt=ref_opt, print_step=print_step)

        # optimizer
        self.optimizer = optimizer

    def solve(self, model):

        def g(c):
            x, _, _, S = model.load_flow(cost=c)
            X = np.sum(x, axis=0)
            inv_c = self.compute_inv_c(c)
            grad = X - inv_c # for maximize (then take negative for minimize)
            F = np.sum(S) - self.integral_inv_c(c)
            return -grad, x, inv_c, -F

        def f(c):
            S = model.load_flow(cost=c, ret_S=True)
            F = np.sum(S) - self.integral_inv_c(c)
            return -F # maximize -> minimize

        # initialize optimizer
        self.optimizer.initialize(
            f=f,
            g=g,
            x0=self.c0,
            clip=(self.c0, None)
        )
        # record time
        self.t0 = time.perf_counter()

        while True:
            m = self.m
            # Update model by optimizer
            self.grad_c[m], self.x[m], self.inv_c[m], self.c[m+1], self.Z[m+1] =\
                self.optimizer.model_update(m)
            self.grad_c[m] = -self.grad_c[m]
            self.Z[m+1] = -self.Z[m+1]
            self.X[m] = np.sum(self.x[m], axis=0)

            # Convergence test
            if self.convergence_test():
                print('ALGORITHM CONVERGED!')
                self.print_record(m, grads=True)
                break
            elif self.m % self.print_step == 0:
                self.print_record(m, grads=True)

                # update minimum step size: keep it large until a certain point for fast solution update
                if self.m > 0 and self.m % 100 == 0 and \
                    hasattr(self.optimizer, 'min_s') and self.rel_dif_Z[m] < 1e-5:
                    self.optimizer.min_s *= 0.1

            self.m += 1

            # dynamic_accuracy
            if self.dynamic_accuracy:
                self.update_accuracy(model, decay=0.1)

    def write_results(self, file_path):
        # for step_size
        BT = False
        if hasattr(self.optimizer, 'with_BT'):
            BT = self.optimizer.with_BT

        if self.ref_c is None:
            dif_c = {
                m: np.max( np.abs( (self.c[m] - self.c[self.m])/np.clip(self.c[self.m], 1., None) ))
                for m in range(1,self.m)
            }
        if self.ref_Z is None:
            dif_Z = {
                m: np.abs(self.Z[self.m] - self.Z[m])/ self.Z[self.m]
                for m in range(1,self.m)
            }

        with open(file_path, 'w') as f:
            header = 'm,objective_value,relative_obj_dif,relative_cost_dif,grad_mean,relative_grad_mean,CPU_time,step,rdifc,rdifz\n'
            f.write(header)
            for m in range(1,self.m):
                step = self.optimizer.s[m] if BT and m in self.optimizer.s.keys() else 0
                txt = str(m) + ',' + str(self.Z[m]) + ',' + str(self.rel_dif_Z[m]) + ',' + str(self.rel_dif_c[m]) +\
                        ',' + str(np.mean(self.grad_c[m])) + ',' + str(np.mean(self.grad_c[m]/self.X[m])) +\
                        ',' + str(self.cpu_time[m]) + ',' + str(step) + ','
                if self.ref_c is None:
                    txt += str(dif_c[m])
                txt += ','
                if self.ref_Z is None:
                    txt += str(dif_Z[m])
                txt += '\n'
                f.write(txt)

class Primal(NGEVSUE):

    def __init__(self,
                graph,
                line_search,
                beta=[1.,4.],
                threshold=[1e-3, None, None, None],
                m_min=10,
                m_max=500,
                ref_data=None,
                ref_opt=None,
                print_step=50
                ):

        # NGEV SUE class
        super().__init__(graph,
                        beta=beta,
                        threshold=[threshold[0], None, None, threshold[3]],
                        m_min=m_min, m_max=m_max, ref_data=ref_data, ref_opt=ref_opt, print_step=print_step)

        # line search
        self.line_search = line_search

        # variables
        self.alpha = {}
        self.y = {}
        self.d = {}

    def initialize_variables(self, model):
        # initialize variables
        self.m = 0
        self.x[0], self.C[0], self.H[0], self.S[0] =\
            model.load_flow(cost=self.c[0])
        self.X[0] = np.sum(self.x[0], axis=0)
        self.int_C[0] = self.integral_c(self.X[0])
        self.Z[0] = self.int_C[0] - np.sum(self.H[0])

        # define entropy function
        self.set_entropy_fnc(model)

        # define line search
        def f(x): return self.integral_c(np.sum(x, axis=0)) - self.get_entropy(model, x)
        self.line_search.define(f)
        # start time
        self.t0 = time.perf_counter()

    def solve(self, model):
        # STEP0: Initialization
        # initialize variables
        self.initialize_variables(model)

        while True:
            m = self.m

            # STEP1: NGEV assignment for directional vector
            #print('Step1: Assignment starting... at {}'.format(time.perf_counter() - self.t0))
            self.c[m] = self.update_cost(self.X[m])
            self.y[m], _, _, _ = model.load_flow(cost=self.c[m])
            self.d[m] = self.y[m] - self.x[m]

            # STEP2: Line Search to optimize Step Size
            #print('Step2: Line search starting... at {}'.format(time.perf_counter() - self.t0))
            self.alpha[m] = self.line_search._do(m, self.x[m], self.d[m])
            # print('m:{}, alpha:{}'.format(m, self.alpha[m]))

            # STEP3: Update solution
            #print('Step3: Update starting... at {}'.format(time.perf_counter() - self.t0))
            self.x[m+1] = self.x[m] + self.alpha[m] * self.d[m]
            self.X[m+1] = np.sum(self.x[m+1], axis=0)
            self.Z[m+1] = self.integral_c(self.X[m+1]) - self.get_entropy(model, self.x[m+1])

            # STEP4: Convergence test
            if self.convergence_test():
                print('ALGORITHM CONVERGED!')
                self.finish(m)
                break
            elif self.m % self.print_step == 0: # print
                self.print_record(m)

            # update MTA tolerance
            if self.dynamic_accuracy:
                if self.alpha[m] < self.eps and model.tolerance[0] > 1e-100: # and m < self.m_min:
                    print(f'Solution is not properly updated: m={m}, alpha={self.alpha[m]}')
                    # print('You may want to try smaller tolerance for MCA.')
                    self.update_accuracy(model, decay=1e-10)
                    print(f'Modified the MCA tolerance: {model.tolerance[0]}')
                else:
                    self.update_accuracy(model, decay=0.1)

            self.m += 1

    def finish(self, m):
        self.print_record(m)
        self.inv_c = {m: self.compute_inv_c(self.c[m])}
        self.grad_c = {m: self.X[m] - self.inv_c[m]}

    def write_results(self, file_path):
        if self.ref_x is None:
            dif_x = {
                m: np.max( np.abs( (self.X[m] - self.X[self.m])/np.clip(self.X[self.m], 1., None) ))
                for m in range(1,self.m)
            }
        if self.ref_Z is None:
            dif_Z = {
                m: np.abs(self.Z[self.m] - self.Z[m])/ self.Z[self.m]
                for m in range(1,self.m)
            }
        with open(file_path, 'w') as f:
            header = 'm,objective_value,relative_obj_dif,relative_flow_dif,CPU_time,rdifx,rdifz\n'
            f.write(header)
            for m in range(1,self.m):
                txt = str(m) + ',' + str(self.Z[m]) + ',' + str(self.rel_dif_Z[m]) + ',' + str(self.rel_dif_x[m]) +\
                        ',' + str(self.cpu_time[m]) + ','
                if self.ref_x is None:
                    txt += str(dif_x[m])
                txt += ','
                if self.ref_Z is None:
                    txt += str(dif_Z[m])
                txt += '\n'
                f.write(txt)

class Probit(NGEVSUE):

    def __init__(self,
                graph,
                line_search,
                beta=[1.,4.],
                threshold=[1e-3, None, None, None],
                m_min=10,
                m_max=500,
                ref_data=None,
                ref_opt=None,
                print_step=50
                ):

        # NGEV SUE class
        super().__init__(graph,
                        beta=beta,
                        threshold=[threshold[0], None, None, threshold[3]],
                        m_min=m_min, m_max=m_max, ref_data=ref_data, ref_opt=ref_opt, print_step=print_step)

        # line search
        self.line_search = line_search

        # variables
        self.alpha = {}
        self.y = {}
        self.d = {}

    def initialize_variables(self, model):
        # initialize variables
        self.m = 0
        self.x[0], _, _, _ = model.load_flow(cost=self.c[0])
        self.X[0] = np.sum(self.x[0], axis=0)

        # start time
        self.t0 = time.perf_counter()

    def solve(self, model):
        # STEP0: Initialization
        # initialize variables
        self.initialize_variables(model)

        while True:
            m = self.m

            # STEP1: NGEV assignment for directional vector
            #print('Step1: Assignment starting... at {}'.format(time.perf_counter() - self.t0))
            self.c[m] = self.update_cost(self.X[m])
            self.y[m], _, _, _ = model.load_flow(cost=self.c[m])
            self.d[m] = self.y[m] - self.x[m]

            # STEP2: Line Search to optimize Step Size
            #print('Step2: Line search starting... at {}'.format(time.perf_counter() - self.t0))
            self.alpha[m] = self.line_search._do(m, self.x[m], self.d[m])
            # print('m:{}, alpha:{}'.format(m, self.alpha[m]))

            # STEP3: Update solution
            #print('Step3: Update starting... at {}'.format(time.perf_counter() - self.t0))
            self.x[m+1] = self.x[m] + self.alpha[m] * self.d[m]
            self.X[m+1] = np.sum(self.x[m+1], axis=0)

            # STEP4: Convergence test
            if self.convergence_test():
                print('ALGORITHM CONVERGED!')
                self.finish(m)
                break
            elif self.m % self.print_step == 0: # print
                self.print_record(m)

            self.m += 1

    def finish(self, m):
        self.print_record(m)
        self.inv_c = {m: self.compute_inv_c(self.c[m])}
        self.grad_c = {m: self.X[m] - self.inv_c[m]}

    def convergence_test(self):
        conv_x = True
        m = self.m

        # relative diffrence
        ref_x = self.X[m] if self.ref_x is None else self.ref_x
        self.rel_dif_x[m] = np.max(
            np.abs( (self.X[m+1] - ref_x)/np.clip(ref_x, 1., None) )
            ) if self.threshold['x'] < 1. else 0.

        # convergence
        conv_x = self.rel_dif_x[m] < self.threshold['x']

        # record time
        self.cpu_time[m] = time.perf_counter() - self.t0
        time_over = self.cpu_time[m] > self.time_limit

        convergence = (m > self.m_max) or time_over or (m > self.m_min and conv_x)
        return convergence

    def write_objectives(self, model, filepath):
        pass

    def print_record(self, m):
        print('m:{} \t wrap:{}s \t dif:{} \t c:{} \t x:{}'.format(
            m,
            self.cpu_time[m] - self.cpu_time[m-1] if m > 0 else self.cpu_time[m],
            self.rel_dif_x[m],
            self.c[m][:5],
            self.X[m][:5]
            )
        )

    def write_results(self, file_path):
        if self.ref_x is None:
            dif_x = {
                m: np.max( np.abs( (self.X[m] - self.X[self.m])/np.clip(self.X[self.m], 1., None) ))
                for m in range(1,self.m)
            }

        with open(file_path, 'w') as f:
            header = 'm,relative_flow_dif,CPU_time,rdifx\n'
            f.write(header)
            for m in range(1,self.m):
                txt = str(m) + ',' + str(self.rel_dif_x[m]) + ',' + str(self.cpu_time[m]) + ','
                if self.ref_x is None:
                    txt += str(dif_x[m])
                txt += '\n'
                f.write(txt)
