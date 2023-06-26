import numpy as np
import matplotlib.pyplot as plt

class Comparator(object):

    def __init__(self, ref_val=None):
        self.models = []
        self.model_names = []
        self.max_iter = 0
        self.opt_Z = 0. if ref_val is None else float(ref_val[' objective_value '][0])
        self.max_Z = 0.
        self.avg_Z = 0.
        self.colors = ['r', 'g', 'b', 'y', 'k', 'r', 'g', 'b', 'y', 'k']
        self.markers = ['o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^']

    def add_model(self, model, name):
        self.models.append(model)
        self.model_names.append(name)
        if self.max_iter < model.m:
            self.max_iter = model.m
        # update objective value
        self.avg_Z = ((self.avg_Z * (len(self.models)-1)) + model.Z[model.m])/len(self.models)
        if self.max_Z < model.Z[model.m]:
            self.max_Z = model.Z[model.m]

    def write_results(self, file_path):
        with open(file_path, 'w') as f:
            header = 'method\tnetwork_size\tnum_nodes\tnum_links\tnum_ods\tnum_dests\titerations' +\
                        '\tCPU_time\tCPU_time_iter\tCPU_time_1stiter\tobjective_value\trelative_dif' +\
                        '\tlink_util_mean\tlink_util_max\tlink_util_min\tlink_util_var\n'
            f.write(header)
            for model, model_name in zip(self.models, self.model_names):
                method_name, network_size = model_name.split('_')
                graph = model.graph
                m = model.m
                link_util = model.X[m]/model.mu
                rel_dif = model.rel_dif_x[m] if m in model.rel_dif_x.keys() else model.rel_dif_c[m]
                txt = method_name + '\t' + str(network_size) + '\t' + str(len(graph.nodes)) + \
                         '\t' + str(len(graph.links)) + '\t' + str(graph.num_ods) + '\t' + str(len(graph.dests)) + '\t' + str(m) + \
                         '\t' + str(model.cpu_time[m]-model.cpu_time[0]) + '\t' + str((model.cpu_time[m]-model.cpu_time[0])/m) + '\t' + str(model.cpu_time[0]) +\
                         '\t' + str(model.Z[m]) + '\t' + str(rel_dif) +\
                         '\t' + str(np.mean(link_util)) + '\t' + str(np.max(link_util)) + '\t' + str(np.min(link_util)) + '\t' + str(np.var(link_util)) + '\n'
                f.write(txt)

    def visualize_results(self, file_dir_=None, x_axis='Iteration'):

        # modify cpu time
        for model in self.models:
            model.cpu_time = {
                m: model.cpu_time[m] - model.cpu_time[0]
                for m in model.cpu_time.keys()
            }

        # objective
        plt.figure(figsize=(10,8))
        for k, model in enumerate(self.models):
            if x_axis == 'Iteration':
                x = np.arange(1,model.m)
            elif x_axis == 'CPU time':
                x = [model.cpu_time[m] for m in range(1,model.m)]
            # y = [np.log10(np.abs(model.Z[m]-self.max_Z)/self.max_Z) for m in range(1,model.m)]
            opt_Z = self.opt_Z if self.opt_Z > 0 else self.max_Z
            y = [np.log10(model.Z[m]/opt_Z) for m in range(1,model.m)]
            plt.plot(x, y,
                    color=self.colors[k],
                    marker=self.markers[k],
                    label=self.model_names[k],
                    alpha=0.4)
        plt.ylim(-0.001, 0.001)
        plt.xlabel(x_axis)
        # plt.ylabel('log(|Z-Z_opt|/Z_opt')
        plt.ylabel('log(Z/Z_opt)')
        plt.legend(loc='upper right')
        plt.title('Objective Values')
        if file_dir_:
            plt.savefig(file_dir_+'_Z.png')
        else:
            plt.show()

        # relative difference of c
        plt.figure(figsize=(10,8))
        for k, model in enumerate(self.models):
            if len(model.rel_dif_c) > 0:
                if x_axis == 'Iteration':
                    x = np.arange(1,model.m)
                elif x_axis == 'CPU time':
                    x = [model.cpu_time[m] for m in range(1,model.m)]
                y = [np.log10(model.rel_dif_c[m]) for m in range(1,model.m)]
                plt.plot(x, y,
                        color=self.colors[k],
                        marker=self.markers[k],
                        label=self.model_names[k],
                        alpha=0.4)
        plt.xlabel(x_axis)
        plt.ylabel('log(|c-c_ref|/c_ref)')
        plt.legend(loc='upper right')
        plt.title('Relative Cost Errors')
        if file_dir_:
            plt.savefig(file_dir_+'_error_c.png')
        else:
            plt.show()

        # relative difference of x
        plt.figure(figsize=(10,8))
        for k, model in enumerate(self.models):
            if len(model.rel_dif_x) > 0:
                if x_axis == 'Iteration':
                    x = np.arange(1,model.m)
                elif x_axis == 'CPU time':
                    x = [model.cpu_time[m] for m in range(1,model.m)]
                y = [np.log10(model.rel_dif_x[m]) for m in range(1,model.m)]
                plt.plot(x, y,
                        color=self.colors[k],
                        marker=self.markers[k],
                        label=self.model_names[k],
                        alpha=0.4)
        plt.xlabel(x_axis)
        plt.ylabel('log(|x-x_ref|/x_ref)')
        plt.legend(loc='upper right')
        plt.title('Relative Flow Errors')
        if file_dir_:
            plt.savefig(file_dir_+'_error_x.png')
        else:
            plt.show()

        # relative mean gradients
        plt.figure(figsize=(10,8))
        for k, model in enumerate(self.models):
            if len(model.rel_grads) > 0:
                if x_axis == 'Iteration':
                    x = np.arange(1,model.m)
                elif x_axis == 'CPU time':
                    x = [model.cpu_time[m] for m in range(1,model.m)]
                y = [np.log10(model.rel_grads[m]) for m in range(1,model.m)]
                plt.plot(x, y,
                        color=self.colors[k],
                        marker=self.markers[k],
                        label=self.model_names[k],
                        alpha=0.4)
        plt.xlabel(x_axis)
        plt.ylabel('log(mean(G/X))')
        plt.legend(loc='upper right')
        plt.title('Relative Mean Gradients')
        if file_dir_:
            plt.savefig(file_dir_+'_grad.png')
        else:
            plt.show()

        # step size process
        plt.figure(figsize=(10,8))
        for k, model in enumerate(self.models):
            if hasattr(model, 'optimizer'):
                optimizer = model.optimizer
                if hasattr(optimizer, 'with_BT'):
                    if optimizer.with_BT:
                        x = np.arange(1,model.m)
                        y = [np.log10(optimizer.s[m]) for m in range(1,model.m)]
                        plt.plot(x, y,
                                color=self.colors[k],
                                marker=self.markers[k],
                                label=self.model_names[k],
                                alpha=0.4)
        plt.xlabel('Iteration')
        plt.ylabel('log(step size)')
        plt.legend(loc='upper right')
        plt.title('Step Size in Backtracking')
        if file_dir_:
            plt.savefig(file_dir_+'_step.png')
        else:
            plt.show()
