import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from utils.network_generator import NetworkGenerator
from IPython import display

class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None,
         file_path=None):
    """Plot data points."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or
                isinstance(X, list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

    if file_path is not None:
        plt.savefig(file_path)

class Results(object):

    def __init__(
        self,
        model,
        size,
        demand,
        threshold=1e-3,
        dir_='results/grid_network/_graph/',
        mdir_={'PL':'NGEVSUE_PL_m5_MCA_', 'AGP':'NGEVSUE_AGD_BTTrue_m5_s0.0001_kmin50_eta0.25_MCA_'}
        ):

        self.model = model
        self.size = size
        self.demand = demand
        self.threshold = threshold
        self.dir_ = dir_
        self.mdir_ = mdir_

        self.get_converge_times()

    def get_converge_times(self):

        Record = namedtuple('Record', 'model size demand iteration cpu_time Zdif num_nodes num_links num_ods num_dests LD')
        records = []
        self.x_min = 1e+10
        self.y_min = 1e+10
        self.x_max = 0
        self.y_max = 0

        for d in self.demand:
            for s in self.size:
                generator = NetworkGenerator(s, d, d, 'grid', bidirection=True)
                network_data = generator.generate_network()
                _, N, L, W, D = generator._get_size()
                if L*D < self.x_min:
                    self.x_min = L*D
                if L*D > self.x_max:
                    self.x_max = L*D
                for model_name in self.model:
                    model_dir = self.mdir_[model_name]
                    file_name = self.dir_ + model_dir + str(s) + '_' + str(int(d)) + 'q_metric.csv'
                    data = pd.read_csv(file_name, sep='\t')
                    data.columns = data.columns.str.strip()
                    done = False
                    optZ = data['objective_value'].tail(1).values[0]
                    for i in range(len(data)):
                        if 'relative_obj_dif' in data:
                            Zdif = data['relative_obj_dif'][i]
                        else:
                            Zdif = (optZ - data['objective_value'][i])/optZ
                        done = Zdif < self.threshold and i > 3

                        if done:
                            cpu_time = data['CPU_time'][i] - data['CPU_time'][0]
                            records.append(Record(model_name, s, d, i+1, cpu_time, Zdif, N, L, W, D, L*D))
                            if cpu_time > self.y_max:
                                self.y_max = cpu_time
                            break

                    if not done:
                        cpu_time = data['CPU_time'].tail(1).values[0] - data['CPU_time'][0]
                        records.append(Record(model_name, s, d, len(data), cpu_time, Zdif, N, L, W, D, L*D))
                        if cpu_time < self.y_min:
                            self.y_min = cpu_time
                        if cpu_time > self.y_max:
                            self.y_max = cpu_time

        self.record_df = pd.DataFrame(records, columns=records[0]._fields)

    def visualize(self, model, demand, key='plot', file_name=None):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim([0,self.x_max])
        ax.set_ylim([0,self.y_max])
        for d in demand:
            for m in model:
                df = self.record_df.copy()
                df = df[df['model']==m]
                df = df[df['demand']==d]
                if key == 'plot':
                    df.plot(x='LD',y='cpu_time',label=m+str(d),ax=ax)
                elif key == 'scatter':
                    df.plot.scatter(x='LD',y='cpu_time',label=m+str(d),ax=ax)
                elif key == 'both':
                    df.plot.scatter(x='LD',y='cpu_time',ax=ax)
                    df.plot(x='LD',y='cpu_time',label=m+str(d),ax=ax)
        ax.set_xlabel('L*D')
        ax.set_ylabel('CPU time [s]')
        if file_name is not None:
            fig.savefig(self.dir_ + 'output/' + file_name + '.eps')
        else:
            plt.show()

    def logplot(self, model, demand, key='plot',log=[True,True],file_name=None):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        # x_max = np.max([self.x_max,1e+6])
        # y_max = np.max([self.y_max,1e+5])
        #ax.set_xlim([1e+2,5e+6])
        ax.set_ylim([1e-1,1e+5])
        marks = ['o', 'D']
        colors = ['blue', 'red']
        for d, mrk in zip(demand,marks):
            for m, col in zip(model,colors):
                df = self.record_df.copy()
                df = df[df['model']==m]
                df = df[df['demand']==d]
                # df['LD'] = np.log10(df['LD'])
                # df['cpu_time'] = np.log10(df['cpu_time'])
                if key == 'plot':
                    df.plot(x='LD',y='cpu_time',label=m+str(d),ax=ax,logx=log[0],logy=log[1],color=col,linestyle='--')
                elif key == 'scatter':
                    df.plot.scatter(x='LD',y='cpu_time',label=m+str(d),ax=ax,logx=log[0],logy=log[1],marker=mrk,color=col)
                elif key == 'both':
                    df.plot.scatter(x='LD',y='cpu_time',ax=ax,logx=log[0],logy=log[1],marker=mrk,color=col)
                    df.plot(x='LD',y='cpu_time',label=m+str(d),ax=ax,logx=log[0],logy=log[1],color=col,linestyle='--')
        ax.set_xlabel('L*D')
        ax.set_ylabel('CPU time [s]')
        if file_name is not None:
            fig.savefig(self.dir_ + 'output/' + file_name + '.eps')
        else:
            plt.show()


if __name__=='__main__':
    # res2 = Results(model=['AGP'],size=[4,8,12,16,20,24,28,32,36,40],demand=[10000,15000],threshold=1e-3)
    # res2.visualize(model=['AGP'], demand=[10000,15000], key='plot')

    res = Results(model=['PL','AGP'],size=[4,8,12,16,20,24,28,32,36,40],demand=[10000,15000],threshold=1e-4)
    #res.visualize(model=['PL','AGP'], demand=[10000,15000], key='plot')
    res.visualize(model=['PL','AGP'], demand=[10000,15000], key='both', file_name='PL_AGP_to40')
    #res.logplot(model=['PL','AGP'], demand=[10000,15000], key='both',log=[True,False])
    res.logplot(model=['PL','AGP'], demand=[10000,15000], key='both',log=[True,True], file_name='PL_AGP_to40_loglog')
    print(res.record_df)
