import numpy as np

class LineSearch(object):

    def __init__(self, method='PL', threshold=1e-10):
        self.threshold = threshold
        if method == 'PL':
            self._do = self.golden_section
        elif method == 'MSA':
            self._do = self.msa

    def define(self, f):
        self.f = f

    def msa(self, m, x, d):
        if m == 0:
            return 1/(m+2)
        else:
            return 1/(m+2)

    def golden_section(self, m, x, d):
        lb = 0.
        ub = 1.
        ratio = (-1. + np.sqrt(5.))/2.
        p = ub - ratio * (ub - lb)
        q = lb + ratio * (ub - lb)
        xp = x + p * d
        xq = x + q * d
        Fp = self.f(xp)
        Fq = self.f(xq)

        width = 1e+8
        counter = 0
        while True:
            if Fp <= Fq:
                ub = q
                q = p
                Fq = Fp
                p = ub - ratio * (ub - lb)
                xp = x + p * d
                Fp = self.f(xp)
            else:
                lb = p
                p = q
                Fp = Fq
                q = lb + ratio * (ub - lb)
                xq = x + q * d
                Fq = self.f(xq)
                width = abs(ub-lb)/2.

            counter += 1
            if counter > 100 or width < self.threshold:
                break

        alpha = (lb+ub)/2
        return alpha

class GradientDescent(object):

    def __init__(self,
                init_lr=1e-3):

        # params
        self.eps = 1e-8
        self.inf = 1e+10
        self.s = init_lr

    def initialize(self,
                    f, # objective function to MINIMIZE
                    g, # gradient function
                    x0, # initial value
                    clip, # lower and upper bounds
                    ):

        # functions
        self.f = f
        self.g = g
        self.clip = clip

        # variables
        self.m = 0
        self.j = 0
        self.x = {0: x0}
        self.grad = {}
        self.t = {0: 1.}
        self.f_val = {0: f(x0)}

    def model_update(self, m):

        j = self.j

        self.grad[m], v1, v2, _ = self.g(self.x[m])
        self.x[m+1] = self.x[m] - self.s * self.grad[m]
        self.x[m+1] = np.clip(self.x[m+1], self.clip[0], self.clip[1])
        self.f_val[m+1] = self.f(self.x[m+1])

        return self.grad[m], v1, v2, self.x[m+1], self.f_val[m+1]

class FISTA(object):

    def __init__(self,
                init_lr=1e-3,
                k_min=50,
                with_BT=False,
                eta=0.95,
                restart='f',
                min_lr=1e-100):

        # params
        self.eps = 1e-8
        self.inf = 1e+10
        self.s0 = init_lr
        self.k_min = k_min
        self.with_BT = with_BT
        self.eta = eta
        self.restart = restart
        self.min_s = min_lr

    def initialize(self,
                    f, # objective function to MINIMIZE
                    g, # gradient function
                    x0, # initial value
                    clip, # lower and upper bounds
                    ):

        # functions
        self.f = f
        self.g = g
        self.clip = clip

        # variables
        self.m = 0
        self.j = 0
        self.x = {0: x0}
        self.y = {0: x0}
        self.grad = {}
        self.s = {0: self.s0}
        self.t = {0: 1.}
        self.f_val = {0: f(x0)}

    def model_update(self, m):

        j = self.j
        self.grad[m], v1, v2, f_y = self.g(self.y[m])

        if not self.with_BT:
            self.x[m+1] = self.y[m] - self.s0 * self.grad[m]
            self.x[m+1] = np.clip(self.x[m+1], self.clip[0], self.clip[1])
            self.f_val[m+1] = self.f(self.x[m+1])
        elif self.s[m] <= self.min_s: # lower bound of step_size
            self.s[m+1] = self.min_s #self.s[m]
            self.x[m+1] = self.y[m] - self.s[m+1] * self.grad[m]
            self.x[m+1] = np.clip(self.x[m+1], self.clip[0], self.clip[1])
            self.f_val[m+1] = self.f(self.x[m+1])
        else: # backtracking
            counter = 0
            #f_y = self.f(self.y[m]) # this can be calculated together with gradient
            while True:
                s = (self.eta**counter) * self.s[m]
                x_s = self.y[m] - s * self.grad[m]
                x_s = np.clip(x_s, self.clip[0], self.clip[1])
                Q_s = f_y + np.dot(self.grad[m], (x_s-self.y[m])) +\
                        np.dot(x_s-self.y[m], x_s-self.y[m])/(2*s)
                F_s = self.f(x_s)
                if F_s <= Q_s or s <= self.min_s:
                    self.s[m+1] = s if s > self.min_s else self.min_s
                    self.x[m+1] = x_s
                    self.f_val[m+1] = F_s
                    if counter > 0:
                        print('s updated:', m, counter, self.s[m+1], F_s, Q_s)
                    break
                else:
                    # print(counter, s, F_s, Q_s)
                    counter += 1

        self.t[j+1] = (1 + np.sqrt(1 + 4 * (self.t[j] ** 2)))/2
        self.y[m+1] = self.x[m+1] + ((self.t[j] - 1)/self.t[j+1]) * (self.x[m+1] - self.x[m])
        self.y[m+1] = np.clip(self.y[m+1], self.clip[0], self.clip[1])

        # Adaptive restart
        if self.restart == 'f':
            crit = self.f_val[m+1] < self.f_val[m] # m+1 should be smaller (min)
        elif self.restart == 'g':
            crit = np.dot(self.grad[m], (self.x[m+1] - self.x[m])) < 0. # should be different sign (for min)
        elif self.restart == 's':
            crit = np.linalg.norm(self.x[m] - self.x[m-1]) -\
                    np.linalg.norm(self.x[m+1] - self.x[m]) > 0. if m > 1 else True # dif should get smaller?
        if j >= self.k_min and not crit:
            print('Adaptive restart: j={}'.format(j))
            self.j = 0
        else:
            self.j += 1

        return self.grad[m], v1, v2, self.x[m+1], self.f_val[m+1]


class Adam(object):

    def __init__(self, params=[0.01, 0.9, 0.999, 1e-8]):

        # params
        self.eps = 1e-8
        self.inf = 1e+10
        self.eta = params[0]
        self.beta1 = params[1]
        self.beta2 = params[2]
        self.epsilon = params[3]

    def initialize(self,
                    f, # objective function to MINIMIZE
                    g, # gradient function
                    x0, # initial value
                    clip, # lower and upper bounds
                    ):

        # functions
        self.f = f
        self.g = g
        self.clip = clip

        # variables
        self.t = 0
        self.x = {1: x0}
        self.grad = {}
        self.m = np.zeros(shape=x0.shape)
        self.v = np.zeros(shape=x0.shape)
        self.f_val = {1: f(x0)}

    def model_update(self, t_input):

        self.t += 1
        t = self.t

        self.grad[t], v1, v2, _ = self.g(self.x[t])

        self.m = self.beta1 * self.m + (1-self.beta1) * self.grad[t]
        self.v = self.beta2 * self.v + (1-self.beta2) * (self.grad[t]**2)
        m = self.m / (1 - (self.beta1**t))
        v = self.v / (1 - (self.beta2**t))

        self.x[t+1] = self.x[t] - self.eta * (m / (np.sqrt(v) + self.epsilon))
        self.x[t+1] = np.clip(self.x[t+1], self.clip[0], self.clip[1])
        self.f_val[t+1] = self.f(self.x[t+1])

        return self.grad[t], v1, v2, self.x[t+1], self.f_val[t+1]

class NAdam(object):

    def __init__(self, params=[0.01, 0.9, 0.999, 1e-8]):

        # params
        self.eps = 1e-8
        self.inf = 1e+10
        self.eta = params[0]
        self.beta1 = params[1]
        self.beta2 = params[2]
        self.epsilon = params[3]

    def initialize(self,
                    f, # objective function to MINIMIZE
                    g, # gradient function
                    x0, # initial value
                    clip, # lower and upper bounds
                    ):

        # functions
        self.f = f
        self.g = g
        self.clip = clip

        # variables
        self.t = 0
        self.x = {1: x0}
        self.grad = {}
        self.m = np.zeros(shape=x0.shape)
        self.v = np.zeros(shape=x0.shape)
        self.f_val = {1: f(x0)}

    def model_update(self, t_input=None):

        self.t += 1
        t = self.t

        self.grad[t], v1, v2, _ = self.g(self.x[t])

        self.m = self.beta1 * self.m + (1-self.beta1) * self.grad[t]
        self.v = self.beta2 * self.v + (1-self.beta2) * (self.grad[t]**2)
        b = np.sqrt(1 - (self.beta2**t))/(1 - (self.beta1**t))

        self.x[t+1] = self.x[t] - self.eta * b * (
            (self.beta1 * self.m + (1-self.beta1) * self.grad[t])/(np.sqrt(self.v) + self.epsilon)
        )
        self.x[t+1] = np.clip(self.x[t+1], self.clip[0], self.clip[1])
        self.f_val[t+1] = self.f(self.x[t+1])

        return self.grad[t], v1, v2, self.x[t+1], self.f_val[t+1]
