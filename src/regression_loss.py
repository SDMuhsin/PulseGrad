import numpy as np
import matplotlib.pyplot as plt
import math as mt

nb_iters = 300
lrn_rate = 0.1
beta1 = 0.95
beta2 = 0.999
eps = 0.00000001

class Adam(object):
    def __init__(self, lrn_rate, beta1, beta2, eps):
        self.lrn_rate = lrn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.idx = 0
        self.m = 0.0 # 1st order
        self.v = 0.0 # 2nd order

    def update(self, x, g):
        self.idx += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g ** 2
        m_adj = self.m / (1.0 - np.power(self.beta1, self.idx))
        v_adj = self.v / (1.0 - np.power(self.beta2, self.idx))
        x_new = x - self.lrn_rate * m_adj / np.sqrt(v_adj + self.eps)
        return x_new

class diffGrad(object):
    def __init__(self, lrn_rate, beta1, beta2, eps):
        self.lrn_rate = lrn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.idx = 0
        self.m = 0.0 # 1st order
        self.v = 0.0 # 2nd order
        self.g_prev = 0.0

    def update(self, x, g):
        self.idx += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g ** 2
        m_adj = self.m / (1.0 - np.power(self.beta1, self.idx))
        v_adj = self.v / (1.0 - np.power(self.beta2, self.idx))
        dfc = 1.0 / (1.0 + np.exp(-np.abs(self.g_prev - g)))
        x_new = x - self.lrn_rate * m_adj * dfc / (np.sqrt(v_adj) + self.eps)
        self.g_prev = g
        return x_new

class ReweightedAdam(object):
    """
    ReweightedAdam Optimizer:
    Combines the raw gradient and its exponential moving average
    in the final step, avoiding friction-based scaling or direct
    dependence on g_{t-1} - g_t.
    """
    def __init__(self, lrn_rate, beta1, beta2, eps):
        self.lrn_rate = lrn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.idx = 0
        self.m = 0.0  # First moment
        self.v = 0.0  # Second moment
        self.g_prev = 0.0  # Not strictly needed, but kept for API consistency

    def update(self, x, g):
        """
        x: current parameter
        g: current gradient
        returns: updated parameter
        """
        self.idx += 1

        # Update moments (same as Adam)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g ** 2)

        # Bias correction
        m_hat = self.m / (1.0 - self.beta1 ** self.idx)
        v_hat = self.v / (1.0 - self.beta2 ** self.idx)

        # Reweighting coefficient
        # (Dimension-wise, for brevity, we show scalar code.)
        # If using vector parameters, do element-wise operations below.
        denom = np.abs(m_hat) + np.abs(g) + self.eps
        lam = np.abs(g) / denom  # lam in [0, 1]

        # Final effective gradient
        u = (1.0 - lam) * m_hat + lam * g

        # Parameter update
        x_new = x - self.lrn_rate * u / (np.sqrt(v_hat) + self.eps)

        self.g_prev = g  # retained for compatibility; not used here
        return x_new

class AdaDeltaGradV1(object):
    def __init__(self, lrn_rate, beta1, beta2, eps, lam=1.0):
        """
        :param lrn_rate: Learning rate (alpha)
        :param beta1: Exponential decay rate for first moment
        :param beta2: Exponential decay rate for second moment
        :param eps: Small constant to avoid division by zero
        :param lam: Logistic rate parameter controlling how quickly scale saturates
        """
        self.lrn_rate = lrn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lam = lam
        
        self.idx = 0
        self.m = 0.0  # 1st order moment
        self.v = 0.0  # 2nd order moment

    def update(self, x, g):
        """
        Performs one update step of the Experimental optimizer.

        :param x: Current parameter value
        :param g: Current gradient
        :return: Updated parameter value
        """
        # Increment iteration
        self.idx += 1

        # Standard Adam updates for moments
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g ** 2)

        # Bias-corrected estimates
        m_hat = self.m / (1.0 - np.power(self.beta1, self.idx))
        v_hat = self.v / (1.0 - np.power(self.beta2, self.idx))

        # Proposed scaling factor based on delta = g - m
        delta = g - self.m
        # A logistic-based scale factor in (0, 1], saturates if |delta| is large
        scale = 1.0 / (1.0 + np.exp(self.lam * np.abs(delta)))

        # Final update
        x_new = x - self.lrn_rate * scale * (m_hat / (np.sqrt(v_hat) + self.eps))

        return x_new

class Experimental(object):
    def __init__(self, lrn_rate, beta1, beta2, eps, lam=1.0):
        """
        :param lrn_rate: Learning rate (alpha)
        :param beta1: Exponential decay rate for first moment
        :param beta2: Exponential decay rate for second moment
        :param eps: Small constant to avoid division by zero
        :param lam: Logistic rate parameter controlling scaling saturation
        """
        self.lrn_rate = lrn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lam = lam

        self.idx = 0
        self.m = 0.0  # First moment
        self.v = 0.0  # Second moment

    def update(self, x, g):
        """
        Performs one update step using Nesterov-accelerated momentum scaled by a logistic factor.
        
        :param x: Current parameter value
        :param g: Current gradient
        :return: Updated parameter value
        """
        # Increment iteration counter
        self.idx += 1

        # Update biased first and second moments (Adam style)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g ** 2)

        # Bias correction
        m_hat = self.m / (1.0 - np.power(self.beta1, self.idx))
        v_hat = self.v / (1.0 - np.power(self.beta2, self.idx))

        # Logistic scaling factor based on the difference between current gradient and momentum
        # (Preserves stability by damping when there's significant deviation)
        delta = g - self.m
        scale = 1.0 / (1.0 + np.exp(self.lam * np.abs(delta)))

        # Incorporate Nesterov acceleration: Lookahead gradient term
        # This term accelerates convergence in stable regions (when scale ~ 1)
        m_nesterov = self.beta1 * m_hat + ((1.0 - self.beta1) * g) / (1.0 - np.power(self.beta1, self.idx))

        # Final parameter update with Nesterov acceleration and logistic scaling
        x_new = x - self.lrn_rate * scale * (m_nesterov / (np.sqrt(v_hat) + self.eps))
        
        return x_new

def fun1(x):
    if x <= 0:
        return (x + 0.3)*(x + 0.3)
    else:
        return (x - 0.2)*(x - 0.2) + 0.05

def calc_grad(x):
    if x <= 0:
        return 2*x + 0.6
    else:
        return 2*x - 0.4

# optimize with the specified solver
def solve(x0, solver):
    x = np.zeros(nb_iters)
    y = np.zeros(nb_iters)
    x[0] = x0
    for idx_iter in range(1, nb_iters):
        g = calc_grad(x[idx_iter - 1])
        x[idx_iter] = solver.update(x[idx_iter - 1], g)
        y[idx_iter] = fun1(x[idx_iter])
    return x, y

# Adam & diffGrad
x = {}
y = {}
x0 = -1.0
solver = Adam(lrn_rate, beta1, beta2, eps)
x['adam'], y['adam'] = solve(x0, solver)
solver = diffGrad(lrn_rate, beta1, beta2, eps)
x['diffGrad'], y['diffGrad'] = solve(x0, solver)
solver = Experimental(lrn_rate, beta1, beta2, eps)
x['Experimental'], y['Experimental'] = solve(x0, solver)

# visualization
plt.rcParams['figure.dpi']= 300
plt.rcParams['figure.figsize'] = [6.0, 4.0]
#plt.plot(np.arange(nb_iters) + 1, x['adam'], label='Adam')
#plt.plot(np.arange(nb_iters) + 1, x['diffGrad'], label='diffGrad')
plt.plot(np.arange(nb_iters) + 1, y['adam'], label='Adam')
plt.plot(np.arange(nb_iters) + 1, y['diffGrad'], label='diffGrad')
plt.plot(np.arange(nb_iters) + 1, y['Experimental'], label='Experimental')
plt.legend()
plt.grid()
plt.savefig("./results/regression_f1.png")
plt.close()


def calc_grad(x):
    if x <= -0.9:
        return -40
    else:
        return x * mt.cos(8 * x) + mt.sin(8 * x) + 3 * x * x

# optimize with the specified solver
def solve(x0, solver):
    x = np.zeros(nb_iters)
    x[0] = x0
    for idx_iter in range(1, nb_iters):
        g = calc_grad(x[idx_iter - 1])
        x[idx_iter] = solver.update(x[idx_iter - 1], g)
    return x

# Adam & diffGrad
x = {}
x0 = -1.0
solver = Adam(lrn_rate, beta1, beta2, eps)
x['adam'] = solve(x0, solver)
solver = diffGrad(lrn_rate, beta1, beta2, eps)
x['diffGrad'] = solve(x0, solver)
solver = Experimental(lrn_rate, beta1, beta2, eps)
x['Experimental'] = solve(x0, solver)

# visualization
plt.rcParams['figure.dpi']= 300
plt.rcParams['figure.figsize'] = [6.0, 4.0]
plt.plot(np.arange(nb_iters) + 1, x['adam'], label='Adam')
plt.plot(np.arange(nb_iters) + 1, x['diffGrad'], label='diffGrad')
plt.plot(np.arange(nb_iters) + 1, x['Experimental'], label='Experimental')
plt.legend()
plt.grid()
plt.savefig("./results/regression_f2.png")
plt.close()


def calc_grad(x):
    if x <= -0.5:
        return 2 * x
    elif x <= -0.4:
        return 1.0
    elif x <= 0.0:
        return -7/8
    elif x <= 0.4:
        return 7/8
    elif x <= 0.5:
        return -1.0
    else:
        return 2 * x

# optimize with the specified solver
def solve(x0, solver):
    x = np.zeros(nb_iters)
    x[0] = x0
    for idx_iter in range(1, nb_iters):
        g = calc_grad(x[idx_iter - 1])
        x[idx_iter] = solver.update(x[idx_iter - 1], g)
    return x

# Adam & diffGrad
x = {}
x0 = -1.0
solver = Adam(lrn_rate, beta1, beta2, eps)
x['adam'] = solve(x0, solver)
solver = diffGrad(lrn_rate, beta1, beta2, eps)
x['diffGrad'] = solve(x0, solver)
solver = Experimental(lrn_rate, beta1, beta2, eps)
x['Experimental'] = solve(x0, solver)
# visualization
plt.rcParams['figure.dpi']= 300
plt.rcParams['figure.figsize'] = [6.0, 4.0]
plt.plot(np.arange(nb_iters) + 1, x['adam'], label='Adam')
plt.plot(np.arange(nb_iters) + 1, x['diffGrad'], label='diffGrad')
plt.plot(np.arange(nb_iters) + 1, x['Experimental'], label='Experimental')
plt.legend()
plt.grid()
plt.savefig("./results/regression_f3.png")
