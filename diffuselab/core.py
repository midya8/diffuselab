"""Small, explicit DDPM primitives. Timesteps are 1-based; index 0 is clean."""
import math
import random


def number(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(name + ' must be finite')
    return float(x)


def integer(x, low, high, name):
    if type(x) is not int or not low <= x <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return x


def vector(x, n, name):
    if not isinstance(x, list) or len(x) != n:
        raise ValueError(f'{name} needs {n} coordinates')
    return [number(v, name) for v in x]


class Schedule:
    def __init__(self, steps=64, beta_start=.001, beta_end=.15):
        integer(steps, 2, 256, 'steps')
        a, b = number(beta_start, 'beta_start'), number(beta_end, 'beta_end')
        if not 1e-6 <= a <= b <= .25:
            raise ValueError('require 1e-6 <= beta_start <= beta_end <= .25')
        self.steps, self.beta_start, self.beta_end = steps, a, b
        self.beta = [0.] + [a + (b-a)*i/(steps-1) for i in range(steps)]
        self.abar = [1.]
        for beta in self.beta[1:]:
            self.abar.append(self.abar[-1]*(1-beta))
        self.variance = [0.] + [self.beta[t]*(1-self.abar[t-1])/(1-self.abar[t])
                                  for t in range(1, steps+1)]

    def check_t(self, t):
        integer(t, 1, self.steps, 't')

    def forward(self, clean, t, noise):
        self.check_t(t)
        clean, noise = vector(clean, 2, 'clean'), vector(noise, 2, 'noise')
        a, b = math.sqrt(self.abar[t]), math.sqrt(1-self.abar[t])
        return [a*x+b*z for x, z in zip(clean, noise)]

    def reverse(self, noisy, t, epsilon, noise):
        self.check_t(t)
        noisy, epsilon, noise = (vector(x, 2, name) for x, name in
                                 [(noisy, 'noisy'), (epsilon, 'epsilon'), (noise, 'noise')])
        a = math.sqrt(1-self.beta[t])
        b = self.beta[t]/math.sqrt(1-self.abar[t])
        sd = math.sqrt(self.variance[t]) if t > 1 else 0.
        result = [(x-b*e)/a+sd*z for x, e, z in zip(noisy, epsilon, noise)]
        if not all(math.isfinite(x) for x in result):
            raise ValueError('nonfinite reverse state')
        return result


def features(x, t, schedule):
    schedule.check_t(t)
    u = t/schedule.steps
    return vector(x, 2, 'x') + [u, math.sin(math.pi*u), math.cos(math.pi*u),
                               math.sin(2*math.pi*u), math.cos(2*math.pi*u)]


class Network:
    """7 -> tanh(hidden) -> 2; row-major flat weights and exact MSE gradients."""
    def __init__(self, hidden=48, seed=7):
        integer(hidden, 1, 128, 'hidden')
        integer(seed, 0, 2**32-1, 'seed')
        self.hidden = hidden
        r = random.Random(seed)
        self.params = [r.gauss(0, 1/math.sqrt(7)) for _ in range(hidden*7)]
        self.params += [0.]*hidden
        self.params += [r.gauss(0, .1/math.sqrt(hidden)) for _ in range(2*hidden)]
        self.params += [0., 0.]

    def run(self, f):
        f = vector(f, 7, 'features')
        h, p = self.hidden, self.params
        a = [math.tanh(sum(p[j*7+k]*f[k] for k in range(7))+p[7*h+j]) for j in range(h)]
        out = [sum(p[8*h+d*h+j]*a[j] for j in range(h))+p[10*h+d] for d in range(2)]
        return out, a

    def loss_grad(self, f, target):
        f, target = vector(f, 7, 'features'), vector(target, 2, 'target')
        out, a = self.run(f)
        h, p = self.hidden, self.params
        g = [0.]*len(p)
        delta = [out[d]-target[d] for d in range(2)]
        for d in range(2):
            for j in range(h):
                g[8*h+d*h+j] = delta[d]*a[j]
            g[10*h+d] = delta[d]
        for j in range(h):
            dh = sum(delta[d]*p[8*h+d*h+j] for d in range(2))*(1-a[j]**2)
            for k in range(7):
                g[j*7+k] = dh*f[k]
            g[7*h+j] = dh
        return sum(d*d for d in delta)/2, g

    def predict(self, x, t, schedule):
        return self.run(features(x, t, schedule))[0]


def train(points, schedule=None, hidden=48, updates=4000, batch_size=32,
          learning_rate=.003, seed=7):
    if not isinstance(points, list) or not 2 <= len(points) <= 8192:
        raise ValueError('need 2..8192 training points')
    points = [vector(p, 2, 'point') for p in points]
    if any(abs(v) > 100 for p in points for v in p):
        raise ValueError('training coordinates must be within [-100,100]')
    integer(updates, 1, 20000, 'updates')
    integer(batch_size, 1, 128, 'batch_size')
    lr = number(learning_rate, 'learning_rate')
    if not 1e-6 <= lr <= .1:
        raise ValueError('learning_rate outside [1e-6,.1]')
    schedule = schedule or Schedule()
    net, rng = Network(hidden, seed), random.Random(seed+1)
    m, v = [0.]*len(net.params), [0.]*len(net.params)
    history = []
    for step in range(1, updates+1):
        grad, loss = [0.]*len(net.params), 0.
        for _ in range(batch_size):
            x, t = rng.choice(points), rng.randint(1, schedule.steps)
            eps = [rng.gauss(0, 1), rng.gauss(0, 1)]
            xt = schedule.forward(x, t, eps)
            value, g = net.loss_grad(features(xt, t, schedule), eps)
            loss += value/batch_size
            for j, z in enumerate(g):
                grad[j] += z/batch_size
        # Adam, with no weight decay, gradient clipping, EMA or checkpoint search.
        for j, g in enumerate(grad):
            m[j] = .9*m[j]+.1*g
            v[j] = .999*v[j]+.001*g*g
            net.params[j] -= lr*(m[j]/(1-.9**step))/(math.sqrt(v[j]/(1-.999**step))+1e-8)
        if not all(math.isfinite(x) for x in net.params) or not math.isfinite(loss):
            raise ValueError('training diverged')
        if step == 1 or step % 500 == 0 or step == updates:
            history.append({'update': step, 'batch_noise_mse': loss})
    return net, history


def sample(net, schedule, count=256, seed=17):
    integer(count, 1, 2048, 'count')
    integer(seed, 0, 2**32-1, 'seed')
    r = random.Random(seed)
    points = [[r.gauss(0, 1), r.gauss(0, 1)] for _ in range(count)]
    for t in range(schedule.steps, 0, -1):
        points = [schedule.reverse(x, t, net.predict(x, t, schedule),
                                  [r.gauss(0, 1), r.gauss(0, 1)] if t > 1 else [0., 0.])
                  for x in points]
    return points


def artifact(net, schedule):
    return {'format': 'diffuselab_v1', 'hidden': net.hidden, 'parameters': list(net.params),
            'schedule': {'steps': schedule.steps, 'beta_start': schedule.beta_start,
                         'beta_end': schedule.beta_end}}


def load(value):
    if not isinstance(value, dict) or set(value) != {'format','hidden','parameters','schedule'}:
        raise ValueError('invalid model fields')
    if value['format'] != 'diffuselab_v1':
        raise ValueError('unsupported model format')
    s = value['schedule']
    if not isinstance(s, dict) or set(s) != {'steps','beta_start','beta_end'}:
        raise ValueError('invalid schedule fields')
    schedule = Schedule(**s)
    net = Network(value['hidden'])
    net.params = vector(value['parameters'], 10*net.hidden+2, 'parameters')
    return net, schedule
