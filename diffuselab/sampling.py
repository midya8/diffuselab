"""Generalized DDIM updates; alpha_bar here is alpha in the DDIM paper."""
import math
import random
from .core import integer, number, vector


def timestep_grid(schedule, steps):
    integer(steps, 1, schedule.steps, 'sampling steps')
    if steps == 1:
        return [schedule.steps]
    return list(reversed([1+i*(schedule.steps-1)//(steps-1) for i in range(steps)]))


def eta_value(eta):
    eta = number(eta, 'eta')
    if not 0 <= eta <= 1:
        raise ValueError('eta must be in [0,1]')
    return eta


def ddim_step(schedule, noisy, t, previous, epsilon, noise, eta=0.):
    schedule.check_t(t)
    integer(previous, 0, t-1, 'previous timestep')
    eta = eta_value(eta)
    noisy, epsilon, noise = (vector(x, 2, name) for x, name in
                             [(noisy,'noisy'), (epsilon,'epsilon'), (noise,'noise')])
    a, ap = schedule.abar[t], schedule.abar[previous]
    sigma2 = eta*eta*(1-ap)/(1-a)*(1-a/ap)
    direction2 = 1-ap-sigma2
    # Only roundoff-sized negative radicands may be truncated.
    if sigma2 < -1e-12 or direction2 < -1e-12:
        raise ValueError('invalid DDIM variance')
    sigma, direction = math.sqrt(max(0.,sigma2)), math.sqrt(max(0.,direction2))
    clean = [(x-math.sqrt(1-a)*e)/math.sqrt(a) for x,e in zip(noisy,epsilon)]
    result = [math.sqrt(ap)*x0+direction*e+sigma*z for x0,e,z in zip(clean,epsilon,noise)]
    if not all(math.isfinite(x) for x in result):
        raise ValueError('nonfinite DDIM state')
    return result


def sample_ddim(net, schedule, count=256, seed=17, steps=16, eta=0.):
    integer(count,1,2048,'count')
    integer(seed,0,2**32-1,'seed')
    grid, eta = timestep_grid(schedule,steps), eta_value(eta)
    r = random.Random(seed)
    points = [[r.gauss(0,1),r.gauss(0,1)] for _ in range(count)]
    calls = 0
    for j,t in enumerate(grid):
        previous = grid[j+1] if j+1 < len(grid) else 0
        updated = []
        for x in points:
            eps = net.predict(x,t,schedule)
            calls += 1
            z = [r.gauss(0,1),r.gauss(0,1)] if eta > 0 and previous > 0 else [0.,0.]
            updated.append(ddim_step(schedule,x,t,previous,eps,z,eta))
        points = updated
    return {'points':points, 'timesteps':grid, 'eta':eta,
            'denoiser_calls':calls, 'denoiser_calls_per_sample':len(grid),
            'call_reduction_vs_full_ddpm':1-len(grid)/schedule.steps}
