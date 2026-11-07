"""One disclosed development fixture, not an image generation benchmark."""
import math
import random
from .core import Schedule, Network, train, sample, features


def centers():
    return [[2*math.cos(2*math.pi*k/8), 2*math.sin(2*math.pi*k/8)] for k in range(8)]


def fixture(count, seed):
    r, c = random.Random(seed), centers()
    return [[v+r.gauss(0, .12) for v in r.choice(c)] for _ in range(count)]


def metrics(points, reference):
    c = centers()
    distance = [[sum((a-b)**2 for a,b in zip(x,y)) for y in c] for x in points]
    bins = [0]*len(c)
    for row in distance:
        bins[min(range(len(c)), key=lambda j: row[j])] += 1
    # Symmetric empirical nearest-neighbor squared distance, not a density likelihood.
    def directed(a,b):
        return sum(min(sum((x[k]-y[k])**2 for k in range(2)) for y in b) for x in a)/len(a)
    return {'mean_nearest_center_distance': sum(math.sqrt(min(row)) for row in distance)/len(points),
            'within_0_4_of_center': sum(min(row) <= .16 for row in distance)/len(points),
            'nearest_center_counts': bins,
            'modes_with_at_least_5_nearby_samples': sum(sum(row[j] <= .16 for row in distance) >= 5 for j in range(8)),
            'symmetric_nn_squared_distance': (directed(points, reference)+directed(reference, points))/2}


def noise_mse(net, schedule, points, seed):
    r, total = random.Random(seed), 0.
    for x in points:
        t, eps = r.randint(1, schedule.steps), [r.gauss(0,1),r.gauss(0,1)]
        f = features(schedule.forward(x,t,eps),t,schedule)
        total += net.loss_grad(f,eps)[0]
    return total/len(points)


def demo(include_samplers=False):
    schedule = Schedule()
    training, reference = fixture(1024, 31), fixture(256, 37)
    net, history = train(training, schedule)
    untrained = Network()
    trained_samples = sample(net, schedule)
    raw_samples = sample(untrained, schedule)
    r = random.Random(17)
    prior = [[r.gauss(0,1),r.gauss(0,1)] for _ in range(256)]
    result = {'fixture': {'train_seed':31, 'reference_seed':37, 'train_count':1024,
                         'reference_count':256, 'train_seed_network':7, 'sample_seed':17,
                         'updates':4000,'batch_size':32,'hidden':48,'learning_rate':.003,
                         'steps':64,'beta_start':.001,'beta_end':.15,
                         'terminal_alpha_bar':schedule.abar[-1]},
              'noise_mse': {'untrained':noise_mse(untrained,schedule,reference,43),
                            'trained':noise_mse(net,schedule,reference,43)},
              'history':history,
              'metrics':{name:metrics(p,reference) for name,p in
                         [('trained',trained_samples),('untrained',raw_samples),('gaussian_prior',prior)]},
              'points':{'reference':reference,'trained':trained_samples,
                        'untrained':raw_samples,'gaussian_prior':prior}}
    assert result['noise_mse']['trained'] < result['noise_mse']['untrained']
    assert result['metrics']['trained']['within_0_4_of_center'] > result['metrics']['gaussian_prior']['within_0_4_of_center']
    if include_samplers:
        from .sampling import sample_ddim
        arms = {}
        for steps in [8,16,64]:
            audit = sample_ddim(net,schedule,steps=steps)
            arms[f'ddim_{steps}'] = {**audit,'metrics':metrics(audit['points'],reference)}
        result['samplers'] = arms
    return result
