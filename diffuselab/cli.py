import argparse
import json
import sys
from .core import train, sample, artifact, load, Schedule
from .demo import demo


def read(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description='Offline 2-D DDPM laboratory')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('demo')
    sub.add_parser('sampler-demo')
    t = sub.add_parser('train')
    t.add_argument('input', help='JSON array of [x,y] training points')
    t.add_argument('output')
    t.add_argument('--updates', type=int, default=4000)
    t.add_argument('--seed', type=int, default=7)
    s = sub.add_parser('sample')
    s.add_argument('model')
    s.add_argument('--count', type=int, default=256)
    s.add_argument('--seed', type=int, default=17)
    dd = sub.add_parser('ddim')
    dd.add_argument('model')
    dd.add_argument('--count',type=int,default=256)
    dd.add_argument('--seed',type=int,default=17)
    dd.add_argument('--steps',type=int,default=16)
    dd.add_argument('--eta',type=float,default=0.)
    a = p.parse_args()
    try:
        if a.command in ('demo','sampler-demo'):
            result = demo(a.command == 'sampler-demo')
        elif a.command == 'train':
            schedule = Schedule()
            net, history = train(read(a.input), schedule, updates=a.updates, seed=a.seed)
            with open(a.output, 'w', encoding='utf-8') as f:
                json.dump(artifact(net,schedule),f,allow_nan=False)
            result = {'output':a.output,'history':history}
        elif a.command == 'ddim':
            from .sampling import sample_ddim
            net,schedule = load(read(a.model))
            result = sample_ddim(net,schedule,a.count,a.seed,a.steps,a.eta)
        else:
            net,schedule = load(read(a.model))
            result = {'points':sample(net,schedule,a.count,a.seed)}
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError, OSError, TypeError) as e:
        p.exit(2, f'diffuselab: {e}\n')


if __name__ == '__main__':
    main()
