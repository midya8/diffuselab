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
    t = sub.add_parser('train')
    t.add_argument('input', help='JSON array of [x,y] training points')
    t.add_argument('output')
    t.add_argument('--updates', type=int, default=4000)
    t.add_argument('--seed', type=int, default=7)
    s = sub.add_parser('sample')
    s.add_argument('model')
    s.add_argument('--count', type=int, default=256)
    s.add_argument('--seed', type=int, default=17)
    a = p.parse_args()
    try:
        if a.command == 'demo':
            result = demo()
        elif a.command == 'train':
            schedule = Schedule()
            net, history = train(read(a.input), schedule, updates=a.updates, seed=a.seed)
            with open(a.output, 'w', encoding='utf-8') as f:
                json.dump(artifact(net,schedule),f,allow_nan=False)
            result = {'output':a.output,'history':history}
        else:
            net,schedule = load(read(a.model))
            result = {'points':sample(net,schedule,a.count,a.seed)}
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError, OSError, TypeError) as e:
        p.exit(2, f'diffuselab: {e}\n')


if __name__ == '__main__':
    main()
