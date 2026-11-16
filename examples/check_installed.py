"""Check the wheel outside the checkout, including every test and CLI report."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
MODULES = ['diffuselab', 'diffuselab.__main__', 'diffuselab.core',
           'diffuselab.demo', 'diffuselab.sampling', 'diffuselab.cli']
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--demo', choices=['both', 'demo', 'sampler-demo'], default='both',
                    help='split the slow demo checks across separate local runs')
selected = parser.parse_args().demo
COMMANDS = [
    ['train', 'examples/points.json', 'model.json', '--updates', '2'],
    ['demo'], ['sampler-demo'],
    ['sample', 'model.json', '--count', '8', '--seed', '17'],
    ['ddim', 'model.json', '--steps', '16', '--eta', '0', '--count', '8'],
]

if selected != 'both':
    COMMANDS = [args for args in COMMANDS
                if args[0] not in ('demo', 'sampler-demo') or args[0] == selected]


def call(args, cwd, env=ENV):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    wheels = work / 'wheels'
    wheels.mkdir()
    call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels),
          str(ROOT)], work)
    built = list(wheels.glob('*.whl'))
    assert len(built) == 1, built
    venv.create(work / 'env', with_pip=True)
    py = str(work / 'env/bin/python')
    call([py, '-m', 'pip', 'install', '--no-deps', str(built[0])], work)
    outside = work / 'outside'
    outside.mkdir()
    for name in ['tests', 'examples', 'assets']:
        shutil.copytree(ROOT / name, outside / name)
    # Only the test fixtures are copied. No package source goes on this path.
    suite = '''
import importlib,importlib.util,json,pathlib,sys,unittest
outside=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(outside))
names=json.loads(sys.argv[1])
source=pathlib.Path(sys.argv[2]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
for name in names:
    if name != 'diffuselab.__main__':importlib.import_module(name)
def check():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
check()
tests=unittest.defaultTestLoader.discover(str(outside/'tests'))
result=unittest.TextTestRunner(verbosity=2).run(tests)
assert result.testsRun>0 and result.wasSuccessful()
check()
print(json.dumps(dict(tests=result.testsRun,installed_origins=len(names),
                     checked_before_and_after=True)))
'''
    print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT)], outside))
    reference = work / 'reference'
    reference.mkdir()
    shutil.copytree(ROOT / 'examples', reference / 'examples')
    source_env = ENV.copy()
    source_env['PYTHONPATH'] = str(ROOT)
    expected = [json.loads(call([sys.executable, '-m', 'diffuselab'] + args,
                                reference, source_env)) for args in COMMANDS]
    model = json.loads((reference / 'model.json').read_text())
    for entry in [[py, '-I', '-m', 'diffuselab'], [str(work / 'env/bin/diffuselab')]]:
        (outside / 'model.json').unlink(missing_ok=True)
        for args, report in zip(COMMANDS, expected):
            # Compare every report leaf, including the full training history.
            actual_text = call(entry + args, outside)
            assert json.loads(actual_text) == report, (entry, args)
            if args[0] == 'train':
                assert json.loads((outside / 'model.json').read_text()) == model
            if args[0] == 'sampler-demo':
                (outside / 'demo.json').write_text(actual_text)
    figures = ['demo', 'samplers'] if selected != 'demo' else []
    for name in figures:
        print(call([py, '-I', str(outside / 'examples' / ('render_' + name + '.py')),
                    'demo.json', '--check'], outside))
    print(json.dumps(dict(complete_reports=len(COMMANDS), entrypoints=2,
                          complete_model_artifacts=2, reproduced_figures=len(figures))))
