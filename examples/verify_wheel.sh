#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps --wheel-dir "$tmp/dist" "$root"
python -m venv "$tmp/env"
"$tmp/env/bin/python" -m pip install --no-deps "$tmp"/dist/*.whl
cp -R "$root/examples" "$root/assets" "$tmp/"
cd "$tmp"
"$tmp/env/bin/python" -I - "$root" <<'PY'
import importlib
import importlib.util
import pathlib
import sys
import unittest
root = pathlib.Path(sys.argv[1])
names = ['diffuselab','diffuselab.core','diffuselab.cli','diffuselab.demo',
         'diffuselab.sampling','diffuselab.__main__']
# __main__ intentionally invokes the CLI when imported; check its spec only.
for name in names[:-1]: importlib.import_module(name)
def check():
    for name in names:
        origin = pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert 'site-packages' in str(origin) and not origin.is_relative_to(root)
check()
sys.path.insert(0, str(pathlib.Path.cwd()))
suite = unittest.defaultTestLoader.discover(str(root/'tests'))
assert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
check()
print('Six installed module origins before and after the full test suite')
PY
"$tmp/env/bin/python" -I -m diffuselab sampler-demo > installed.json
"$tmp/env/bin/python" -I examples/render_demo.py installed.json --check
"$tmp/env/bin/python" -I examples/render_samplers.py installed.json --check
"$tmp/env/bin/python" -I -m diffuselab train "$root/examples/points.json" model.json --updates 2 > train.json
"$tmp/env/bin/python" -I -m diffuselab sample model.json --count 8 --seed 17 > sample.json
"$tmp/env/bin/python" -I -m diffuselab ddim model.json --steps 16 --eta 0 --count 8 > ddim.json
"$tmp/env/bin/diffuselab" sample model.json --count 8 --seed 17 > console.json
cmp sample.json console.json
if [ "$#" -gt 0 ]; then cmp "$1" installed.json; fi
printf '%s\n' 'Complete sampler demo, both figures, training, DDPM/DDIM and console replay pass'
