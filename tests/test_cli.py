import json
import os
import subprocess
import sys
import tempfile
import unittest


class CLITests(unittest.TestCase):
    def test_train_sample(self):
        with tempfile.TemporaryDirectory() as d:
            inp,out=os.path.join(d,'points.json'),os.path.join(d,'model.json')
            with open(inp,'w') as f: json.dump([[0.,1.],[1.,0.]],f)
            p=subprocess.run([sys.executable,'-m','diffuselab','train',inp,out,'--updates','2'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            q=subprocess.run([sys.executable,'-m','diffuselab','sample',out,'--count','4'],capture_output=True,text=True)
            self.assertEqual(q.returncode,0,q.stderr)
            self.assertEqual(len(json.loads(q.stdout)['points']),4)
            q=subprocess.run([sys.executable,'-m','diffuselab','ddim',out,'--count','4','--steps','2'],capture_output=True,text=True)
            self.assertEqual(q.returncode,0,q.stderr)
            audit=json.loads(q.stdout)
            self.assertEqual(audit['denoiser_calls'],8)
            self.assertEqual(audit['timesteps'],[64,1])

    def test_invalid_artifact_clean_error(self):
        with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
            json.dump({'format':'bad'},f);f.flush()
            p=subprocess.run([sys.executable,'-m','diffuselab','sample',f.name],capture_output=True,text=True)
            self.assertEqual(p.returncode,2)
            self.assertIn('invalid model fields',p.stderr)
            self.assertNotIn('Traceback',p.stderr)
