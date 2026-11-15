import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from diffuselab.core import Network, Schedule, artifact, load


class FiniteArithmeticTests(unittest.TestCase):
    def test_finite_loaded_model_rejects_overflow_before_tanh_hides_it(self):
        net = Network(1)
        net.params = [1e308,-1e308]+[0.]*8+[1.,1.]
        model = artifact(net,Schedule())
        loaded,schedule = load(json.loads(json.dumps(model,allow_nan=False)))
        for operation in [lambda: loaded.predict([2.,2.],1,schedule),
                          lambda: loaded.loss_grad([2.,2.,0.,0.,0.,0.,0.],[0.,0.])]:
            with self.assertRaisesRegex(ValueError,'hidden preactivation'):
                operation()
        # Finite parameters can also produce a positive infinity, which tanh
        # would otherwise silently saturate to one rather than reveal overflow.
        loaded.params[1] = 1e308
        with self.assertRaisesRegex(ValueError,'hidden preactivation'):
            loaded.run([2.,2.,0.,0.,0.,0.,0.])

    def test_output_loss_and_gradient_failures_are_distinct(self):
        net = Network(1)
        net.params = [0.]*12
        net.params[7] = 1.
        net.params[8] = 1e308
        net.params[10] = 1.5e308
        with self.assertRaisesRegex(ValueError,'network output'):
            net.run([0.]*7)
        net.params = [0.]*12
        with self.assertRaisesRegex(ValueError,'noise loss'):
            net.loss_grad([0.]*7,[1e308,0.])
        net.params[8] = 1e308
        with self.assertRaisesRegex(ValueError,'loss gradient'):
            net.loss_grad([2.,0.,0.,0.,0.,0.,0.],[2.,0.])

    def test_forward_rejects_nonfinite_derived_state(self):
        schedule = Schedule()
        with self.assertRaisesRegex(ValueError,'forward state'):
            schedule.forward([1.79e308,0.],32,[1.79e308,0.])
        for clean,noise in [([2.,-3.],[.1,.2]),([-1e308,1e308],[0.,0.])]:
            out = schedule.forward(clean,7,noise)
            expected = [math.sqrt(schedule.abar[7])*x+math.sqrt(1-schedule.abar[7])*z
                        for x,z in zip(clean,noise)]
            self.assertEqual(out,expected)

    def test_ordinary_finite_computation_keeps_exact_values(self):
        net = Network(3,17)
        f = [.2,-.7,.3,.4,-.5,.6,-.8]
        h,p = net.hidden,net.params
        hidden = [math.tanh(sum(p[j*7+k]*f[k] for k in range(7))+p[7*h+j]) for j in range(h)]
        expected = [sum(p[8*h+d*h+j]*hidden[j] for j in range(h))+p[10*h+d] for d in range(2)]
        self.assertEqual(net.run(f),(expected,hidden))
        loss,gradient = net.loss_grad(f,[.1,-.2])
        self.assertEqual(loss,sum((x-y)**2 for x,y in zip(expected,[.1,-.2]))/2)
        self.assertTrue(all(math.isfinite(x) for x in gradient))
        # Large but finite values are not rejected by a new magnitude cap.
        net = Network(1);net.params = [0.]*12
        net.params[10] = 1e150
        self.assertEqual(net.run([0.]*7)[0],[1e150,0.])
        self.assertEqual(net.loss_grad([0.]*7,[0.,0.])[0],1e150*1e150/2)

    def test_cli_rejects_nonfinite_network_without_stdout(self):
        net = Network(1)
        net.params = [0.]*12
        net.params[7] = 1.
        net.params[8] = 1e308
        net.params[10] = 1.5e308
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.json'
            path.write_text(json.dumps(artifact(net,Schedule()),allow_nan=False))
            for command,extra in [('sample',[]),('ddim',['--steps','16'])]:
                p = subprocess.run([sys.executable,'-m','diffuselab',command,str(path),'--count','1',*extra],capture_output=True,text=True)
                self.assertEqual(p.returncode,2,p.stderr)
                self.assertEqual(p.stdout,'')
                self.assertIn('network output',p.stderr)
                self.assertNotIn('Traceback',p.stderr)
