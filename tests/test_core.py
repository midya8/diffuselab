import copy
import json
import math
import unittest
from diffuselab.core import Schedule, Network, features, train, sample, artifact, load
from diffuselab.demo import fixture, metrics, noise_mse


class ScheduleTests(unittest.TestCase):
    def test_forward_closed_form(self):
        s=Schedule(2,.1,.2)
        self.assertAlmostEqual(s.abar[2],.72)
        self.assertEqual(s.forward([2.,-1.],1,[0.,0.]),[2*math.sqrt(.9),-math.sqrt(.9)])
        self.assertAlmostEqual(s.forward([2.,-1.],2,[1.,-2.])[0],2*math.sqrt(.72)+math.sqrt(.28))

    def test_posterior_variance(self):
        s=Schedule(2,.1,.2)
        self.assertEqual(s.variance[1],0.)
        self.assertAlmostEqual(s.variance[2],.2*.1/.28)

    def test_reverse_matches_posterior_mean(self):
        s=Schedule(9,.01,.2)
        for t in range(1,10):
            x0,eps=[.7,-.2],[.1,-1.3]
            xt=s.forward(x0,t,eps)
            actual=s.reverse(xt,t,eps,[0.,0.])
            for d in range(2):
                expected=(math.sqrt(s.abar[t-1])*s.beta[t]*x0[d]
                          +math.sqrt(1-s.beta[t])*(1-s.abar[t-1])*xt[d])/(1-s.abar[t])
                self.assertAlmostEqual(actual[d],expected,places=12)

    def test_final_step_ignores_noise(self):
        s=Schedule()
        self.assertEqual(s.reverse([1.,2.],1,[.2,.3],[0.,0.]),s.reverse([1.,2.],1,[.2,.3],[1e9,-1e9]))

    def test_reverse_noise_scale(self):
        s=Schedule()
        a=s.reverse([0.,0.],7,[0.,0.],[1.,-1.])
        self.assertAlmostEqual(a[0],math.sqrt(s.variance[7]))
        self.assertAlmostEqual(a[1],-math.sqrt(s.variance[7]))

    def test_invalid_schedule(self):
        for args in [(True,.01,.1),(1,.01,.1),(257,.01,.1),(8,.2,.1),(8,0,.1),(8,.01,float('nan'))]:
            with self.subTest(args=args),self.assertRaises(ValueError): Schedule(*args)

    def test_invalid_time(self):
        s=Schedule()
        for t in [0,65,True,1.2]:
            with self.assertRaises(ValueError): s.forward([0.,0.],t,[0.,0.])

    def test_invalid_vector(self):
        for x in [[1.], [True,0.], [float('inf'),0.], 'xx']:
            with self.assertRaises(ValueError): Schedule().forward(x,1,[0.,0.])


class NetworkTests(unittest.TestCase):
    def test_every_parameter_gradient(self):
        n=Network(3,5)
        f=[.2,-.7,.1,.4,-.3,.6,.8]
        y=[.5,-.2]
        _,grad=n.loss_grad(f,y)
        for j in range(len(n.params)):
            original=n.params[j]
            n.params[j]=original+1e-6
            plus=n.loss_grad(f,y)[0]
            n.params[j]=original-1e-6
            minus=n.loss_grad(f,y)[0]
            n.params[j]=original
            self.assertAlmostEqual(grad[j],(plus-minus)/2e-6,places=8)

    def test_loss_normalization(self):
        n=Network(1)
        n.params=[0.]*len(n.params)
        loss,g=n.loss_grad([0.]*7,[2.,4.])
        self.assertEqual(loss,10.)
        self.assertEqual(g[-2:],[-2.,-4.])

    def test_features(self):
        s=Schedule()
        f=features([2.,3.],64,s)
        self.assertEqual(f[:3],[2.,3.,1.])
        self.assertAlmostEqual(f[3],0.,places=12)
        self.assertAlmostEqual(f[4],-1.)

    def test_reproducible_training_and_no_mutation(self):
        p=fixture(16,1); before=copy.deepcopy(p)
        a,ha=train(p,hidden=3,updates=4,batch_size=2)
        b,hb=train(p,hidden=3,updates=4,batch_size=2)
        self.assertEqual(a.params,b.params)
        self.assertEqual(ha,hb)
        self.assertEqual(p,before)
        self.assertNotEqual(a.params,Network(3).params)

    def test_sampling_reproducible_and_no_mutation(self):
        n,s=Network(3),Schedule(4)
        old=list(n.params)
        a=sample(n,s,8,4)
        self.assertEqual(a,sample(n,s,8,4))
        self.assertNotEqual(a,sample(n,s,8,5))
        self.assertEqual(n.params,old)
        self.assertTrue(all(math.isfinite(x) for row in a for x in row))

    def test_round_trip_exact(self):
        n,s=Network(3),Schedule(9,.01,.1)
        a=artifact(n,s)
        loaded,ls=load(json.loads(json.dumps(a)))
        self.assertEqual(sample(n,s,4),sample(loaded,ls,4))
        a['parameters'][0]=999.
        self.assertNotEqual(n.params[0],999.)

    def test_strict_artifact(self):
        base=artifact(Network(3),Schedule())
        variants=[]
        for field in base:
            x=copy.deepcopy(base); del x[field]; variants.append(x)
        x=copy.deepcopy(base);x['extra']=0;variants.append(x)
        x=copy.deepcopy(base);x['schedule']['extra']=0;variants.append(x)
        x=copy.deepcopy(base);x['parameters'][0]=float('nan');variants.append(x)
        x=copy.deepcopy(base);x['parameters'].pop();variants.append(x)
        x=copy.deepcopy(base);x['format']='wrong';variants.append(x)
        x=copy.deepcopy(base);x['hidden']=True;variants.append(x)
        for x in variants:
            with self.assertRaises(ValueError): load(x)

    def test_training_bounds(self):
        for kw in [{'updates':True},{'updates':0},{'batch_size':129},{'learning_rate':float('inf')},
                   {'hidden':0},{'seed':-1},{'learning_rate':.2}]:
            with self.assertRaises(ValueError): train([[0.,0.],[1.,1.]],**kw)
        for p in [[], [[0.,0.]], [[101.,0.],[1.,0.]], [[0.,float('nan')],[1.,0.]]]:
            with self.assertRaises(ValueError): train(p,updates=1)

    def test_sample_bounds(self):
        for n in [0,True,2049]:
            with self.assertRaises(ValueError): sample(Network(),Schedule(),n)

    def test_noise_mse_same_draws(self):
        n,s,p=Network(),Schedule(),fixture(16,1)
        self.assertEqual(noise_mse(n,s,p,3),noise_mse(n,s,p,3))

    def test_metrics_do_not_hide_collapse(self):
        p=[[2.,0.]]*256
        r=metrics(p,fixture(256,1))
        self.assertEqual(r['mean_nearest_center_distance'],0.)
        self.assertEqual(r['within_0_4_of_center'],1.)
        self.assertEqual(r['modes_with_at_least_5_nearby_samples'],1)
        self.assertGreater(r['symmetric_nn_squared_distance'],1.)

    def test_mode_counts(self):
        from diffuselab.demo import centers
        p=centers()*5
        r=metrics(p,p)
        self.assertEqual(r['modes_with_at_least_5_nearby_samples'],8)
        self.assertEqual(r['symmetric_nn_squared_distance'],0.)


if __name__=='__main__': unittest.main()
