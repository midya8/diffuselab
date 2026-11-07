import math
import unittest
from unittest.mock import patch
from diffuselab.core import Network, Schedule, sample
from diffuselab.sampling import ddim_step, timestep_grid, sample_ddim


class SamplingTests(unittest.TestCase):
    def test_grid_endpoints_unique_and_count(self):
        for total in [2,7,64,256]:
            s=Schedule(total)
            for steps in range(1,total+1):
                grid=timestep_grid(s,steps)
                self.assertEqual(len(grid),steps)
                self.assertEqual(len(set(grid)),steps)
                self.assertEqual(grid[0],total)
                self.assertTrue(all(a>b for a,b in zip(grid,grid[1:])))
                if steps>1:self.assertEqual(grid[-1],1)

    def test_full_grid(self):
        self.assertEqual(timestep_grid(Schedule(4),4),[4,3,2,1])

    def test_ddpm_step_equivalence(self):
        s=Schedule()
        for t in range(1,65):
            a=s.reverse([.3,-.7],t,[.4,-.2],[.8,-.1])
            b=ddim_step(s,[.3,-.7],t,t-1,[.4,-.2],[.8,-.1],1.)
            for x,y in zip(a,b):self.assertAlmostEqual(x,y,places=12)

    def test_ddpm_full_sampler_matched_draws(self):
        n,s=Network(4),Schedule(9)
        a=sample(n,s,12,11)
        b=sample_ddim(n,s,12,11,9,1.)['points']
        for x,y in zip(a,b):
            for u,v in zip(x,y):self.assertAlmostEqual(u,v,places=11)

    def test_oracle_epsilon_deterministic_path(self):
        s=Schedule()
        clean,eps=[.3,-.4],[.7,-1.2]
        for t,prev in [(64,40),(40,7),(7,0)]:
            noisy=s.forward(clean,t,eps)
            actual=ddim_step(s,noisy,t,prev,eps,[100.,-100.],0.)
            expected=[math.sqrt(s.abar[prev])*x+math.sqrt(1-s.abar[prev])*e for x,e in zip(clean,eps)]
            for x,y in zip(actual,expected):self.assertAlmostEqual(x,y,places=12)

    def test_skip_variance_independent_scalar_formula(self):
        s=Schedule()
        t,prev=55,8
        eta=.7
        a,ap=s.abar[t],s.abar[prev]
        sig=eta*math.sqrt((1-ap)/(1-a))*math.sqrt(1-a/ap)
        x,e,z=.2,-.5,.8
        expected=math.sqrt(ap/a)*(x-math.sqrt(1-a)*e)+math.sqrt(1-ap-sig*sig)*e+sig*z
        self.assertAlmostEqual(ddim_step(s,[x,x],t,prev,[e,e],[z,z],eta)[0],expected)

    def test_final_jump_no_noise(self):
        s=Schedule()
        a=ddim_step(s,[.5,.6],64,0,[.2,.3],[0.,0.],1.)
        b=ddim_step(s,[.5,.6],64,0,[.2,.3],[1e9,-1e9],1.)
        self.assertEqual(a,b)

    def test_deterministic_eta_uses_only_initial_gaussians(self):
        import random
        original=random.Random.gauss
        draws=[]
        def logged(r,mu,sd):
            draws.append(1)
            return original(r,mu,sd)
        n,s=Network(3),Schedule()
        with patch('random.Random.gauss',logged):
            a=sample_ddim(n,s,5,3,8,0.)
        self.assertEqual(len(draws),10)
        self.assertEqual(a,sample_ddim(n,s,5,3,8,0.))
        self.assertNotEqual(a['points'],sample_ddim(n,s,5,4,8,0.)['points'])

    def test_counted_denoiser_calls_and_no_mutation(self):
        class Counter:
            def __init__(self):self.calls=0
            def predict(self,x,t,s):self.calls+=1;return [0.,0.]
        n,s=Counter(),Schedule()
        r=sample_ddim(n,s,5,3,8)
        self.assertEqual(n.calls,40)
        self.assertEqual(r['denoiser_calls'],40)
        self.assertEqual(r['denoiser_calls_per_sample'],8)
        self.assertEqual(r['call_reduction_vs_full_ddpm'],.875)
        net=Network(3);old=list(net.params)
        sample_ddim(net,s,5,3,8)
        self.assertEqual(net.params,old)

    def test_invalid_arguments(self):
        n,s=Network(),Schedule()
        for kwargs in [{'eta':-1},{'eta':1.1},{'eta':True},{'eta':float('nan')},
                       {'steps':0},{'steps':65},{'steps':True},{'count':0},{'seed':-1}]:
            with self.assertRaises(ValueError):sample_ddim(n,s,**kwargs)
        for prev in [-1,5,True]:
            with self.assertRaises(ValueError):ddim_step(s,[0.,0.],5,prev,[0.,0.],[0.,0.])

    def test_nonfinite_output_fails(self):
        with self.assertRaises(ValueError):
            ddim_step(Schedule(),[1e308,1e308],64,0,[-1e308,-1e308],[0.,0.])
