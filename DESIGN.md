# DiffuseLab numerical choices

DiffuseLab learns a noise predictor for supplied two-dimensional points. The DDPM and DDIM samplers share that predictor and saved model. This note defines the equations, input bounds and synthetic comparison; the fixture results do not establish performance on other data.

## Diffusion arithmetic

Timesteps are integers `1..T`; `alpha_bar[0]=1` denotes clean data. With `alpha_t=1-beta_t`:

```
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon
mu_theta = (x_t - beta_t/sqrt(1-alpha_bar_t) epsilon_theta(x_t,t)) / sqrt(alpha_t)
variance_t = beta_t (1-alpha_bar_(t-1)) / (1-alpha_bar_t)
x_(t-1) = mu_theta + sqrt(variance_t) z, except z=0 at t=1
```

The fixed reverse variance is the conditional forward posterior variance from the paper, not `beta_t`. This conditional is exact when clean data is supplied; a learned reverse Gaussian with estimated epsilon is only a model. No coordinate clipping or clean-point projection is applied.

Schedule limits: 2-256 steps and `1e-6 <= beta_start <= beta_end <= .25`. These limits keep the supported finite schedule away from zero-denominator endpoints; they are not optimal hyperparameters. The demo uses 64 steps, .001 to .15, terminal alpha-bar 0.006122695811495588. It does not pretend finite terminal noise is exactly standard normal.

## Learning

Input `[x,y,u,sin(pi*u),cos(pi*u),sin(2*pi*u),cos(2*pi*u)]` with `u=t/T`. A single tanh hidden layer, linear two-coordinate output. Loss is `(error_x^2 + error_y^2)/2`, averaged over the minibatch. For two dimensions this is coordinate-mean squared error. Gradients propagate through every weight and bias. Adam uses beta1=.9, beta2=.999, epsilon=1e-8 and bias correction; no weight decay, clipping, EMA or validation-based selection.

Each training draw picks a training row with replacement, a uniform timestep and fresh independent Gaussian epsilon. The final update is used. Logged losses are stochastic minibatch snapshots, not a fixed evaluation series. The noise-MSE comparison uses the same fresh reference rows, timesteps and epsilon for both networks.

Bounds: hidden width 1-128; training rows 2-8192, finite 2-D coordinates in [-100,100]; updates 1-20000; batch size 1-128; learning rate [1e-6,.1]; seed integer [0,2^32-1]. Sampling accepts 1-2048 points. Limits constrain runtime and input shape, not convergence. Divergence/nonfinite reverse states raise errors. Training and sampling do not modify caller data or the model passed to sampling.

## Artifact

The exact keys are `format`, `hidden`, `parameters`, `schedule`; format is `diffuselab_v1`. Schedule keys are exactly `steps`, `beta_start`, `beta_end`. Parameters are finite numbers in row-major order: hidden input weights (7H), hidden biases (H), output weights (2H), output biases (2). Booleans are not numbers. Unknown/missing keys, wrong counts, unsupported versions and invalid settings fail. This is an inference artifact, not an optimizer checkpoint; Adam state is not saved and training cannot resume exactly from it.

## Fixture and controls

The training and fresh reference generators share eight equally weighted known centers but use different seeds. Both are synthetic. Training initialization is identical to the untrained control. The control and trained sampler consume the same initial Gaussian and step-noise streams. The prior uses the same initial Gaussian stream, without reverse steps. These are descriptive fixture comparisons, not causal estimates or broad rankings. The demo assertions detect regressions on this known fixture, not a general quality gate.

For output P and reference R, symmetric NN squared distance is half of mean_{p in P} min_{r in R} ||p-r||^2 plus mean_{r in R} min_{p in P} ||r-p||^2. Units are coordinate squared. Mode coverage requires at least 5 generated samples within radius 0.4 of a center. Nearest-center bins count every output, including far-away samples, so the separate nearby coverage is needed. Neither metric proves balanced frequencies, calibrated density or absence of memorization.

The SVG shares [-3,3] x/y axes for all four panels and reports outside-plot counts rather than silently dropping control failures. JSON keeps every sample. Floating-point reductions can vary slightly across Python/runtime versions; no cross-platform byte-identical model guarantee is made.


## DDIM selected-grid sampling

`sampling.py` uses the same artifact/network. For current t and selected previous s<t, with s=0 denoting clean output:

```
sigma^2 = eta^2 * (1-alpha_bar_s)/(1-alpha_bar_t) * (1-alpha_bar_t/alpha_bar_s)
x0_hat = (x_t - sqrt(1-alpha_bar_t)*epsilon_hat)/sqrt(alpha_bar_t)
x_s = sqrt(alpha_bar_s)*x0_hat + sqrt(1-alpha_bar_s-sigma^2)*epsilon_hat + sigma*z
```

The paper's alpha equals this code's alpha_bar. Eta is restricted to [0,1]; tiny negative variance radicands within 1e-12 can be rounded to zero, while larger negative values fail. This range is an implementation boundary, not a full characterization of valid eta values. No clean-point clipping is applied.

For S>=2 steps, ascending grid points are `1+floor(i*(T-1)/(S-1))`, i=0..S-1, then reversed. S=1 jumps from T to clean output. S must be an integer 1..T; all returned timesteps are unique and descending. The previous point is the next selected timestep, not t-1 unless the full schedule is used. The last selected timestep jumps to 0. Returned JSON includes the full grid, eta, total actual predictor calls, calls per sample and call reduction versus T-step DDPM. No sampler setting is stored in v1 inference artifacts.

Eta=0 consumes Gaussian draws only for the initial latent. Eta>0 consumes per-step draws only when previous>0. Full-grid eta=1 therefore consumes the same draws in the same order as the original sampler, and the algebra agrees within floating-point error. This equivalence test is about the sampler equations, not quality. Other grids or eta values do not preserve DDPM paths or output distributions. The oracle-epsilon skip test checks a known clean point/noise pair, not learned epsilon accuracy.
