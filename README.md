# DiffuseLab

**Train a diffusion model without hiding the reverse process.** A dependency-free 2-D DDPM with a learned noise predictor, explicit Gaussian arithmetic, handwritten backpropagation and tests that catch mode-collapse-friendly metrics.

![Samples and controls on the same axes](assets/diffusion-audit.svg)

## Run it offline

```sh
python -m unittest discover -s tests -v
python -m diffuselab demo > /tmp/diffuse-demo.json
python examples/render_demo.py /tmp/diffuse-demo.json --check
python -m diffuselab train examples/points.json /tmp/model.json --updates 1000
python -m diffuselab sample /tmp/model.json --count 128 --seed 17
```

Python 3.10+. No libraries, downloads, GPU, account or paid API. The full demo trains for 4,000 updates and takes about 20 seconds on the development machine; hardware and Python version affect runtime. The eight-point CLI file is only a smoke example, not the demo training set.

## What is implemented

- A linear beta schedule, closed-form forward noising and a 1-based timestep contract.
- A `7 -> tanh(48) -> 2` noise predictor: two coordinates, normalized time and four sine/cosine features. 482 learned parameters, exact MSE gradients, minibatch Adam.
- Epsilon-parameterized DDPM reverse means with the **forward posterior variance** as fixed reverse variance. No final-step noise, clipping, EMA, learned variance or checkpoint selection.
- Seeded training and sampling, strict `diffuselab_v1` JSON model artifacts, offline CLI.
- Separate reference data, a same-initialization untrained reverse-process control, and a standard Gaussian-prior control using the same initial sample seed.
- Distance, near-center fraction, per-mode counts, nearby-mode coverage and symmetric empirical nearest-neighbor squared distance. A deliberately collapsed generator passes center-distance metrics but fails coverage and the symmetric metric in a test.

See [DESIGN.md](DESIGN.md) for equations, data boundaries and the artifact contract.

## Measured development fixture

Eight equally likely centers on a radius-2 circle with isotropic noise SD 0.12. 1,024 training points (seed 31), 256 fresh reference points (seed 37). Network seed 7, training draws seed 8, sample seed 17. No seed, architecture, schedule or parameter search was used for this initial result. The fixture was visible during development and is **not an untouched benchmark**.

| Output, 256 samples | Mean nearest-center distance | Within 0.4 | Modes with >=5 nearby samples | Symmetric NN squared distance |
| --- | ---: | ---: | ---: | ---: |
| Learned DDPM | 0.378199 | 60.55% | 8/8 | 0.037103 |
| Untrained same network | 20.100807 | 0.00% | 0/8 | 259.754704 |
| Standard Gaussian prior | 0.977930 | 7.42% | 1/8 | 0.391052 |

Fresh-draw noise-prediction MSE (one mean per coordinate, fixed seed 43): **0.356809 trained vs 0.962031 untrained**. The snapshot training losses use different random minibatches, so they are not a monotone learning curve.

The learned model reaches all eight neighborhoods but is imperfect: 39.45% of samples are farther than 0.4 from any center. Nearest-center bins are `[16, 36, 23, 28, 23, 37, 43, 50]`, not balanced. The untrained chain expands badly; it is an implementation control, not a competitive generative baseline. The Gaussian prior is an additional simple reference, not a trained Gaussian mixture.

## Audit, not hype

22 tests include all-parameter central finite differences, independent posterior-mean algebra, schedule indexing, final-step determinism, serialization, bounds, CLI and a mode-collapse counterexample. CI runs on Python 3.10-3.13 and regenerates the figure from the actual demo on 3.11.

This is an educational low-dimensional generator. It is not an image model, latent diffusion system, density estimator, FID evaluation, security tool or production library. Nearest-neighbor metrics depend on sample size and are not likelihoods or distributional guarantees. This demo does not isolate causal contributions of architecture, schedule or optimizer. The terminal alpha-bar is about 0.006123, not zero: starting from a standard normal is an approximation to the finite forward marginal. No privacy guarantee, robust generalization claim or real-data evaluation is made.

## Source credit

Independent educational implementation of the equations in **Ho, Jain and Abbeel (2020), [Denoising Diffusion Probabilistic Models](https://arxiv.org/html/2006.11239v2)**: forward marginal Eq. 4, posterior variance Eq. 7, epsilon reverse mean Eq. 11, simplified noise loss and Algorithms 1-2. Their published image-generation results are not results of this project. Source consulted for the mathematics, not copied code or pretrained weights.

MIT license.
