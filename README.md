# Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning

This repository contains code for the paper
[Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning]()

```bibtex

```
# Introduction
**Cyclical Stochastic Gradient MCMC (cSG-MCMC)** is proposed to efficiently explore complex multimodal distributions, such as those encountered for modern deep neural networks. The key idea is to adapt a cyclical stepsize schedule, where larger steps discover new modes, and smaller steps characterize each mode. We prove that our proposed learning rate schedule provides faster convergence to samples from a stationary distribution than SG-MCMC with standard decaying schedules. Below is an illustration of the proposed cyclical stepsize schedule (red) and the traditional decreasing stepsize schedule (blue) for SG-MCMC algorithms.

<p align="center">
  <img src="figs/lr-exp.png" width="500">
</p>


# Dependencies
* [PyTorch 0.3.1](http://pytorch.org/) 
* [torchvision 0.2.0](https://github.com/pytorch/vision/)

# Experiments
## Gaussian Mixture Density (25 Gaussians)

To generate samples from mixture of Gaussians (single chain results), please run

```
experiments/mog25.m
```
The two stepsize schedules are used in lines

    dsgld = sgld( gradUNoise, etaSGLD, L, x0, V );
    dcsgld = csgld( gradUNoise, etacSGLD, L, M, x0, V );

## CIFAR10
To train models with cSG-MCMC on CIFAR10, run:
```
python experiments/cifar10_cSGMCMC.py --dir=<DIR> \
                                      --data_path=<PATH> \
                                      --alpha=<ALPHA>
```
Parameters:

* ```DIR``` &mdash; path to training directory where samples will be stored
* ```PATH``` &mdash; path to the data directory
* ```ALPHA``` &mdash; One minus the momentum term. One is corresponding to SGLD and a number which is less than one is corresponding to SGHMC. (default: 1)

## CIFAR100

Similarly, for CIFAR100, run

```
python experiments/cifar100_cSGMCMC.py --dir=<DIR> \
                                      --data_path=<PATH> \
                                      --alpha=<ALPHA>
```

# Evaluating Samples
## Gaussian Mixture Density (25 Gaussians)
To visualize the results, please use ipython notebook to open the file
```
experiments/plot_density.ipynb
```
Cached results from our runs are included.

## CIFAR10
To test the ensemble of the collected samples on CIFAR10, run
```
experiments/cifar10_ensemble.py
```

## CIFAR100
To test the ensemble of the collected samples on CIFAR100, run
```
experiments/cifar100_ensemble.py
```

# Results
## Gaussian Mixture Density

Sampling from a mixture of 25 Gaussians in the non-parallel setting. (one single chain)

|  SGLD  |   cSGLD 
|:-------------------------:|:-------------------------:
| <img src="figs/sgld.png" width=200>  |   <img src="figs/csgld.png" width=200>


Sampling from a mixture of 25 Gaussians in the parallel setting. (4 chains)

|  SGLD  |   cSGLD 
|:-------------------------:|:-------------------------:
| <img src="figs/psgld.png" width=200>  |   <img src="figs/pcsgld.png" width=200>


## CIFAR
Test Error (%) on CIFAR10 and CIFAR100.

| Dataset                   |  SGLD        | cSGLD        | SGHMC            | cSGHMC          |
| ------------------------- |:------------:|:------------:|:----------------:|:---------------:|
| CIFAR10                   | 5.20 ± 0.06  | 4.29 ± 0.06  | 4.93 ± 0.1       | 4.27 ± 0.03     |
| CIFAR100                  | 23.23 ± 0.01 | 20.55 ± 0.06 | 22.60 ± 0.17     | 20.50 ± 0.11    |


# References
* Code of Gaussian mixtures is adapted from https://github.com/tqchen/ML-SGHMC
* Models are adapted from https://github.com/kuangliu/pytorch-cifar
