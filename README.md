# Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning

This repository contains code for the paper
[Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning]()

```bibtex

```

# Dependencies
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision/)

# Experiments
## Toy dataset (25 Gaussians)

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
## Toy dataset (25 Gaussians)
To visualize the results, please use ipython notebook to open the file
```
experiments/plot_density.ipynb
```
Cached results from our runs are inlcuded.

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

# References
* Code of Gaussian mixtures is adapted from https://github.com/tqchen/ML-SGHMC
* Models are adapted from https://github.com/kuangliu/pytorch-cifar