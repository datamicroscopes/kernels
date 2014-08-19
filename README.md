# microscopes-kernels [![Build Status](https://travis-ci.org/datamicroscopes/kernels.svg?branch=master)](https://travis-ci.org/datamicroscopes/kernels)

Contains various ergodic MCMC kernels (gibbs, slice, metropolis-hastings). Used to do inference on the non-parametric models found in both the [mixturemodel](https://github.com/datamicroscopes/mixturemodel) and the [irm](https://github.com/datamicroscopes/irm) project. More models (such as HDP-HMM and HDP-LDA) will be added in the future. Stay tuned!

### Installation
Follow the instructions in the [common](https://github.com/datamicroscopes/common) project for setting up your Anaconda environment and adding the necessary binstar channels. Then run

    $ conda install microscopes-kernels

### Examples
Here are a few ipython notebooks. More examples can be found in the `examples` directory.

- [2D Gaussian mixture model](http://nbviewer.ipython.org/github/datamicroscopes/kernels/blob/master/examples/gauss2d.ipynb)
- [Digit completion with MNIST data](http://nbviewer.ipython.org/github/datamicroscopes/kernels/blob/master/examples/mnist-predictions.ipynb)
- [Enron e-mail corpus clustering](http://nbviewer.ipython.org/github/datamicroscopes/kernels/blob/master/examples/enron-email.ipynb)
- [Cancer genomics clustering](http://nbviewer.ipython.org/github/datamicroscopes/kernels/blob/master/examples/golub-cancer.ipynb)
