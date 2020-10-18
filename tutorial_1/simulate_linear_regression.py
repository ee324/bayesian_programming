"""
This is code based on the first part of the tutorial found here:
https://docs.pymc.io/notebooks/getting_started.html

versions that work on my local:
arviz==0.10.0
numpy==1.19.2
pymc3==3.9.3

IMPORTANT:
I received an error when running this for the first time:

This probably means that you are not using fork to start your
child processes and you have forgotten to use the proper idiom
in the main module:

    if __name__ == '__main__':
        freeze_support()
        ...

The "freeze_support()" line can be omitted if the program
is not going to be frozen to produce an executable.

To overcome this error, I had to add if __name__ == '__main__': to make this work as a standalone script
Works fine after that change :)

"""

import pymc3 as pm
import numpy as np
import arviz as az


def simulate_linear_regression():

    print("Running on PyMC3 v{}".format(pm.__version__))

    # true values, lets see if we can get these actual values from the posterior summary...
    alpha, sigma = 1, 1
    beta = [1, 2.5]

    # predictor variables...
    size = 100
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2

    print("Random data generated...")

    # Simulate outcome variable
    # so this could have been real data we observe...
    Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma

    print("outcome variable simulated...")

    # this part is where you start...

    basic_model = pm.Model()

    with basic_model:

        # priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # expected value
        mu = alpha + beta[0] * X1 + beta[1] * X2

        # likelihood function...
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

        print("Running MAP estimate...")

        map_estimate = pm.find_MAP(model=basic_model)
        print(map_estimate)

        print("Running sampling...")
        trace = pm.sample(500, return_inferencedata=False)

        print("alpha", len(trace["alpha"]), trace["alpha"][-5:])
        print("beta", len(trace["beta"]), trace["beta"][-5:])
        print("sigma", len(trace["sigma"]), trace["sigma"][-5:])

        axes = az.plot_trace(trace)
        fig = axes.ravel()[0].figure
        fig.savefig("tutorial_1/plots/arviz_plot_basic.png")

        print(az.summary(trace, round_to=2))


if __name__ == '__main__':
    simulate_linear_regression()
