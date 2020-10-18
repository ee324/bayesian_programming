"""
This is code based on the first part of the tutorial found here:
https://docs.pymc.io/notebooks/getting_started.html

versions that work on my local:
arviz==0.10.0
numpy==1.19.2
pandas==1.1.3
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

import os
import pymc3 as pm
import pandas as pd
import arviz as az


def simulate_stochastic_volatility():

    data_file = "data/SP500.csv"
    assert os.path.isfile(data_file)

    print("data file", data_file)

    # info about pm.get_data: https://docs.pymc.io/api/data.html
    returns = pd.read_csv(data_file, parse_dates=True, index_col=0, usecols=["Date", "change"])

    # note, the tutorial had this line of code, but it generated errors on my local...
    # returns = pd.read_csv(pm.get_data("SP500.csv"), parse_dates=True, index_col=0, usecols=["Date", "change"])

    print("Total return samples", len(returns))

    with pm.Model() as sp500_model:

        # The model remembers the datetime index with the name 'date'
        change_returns = pm.Data("returns", returns["change"], dims="date", export_index_as_coords=True)

        nu = pm.Exponential("nu", 1/10.0, testval=5.0)
        sigma = pm.Exponential("sigma", 2.0, testval=0.1)
        s = pm.GaussianRandomWalk("s", sigma=sigma, dims="date")

        volatility_process = pm.Deterministic("volatility_process", pm.math.exp(-2 * s) ** 0.5, dims="date")
        # From tutorial notes:
        # Notice that we transform the log volatility process s into the volatility process by exp(-2*s).
        # Here, exp is a Theano function, rather than the corresponding function in NumPy;
        # Theano provides a large subset of the mathematical functions that NumPy does.

        r = pm.StudentT("r", nu=nu, sigma=volatility_process, observed=change_returns, dims="date")

        print("RV_dims", sp500_model.RV_dims)
        print("coordinates", sp500_model.coords)

        # just a small number of samples to speed up testing...
        trace = pm.sample(50, init="adapt_diag", return_inferencedata=False)

        print("nu", len(trace["nu"]), trace["nu"][-5:])
        print("sigma", len(trace["sigma"]), trace["sigma"][-5:])

        print(az.summary(trace, round_to=2))


if __name__ == '__main__':
    simulate_stochastic_volatility()
