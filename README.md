# astro-forecaster (forecaster 2ish)
-----

-----

An internally overhauled but fundamentally similar version of Forecaster by Jingjing Chen and David Kipping, originally presented in [Probabilistic Forecasting of the Masses and Radii of Other Worlds](https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract) and hosted at [here](https://github.com/chenjj2/forecaster).

The model itself has not changed- no new data was included and the hyperparameter file was not regenerated. All functions were rewritten to take advantage of Numpy vectorization and some additional user features were added.

## Installation and basic use
-----
To install this package with [`pip`](https://pip.pypa.io/en/stable/), run

```bash
pip install astro-forecaster
```

As a basic example, here is how you would calculate the radius posterior of a 1 Jupiter mass object:

```python
import forecaster
import numpy as np

forecasted_radius_posterior = forecaster.Mpost2R(np.ones(int(1e3)), unit='Jupiter', classify=False)
```

See the [demo notebook](https://github.com/ben-cassese/astro-forecaster/blob/main/demo.ipynb) for more examples.

## Changes
-----
Changes include but are not limited to:
* Rewriting all functions to take advantage of Numpy vectorization
* Including the ability to input asymmetric uncertainties in mass or radius
* Enabling pip installation


## Citation
-----
If used, please cite [the original Forecaster paper](https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract) and the bibcode for this implementation eventually hosted on the Astrophysics Source Code Library (ASCL).

