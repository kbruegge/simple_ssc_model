# A simple self-synchrotron compton (SSC) model

These scripts use the great naima library [https://naima.readthedocs.io/en/latest/]
to model the gamma-ray emission from the Crab Nebula. Very similar to the 
example provided in the naima documentation.

Executing the `meyer_model` script creates a lookup table of model values. 
The `naima_fit_lut` script then uses scipy's `RegularGridInterpolator` to make a continuos function 
out of the SSC model. This interpolated function can then be used to fit data using the emcee sampler.

The Crab data given in the fits file here was taken from the gammapy project[gammapy.org].