# BayesianCoaddition

Julia package implementing the Bayesian Coaddition methodology explained in [my thesis].

## How to Use
This package provides a struct `CoaddLikelihood` which contains all of the
necessary information and sufficient statistics stored in the system of exposures.

To initialize a `CoaddLikelihood` use the flux, ivar and psf matrix of the
first exposure in the coadd:

```julia
likelihood = construct_likelihood(flux, ivar, psf_mat)
```

Additional exposures are then accumulated into the coadd through
```julia
accumulate_exposure(likelihood, flux, ivar, psf_mat);
```

Once all exposures have been accumulated you **must** update the coadd's properties.
There are three ways to do this:
1. `update_sigma_f(likelihood, method="approx")`: Uses the accumulated likelihood to find the most probable $\sigma_f$ to regularize the Bayesian coadd. `method` controls whether to use the fast approximation algorithm outlined in the thesis (`approx`) or an exact root finder (`exact`).
2. `update_recon_props(likelihood)`: Uses the summary statistics to generate the "coadd PSF" and the associated reconvolved inverse variance.
3. `update_coadd_props(likelihood, sigma_f_method="approx")`. Combines both of the aforementinog functions into one single function call.

In general `update_coadd_props(likelihood, sigma_f_method="approx")` is satisfactory for most uses. `update_sigma_f` and `update_recon_props` are provided as convenience functions for users who know which coadd they want to produce and want to generate the coadd as fast as possible (i.e. without updating the properties they won't use).

To get the Bayesian coadd use
```julia
coadd = get_bayes_coadd(likelihood, regularize=false)
```
Set `regularize=true` to use the $\sigma_f$ regularization for coadd extraction.

To get the reconvolved coadd use
```julia
coadd, ivar = get_reconvolved_coadd(likelihood)
```
