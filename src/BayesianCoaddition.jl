module BayesianCoaddition
using LinearAlgebra
using SparseArrays
using Roots

export CoaddLikelihood, accumulate_exposure, get_bayes_coadd, construct_likelihood, update_coadd_props
export update_sigma_f
export update_recon_props, get_reconvolved_coadd

mutable struct CoaddLikelihood
    A
    phi
    sigma_f::Float64
    coadd_psf
    ivar
end

# Should be used to generate a coadd likelihood struct from a single exposure.
function construct_likelihood(flux, ivar, psf_mat)
    A = psf_mat * diagm(ivar) * psf_mat'
    phi = psf_mat * (flux .* ivar)
    return CoaddLikelihood(A, phi, 1, psf_mat, ivar)
end

"""
    get_bayes_coadd(likelihood::CoaddLikelihood; regularize::Bool=false)

Gets the Bayesian coadd from the coadded likelihood, optionally using regularization.

The Bayesian coadd without regularization is equivalent to the maximum likelihood
estimate of the true object scene defined by the likelihood.
The Bayesian coadd with regularization is the maximum a-priori estimate of the
true object scene defined by the likelihood and the prior defined by the
hyperparameter ``σ_f``.

# Arguments
- `likelihood`: The accumulated coadd likelihood.
- `regularize`: Whether or not to return the regularized coadd or not.
"""
function get_bayes_coadd(likelihood::CoaddLikelihood; regularize::Bool=false)
    if regularize
        sigma_f_matrix = diagm(ones(length(likelihood.phi)) .* (likelihood.sigma_f^-2))
        A_prior = sparse(likelihood.A .+ sigma_f_matrix)
        return A_prior \ likelihood.phi
    else
        return likelihood.A \ likelihood.phi
    end
end

"""
    accumulate_exposure(likelihood::CoaddLikelihood, flux, ivar, psf_mat)

Add an exposure into the accumulated likelihood using the point spread function (PSF),
exposure flux and exposure inverse variance.

# Arguments
- `likelihood`: The accumulated coadd likelihood to add the exposure to.
- `flux`: Exposure flux as a vector.
- `ivar`: Exposure ivar as a vector.
- `psf_mat`: Matrix of PSF function for the exposure, where each row is the PSF of the associated pixel in the flux/ivar grid.
"""
function accumulate_exposure(likelihood::CoaddLikelihood, flux, ivar, psf_mat)
    likelihood.A += psf_mat * diagm(ivar) * psf_mat'
    likelihood.phi += psf_mat * (flux .* ivar)
end


include("sigma_f.jl")
include("reconvolution.jl")

function update_coadd_props(likelihood::CoaddLikelihood, sigma_f_method="approx")
    update_recon_props(likelihood)
    update_sigma_f(likelihood, sigma_f_method)
end

end
