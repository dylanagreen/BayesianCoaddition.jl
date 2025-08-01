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

function get_bayes_coadd(likelihood::CoaddLikelihood; regularize::Bool=false)
    if regularize
        sigma_f_matrix = diagm(ones(length(likelihood.phi)) .* (likelihood.sigma_f^-2))
        A_prior = sparse(likelihood.A .+ sigma_f_matrix)
        return A_prior \ likelihood.phi
    else
        return likelihood.A \ likelihood.phi
    end
end

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
