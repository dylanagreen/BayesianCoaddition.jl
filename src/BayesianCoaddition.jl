module BayesianCoaddition
using LinearAlgebra

export CoaddLikelihood, accumulate_exposure, get_bayes_coadd, construct_likelihood
# export approximate_sigma_f

# include("approximation.jl")

mutable struct CoaddLikelihood
    A
    phi
    sigma_f::Float64
end

# Should be used to generate a coadd likelihood struct from a single exposure.
function construct_likelihood(flux, ivar, psf_mat)
    A = psf_mat * diagm(ivar) * psf_mat'
    phi = psf_mat * (flux .* ivar)
    return CoaddLikelihood(A, phi, 1)
end

function get_bayes_coadd(likelihood::CoaddLikelihood)
    sigma_f_matrix = diagm(ones(length(likelihood.phi)) .* (likelihood.sigma_f^-2))
    A_prior = sparse(likelihood.A .+ sigma_f_matrix)
    return A_prior \ likelihood.phi
end

function accumulate_exposure(likelihood::CoaddLikelihood, flux, ivar, psf_mat)
    likelihood.A += psf_mat * diagm(ivar) * psf_mat'
    likelihood.phi += psf_mat * (flux .* ivar)
end



end
