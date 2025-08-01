function res_decomp(A)
    v, P = eigen(A)
    # Generates the square root of inverse covariance matrix
    Q = P * diagm(sqrt.(v)) * transpose(P)

    # The resolution matrix
    # Inverse of S just does 1/element on the diagonal since
    # by construction it's a diagonal matrix
    S = diagm(1 ./ sum(Q, dims=2)[:, 1])
    R = S * Q

    ivar = sum(Q, dims=2)[:, 1] .^ 2

    return R, ivar
end

function update_recon_props(likelihood::CoaddLikelihood, sigma_f_method="approx")
    # Updating the PSF and the associated ivar for the reconvolved coadd
    G, iv = res_decomp(likelihood.A)
    likelihood.coadd_psf = G
    likelihood.ivar = iv

end

function get_reconvolved_coadd(likelihood::CoaddLikelihood)
    return likelihood.coadd_psf * get_bayes_coadd(likelihood, regularize=false), likelihood.ivar
end