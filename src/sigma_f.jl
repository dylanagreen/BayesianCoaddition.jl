# include("BayesianCoaddition.jl")
# Named this S because it actually comes from the "scaling gaussian"
# that results when multiplying two gaussians (the prior and the likelihood)
function log_S(likelihood::CoaddLikelihood, sigma_f)
  sigma_f_matrix = diagm(ones(length(likelihood.phi)) .* (sigma_f^-2))
  A_prior = sparse(likelihood.A .+ sigma_f_matrix)
  return -logdet(A_prior) * 0.5 - length(likelihood.phi) * log(sigma_f) + 0.5 * likelihood.phi' * (A_prior \ likelihood.phi)
end

# Used for exact root finding.
function dlog_S(likelihood::CoaddLikelihood, sigma_f)
    sigma_f_matrix = diagm(ones(length(likelihood.phi)) .* (sigma_f^-2))
    A_prior = Symmetric(likelihood.A .+ sigma_f_matrix)
    f_bar = A_prior \ likelihood.phi
    return (sigma_f^-2) * (tr(inv(A_prior)) + (f_bar' * f_bar)) - length(likelihood.phi)
end

function fit_and_find_vertex(sigma_vals, S_vals)
    # Finds the rough estimate of the vertex from the coarse evaluations we did
    idx = argmax(S_vals)

    # We will fit to the two points either side of the
    # estimated vertex, since the function is approximately
    # quadratic there (plus a logarithmic component to fit for the log
    # dependence in the true function)
    fit_range = idx - 2 : idx + 2

    if fit_range[1] < 1
        fit_range = fit_range .- fit_range[1] .+ 1
    elseif fit_range[end] > length(S_vals)
        fit_range = fit_range .- (fit_range[end] - length(S_vals))
    end

    fit_vals = sigma_vals[fit_range]
    X_log_fit = [fit_vals .^ 2 fit_vals log.(fit_vals) ones(length(fit_vals))]
    coeffs_log = X_log_fit \ S_vals[fit_range]

    # Analytic maximum found taking derivatives of the fit function and the quadratic formula
    vertex_log = (-coeffs_log[2] - sqrt(coeffs_log[2]^2 - 8 * coeffs_log[1] * coeffs_log[3]))/ (4 * coeffs_log[1])
    return vertex_log
end


function approximate_sigma_f(likelihood::CoaddLikelihood; n_iter::Int=2, n_points::Int=15)
  # Initial scan
  sigma_vals_full = Vector(range(0.05, 1, n_points))
  S_vals = Vector{Float64}()

  for sig in sigma_vals_full
      S = log_S(likelihood, sig)
      append!(S_vals, S)
  end

  vertex = fit_and_find_vertex(sigma_vals_full, S_vals)
  for i in range(0, n_iter)
      insert_idx_log = searchsorted(sigma_vals_full, vertex)

      # Inserts the estimated vertex and its log_s value
      # into the arrays for the next pass
      splice!(sigma_vals_full, insert_idx_log, vertex)
      splice!(S_vals, insert_idx_log, log_S(likelihood, vertex))

      vertex = fit_and_find_vertex(sigma_vals_full, S_vals)
  end

  # return vertex
  # Update the sufficient likelihood.
  return vertex
end

function update_sigma_f(likelihood::CoaddLikelihood)
    # TODO functionality for exact calculation if likelihood params are small enough
    new_sigma_f = approximate_sigma_f(likelihood)

    likelihood.sigma_f = new_sigma_f;
end

