# Structure-preserving approximations of the Serre-Green-Naghdi equations
# in standard and hyperbolic form
# Hendrik Ranocha and Mario Ricchiuto, 2024

# Load packages
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra: LinearAlgebra, Diagonal, Symmetric, cholesky, cholesky!, ldiv!, issuccess
using SparseArrays: sparse, issparse
using Statistics: mean, median, std
using DelimitedFiles: readdlm

using BenchmarkTools: BenchmarkTools, @benchmark, time
using Measurements: ±
using Unitful: Unitful, @u_str

using FastBroadcast: @..

using SummationByPartsOperators
using OrdinaryDiffEq

using DataStructures: top
using DiffEqCallbacks
using SimpleNonlinearSolve

using Optimization
using OptimizationOptimJL

using RecursiveArrayTools: ArrayPartition

using LaTeXStrings
using Plots: Plots, plot, plot!, scatter, scatter!, savefig, annotate!, text

using PrettyTables: PrettyTables, pretty_table, ft_printf



#####################################################################
# Helper utilities for plotting

const figdir = joinpath(dirname(@__DIR__), "figures")
if !isdir(figdir)
    mkdir(figdir)
end

function plot_kwargs()
    fontsizes = (
        xtickfontsize = 14, ytickfontsize = 14,
        xguidefontsize = 16, yguidefontsize = 16,
        legendfontsize = 14)
    (; linewidth = 3, gridlinewidth = 2,
        markersize = 8, markerstrokewidth = 4,
        fontsizes..., size=(600, 500))
end

function compute_eoc(Ns, errors)
    eoc = similar(errors)
    eoc[begin] = NaN # no EOC defined for the first grid
    for idx in Iterators.drop(eachindex(errors, Ns, eoc), 1)
        eoc[idx] = -( log(errors[idx] / errors[idx - 1]) / log(Ns[idx] / Ns[idx - 1]) )
    end
    return eoc
end

function compute_exponential_eoc(Ns, errors)
    eoc = similar(errors)
    eoc[begin] = NaN # no EOC defined for the first grid
    for idx in Iterators.drop(eachindex(errors, Ns, eoc), 1)
        eoc[idx] = exp( log(errors[idx] / errors[idx - 1]) / (Ns[idx] - Ns[idx - 1]) )
    end
    return eoc
end


#####################################################################
# The spatial semidiscretization = ODE RHS
# using nonconservative variables (h, v, w, η)
# for the hyperbolic approximation
function rhs_nonconservative!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D`
    (; g, λ, D) = parameters

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h`, the velocity `v`,
    # and the additional variables `w` and `η`.
    h, v, w, η = u.x
    dh, dv, dw, dη = du.x

    # Unpack the bottom topography and its derivative
    (; b, b_x) = parameters

    # Compute all derivatives required below.
    # We use some temporary storage in the parameters to improve the
    # performance by avoiding allocating new arrays all the time.
    (; η_over_h, h_x, v_x, hv_x, v2_x, h_hpb_x, η_x, η2_h_x, w_x, hvw_x, tmp) = parameters
    # h_x = D * d
    mul!(h_x, D, h)
    # v_x = D * v
    mul!(v_x, D, v)
    # hv2_x = D * (h * v)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    # v2_x = D * (v.^2)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)
    # h_hpb_x = D * (h .* (h + b)
    @.. tmp = h * (h + b)
    mul!(h_hpb_x, D, tmp)
    # η_x = D * η
    mul!(η_x, D, η)
    # η2_h_x = D * (η^2 / h)
    @.. η_over_h = η / h
    @.. tmp = η * η_over_h
    mul!(η2_h_x, D, tmp)
    # w_x = D * w
    mul!(w_x, D, w)
    # hvw_x = D * (h * v * w)
    @.. tmp = h * v * w
    mul!(hvw_x, D, tmp)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + h v v_x + g (h + b) h_x
    #              + ... = 0
    #
    # Split form for energy conservation:
    # h v_t + g (h (h + b))_x - g (h + b) h_x
    #       + 1/2 h (v^2)_x - 1/2 v^2 h_x  + 1/2 v (h v)_x - 1/2 h v v_x
    #       + λ/6 η^2 / h^2 h_x + λ/3 η_x - λ/3 η/h η_x - λ/6 (η^2 / h)_x
    #       + λ/2 b_x - λ/2 η/h b_x = 0
    λ_6 = λ / 6
    λ_3 = λ / 3
    λ_2 = λ / 2
    @.. dv = -(g * h_hpb_x - g * (h + b) * h_x
               + 0.5 * h * v2_x - 0.5 * v^2 * h_x
               + 0.5 * hv_x * v - 0.5 * h * v * v_x
               + λ_6 * (η_over_h * η_over_h * h_x - η2_h_x)
               + λ_3 * (1 - η_over_h) * η_x
               + λ_2 * (1 - η_over_h) * b_x) / h

    # Plain: h w_t + h v w_x = λ - λ η / h
    #
    # Split form for energy conservation:
    # h w_t + 1/2 (h v w)_x + 1/2 h v w_x
    #       - 1/2 h_x v w - 1/2 h w v_x = λ - λ η / h
    @.. dw = ( -(  0.5 * hvw_x
                 + 0.5 * h * v * w_x
                 - 0.5 * h_x * v * w
                 - 0.5 * h * w * v_x) + λ * (1 - η_over_h)) / h

    # No special split form for energy conservation required:
    # η_t + v η_x + 3/2 v b_x = w
    @.. dη = -v * η_x - 1.5 * v * b_x + w

    return nothing
end

function setup(::typeof(rhs_nonconservative!),
               h_func, v_func, b_func;
               g, λ, D)
    # Compute the initial state
    x = grid(D)
    h = h_func.(x)
    v = v_func.(x)
    w = -h .* (D * v)
    η = copy(h)
    u0 = ArrayPartition(h, v, w, η)

    # Compute the bottom topography and its derivative
    b = b_func.(x)
    b_x = D * b

    # create temporary storage
    η_over_h = zero(h)
    h_x = zero(h)
    v_x = zero(h)
    hv_x = zero(h)
    v2_x = zero(h)
    h_hpb_x = zero(h)
    η_x = zero(h)
    η2_h_x = zero(h)
    w_x = zero(h)
    hvw_x = zero(h)
    tmp = zero(h)

    parameters = (; η_over_h, h_x, v_x, hv_x,v2_x, h_hpb_x, η_x, η2_h_x,
                    w_x, hvw_x, tmp, b, b_x, g, λ, D)

    return u0, parameters
end

# The total energy for given states `u`
function energy(::typeof(rhs_nonconservative!),
                u, parameters)
    # unpack physical parameters and SBP operator `D`
    (; g, λ, D, b) = parameters

    # `u` is an `ArrayPartition`. It collects the individual arrays for
    # the water height `h`, the velocity `v`,
    # and the additional variables `w` and `η`.
    h, v, w, η = u.x

    # 1/2 g h^2 + 1/2 h v^2 + 1/6 h w^2 + λ/6 h (1 - η/h)^2
    E = parameters.tmp
    @.. E = 1/2 * g * (h + b)^2 + 1/2 * h * v^2 + 1/6 * h * w^2 + λ/6 * h * (1 - η/h)^2

    return integrate(E, D)
end

# The semidiscrete rate of change of the energy for given states `du, u`
function energy_rate(::typeof(rhs_nonconservative!),
                     du, u, parameters)
    # unpack physical parameters and SBP operator `D`
    (; g, λ, D, b) = parameters

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h`, the velocity `v`,
    # and the additional variables `w` and `η`.
    h, v, w, η = u.x
    dh, dv, dw, dη = du.x

    # E_h = g h + 1/2 v^2 + 1/6 w^2 + λ/6 (1 - η^2 / h^2)
    E_h = @.. g * (h + b) + 1/2 * v^2 + 1/6 * w^2 + λ/6 * (1 - η^2 / h^2)

    # E_v = h * v
    E_v = @.. h * v

    # E_w = h w / 3
    E_w = @.. h * w / 3

    # E_η = λ/3 (η/h - 1)
    E_η = @.. λ/3 * (η / h - 1)

    uEu = @.. -(E_h * dh + E_v * dv + E_w * dw + E_η * dη)

    return integrate(uEu, D)
end


#####################################################################
# The spatial semidiscretization = ODE RHS
# using nonconservative variables (h, v, w, η)
# for the hyperbolic approximation
# with artificial viscosity
function rhs_nonconservative_visc!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D`
    (; g, λ, mu,  D) = parameters

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h`, the velocity `v`,
    # and the additional variables `w` and `η`.
    h, v, w, η = u.x
    dh, dv, dw, dη = du.x

    # Unpack the bottom topography and its derivative
    (; b, b_x) = parameters

    # Compute all derivatives required below.
    # We use some temporary storage in the parameters to improve the
    # performance by avoiding allocating new arrays all the time.
    (; η_over_h, h_x, v_x, hv_x, v2_x, h_hpb_x, η_x, η2_h_x, w_x, hvw_x, tmp, fv_x) = parameters
    # h_x = D * d
    mul!(h_x, D, h)
    # v_x = D * v
    mul!(v_x, D, v)
    # hv2_x = D * (h * v)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    # v2_x = D * (v.^2)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)
    # h_hpb_x = D * (h .* (h + b)
    @.. tmp = h * (h + b)
    mul!(h_hpb_x, D, tmp)
    # η_x = D * η
    mul!(η_x, D, η)
    # η2_h_x = D * (η^2 / h)
    @.. η_over_h = η / h
    @.. tmp = η * η_over_h
    mul!(η2_h_x, D, tmp)
    # w_x = D * w
    mul!(w_x, D, w)
    # hvw_x = D * (h * v * w)
    @.. tmp = h * v * w
    mul!(hvw_x, D, tmp)

    @.. tmp = h * mu * v_x
    mul!(fv_x, D, tmp)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + h v v_x + g (h + b) h_x
    #              + ... = 0
    #
    # Split form for energy conservation:
    # h v_t + g (h (h + b))_x - g (h + b) h_x
    #       + 1/2 h (v^2)_x - 1/2 v^2 h_x  + 1/2 v (h v)_x - 1/2 h v v_x
    #       + λ/6 η^2 / h^2 h_x + λ/3 η_x - λ/3 η/h η_x - λ/6 (η^2 / h)_x
    #       + λ/2 b_x - λ/2 η/h b_x = 0
    λ_6 = λ / 6
    λ_3 = λ / 3
    λ_2 = λ / 2
    @.. dv = -(g * h_hpb_x - g * (h + b) * h_x
               + 0.5 * h * v2_x - 0.5 * v^2 * h_x
               + 0.5 * hv_x * v - 0.5 * h * v * v_x
               + λ_6 * (η_over_h * η_over_h * h_x - η2_h_x)
               + λ_3 * (1 - η_over_h) * η_x
               + λ_2 * (1 - η_over_h) * b_x
               - fv_x) / h

    # Plain: h w_t + h v w_x = λ - λ η / h
    #
    # Split form for energy conservation:
    # h w_t + 1/2 (h v w)_x + 1/2 h v w_x
    #       - 1/2 h_x v w - 1/2 h w v_x = λ - λ η / h
    @.. dw = ( -(  0.5 * hvw_x
                 + 0.5 * h * v * w_x
                 - 0.5 * h_x * v * w
                 - 0.5 * h * w * v_x) + λ * (1 - η_over_h)) / h

    # No special split form for energy conservation required:
    # η_t + v η_x + 3/2 v b_x = w
    @.. dη = -v * η_x - 1.5 * v * b_x + w

    return nothing
end

function setup(::typeof(rhs_nonconservative_visc!),
               h_func, v_func, b_func;
               g, mu, λ, D)
    u0, parameters = setup(rhs_nonconservative!,
                           h_func, v_func, b_func;
                           g, λ, D)
    fv_x = zero(parameters.tmp)

    parameters = (; parameters..., fv_x, mu)

    return u0, parameters
end

function energy(::typeof(rhs_nonconservative_visc!),
                u, parameters)
    energy(rhs_nonconservative!, u, parameters)
end

function energy_rate(::typeof(rhs_nonconservative_visc!),
                     du, u, parameters)
    energy_rate(rhs_nonconservative!, du, u, parameters)
end


#####################################################################
# The spatial semidiscretization = ODE RHS
# using nonconservative variables (h, v)
# for the classical Serre-Green-Naghdi equations
# with flat bathymetry
function rhs_serre_green_naghdi_flat!(du, u, parameters, t)
    D = parameters.D
    if D isa PeriodicUpwindOperators
        rhs_sgn_flat_upwind!(du, u, parameters, t)
    else
        rhs_sgn_flat_central!(du, u, parameters, t)
    end
    return nothing
end

function rhs_sgn_flat_central!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D` as well as the
    # SBP operator in sparse matrix form `Dmat`
    (; g, D, Dmat) = parameters

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # Compute all derivatives required below
    (; h_x, v_x, h2_x, hv_x, v2_x,
       h2_v_vx_x, h_vx_x, p_x, tmp,
       M_h, M_h3_3) = parameters
    mul!(h_x, D, h)
    mul!(v_x, D, v)
    @.. tmp = h^2
    mul!(h2_x, D, tmp)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)

    @.. tmp = h^2 * v * v_x
    mul!(h2_v_vx_x, D, tmp)
    @.. tmp = h * v_x
    mul!(h_vx_x, D, tmp)
    inv6 = 1 / 6
    @.. tmp = (  0.5 * h^2 * (h * v_x + h_x * v) * v_x
               - inv6 * h * h2_v_vx_x
               - inv6 * h^2 * v * h_vx_x)
    mul!(p_x, D, tmp)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + ... = 0
    #
    # Split form for energy conservation:
    @.. tmp = -(  g * h2_x - g * h * h_x
                + 0.5 * h * v2_x
                - 0.5 * v^2 * h_x
                + 0.5 * hv_x * v
                - 0.5 * h * v * v_x
                + p_x)
    # The code below is equivalent to
    #   dv .= (Diagonal(h) - Dmat * Diagonal(1/3 .* h.^3) * Dmat) \ tmp
    # but faster since the symbolic factorization is reused.
    # Floating point errors accumulate a bit and the system matrix
    # is not necessarily perfectly symmetric but only up to round-off errors.
    # We wrap it here to avoid issues with the factorization.
    @.. M_h = h
    scale_by_mass_matrix!(M_h, D)
    inv3 = 1 / 3
    @.. M_h3_3 = inv3 * h^3
    scale_by_mass_matrix!(M_h3_3, D)
    system_matrix = Symmetric(Diagonal(M_h)
                                + Dmat' * Diagonal(M_h3_3) * Dmat)
    if issparse(system_matrix)
        (; factorization) = parameters
        cholesky!(factorization, system_matrix; check = false)
        if issuccess(factorization)
            scale_by_mass_matrix!(tmp, D)
            dv .= factorization \ tmp
        else
            # The factorization may fail if the time step is too large
            # and h becomes negative.
            fill!(dv, Inf)
        end
    else
        factorization = cholesky!(system_matrix)
        scale_by_mass_matrix!(tmp, D)
        ldiv!(dv, factorization, tmp)
    end

    return nothing
end

function rhs_sgn_flat_upwind!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D` as well as the
    # SBP upwind operator in sparse matrix form `Dmat_minus`
    (; g, Dmat_minus) = parameters
    D_upwind = parameters.D
    D = D_upwind.central

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # Compute all derivatives required below
    (; h_x, v_x, v_x_upwind, h2_x, hv_x, v2_x,
       h2_v_vx_x, h_vx_x, p_x, tmp,
       M_h, M_h3_3) = parameters
    mul!(h_x, D, h)
    mul!(v_x, D, v)
    mul!(v_x_upwind, D_upwind.minus, v)
    @.. tmp = h^2
    mul!(h2_x, D, tmp)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)

    @.. tmp = h^2 * v * v_x
    mul!(h2_v_vx_x, D, tmp)
    @.. tmp = h * v_x
    mul!(h_vx_x, D, tmp)
    # p_+
    @.. tmp = 0.5 * h^2 * (h * v_x + h_x * v) * v_x_upwind
    mul!(p_x, D_upwind.plus, tmp)
    # p_0
    minv6 = -1 / 6
    @.. tmp = minv6 * (  h * h2_v_vx_x
                       + h^2 * v * h_vx_x)
    mul!(p_x, D, tmp, 1.0, 1.0)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + ... = 0
    #
    # Split form for energy conservation:
    @.. tmp = -(  g * h2_x - g * h * h_x
                + 0.5 * h * v2_x
                - 0.5 * v^2 * h_x
                + 0.5 * hv_x * v
                - 0.5 * h * v * v_x
                + p_x)
    # The code below is equivalent to
    #   dv .= (Diagonal(h) - Dmat_plus * Diagonal(1/3 .* h.^3) * Dmat_minus) \ tmp
    # but faster since the symbolic factorization is reused.
    # Floating point errors accumulate a bit and the system matrix
    # is not necessarily perfectly symmetric but only up to round-off errors.
    # We wrap it here to avoid issues with the factorization.
    @.. M_h = h
    scale_by_mass_matrix!(M_h, D)
    inv3 = 1 / 3
    @.. M_h3_3 = inv3 * h^3
    scale_by_mass_matrix!(M_h3_3, D)
    system_matrix = Symmetric(Diagonal(M_h)
                                + Dmat_minus' * Diagonal(M_h3_3) * Dmat_minus)
    (; factorization) = parameters
    cholesky!(factorization, system_matrix; check = false)
    if issuccess(factorization)
        scale_by_mass_matrix!(tmp, D)
        dv .= factorization \ tmp
    else
        # The factorization may fail if the time step is too large
        # and h becomes negative.
        fill!(dv, Inf)
    end

    return nothing
end

function setup(::typeof(rhs_serre_green_naghdi_flat!),
               h_func, v_func, b_func::typeof(zero);
               g, λ, D)
    # Compute the initial state
    x = grid(D)
    h = h_func.(x)
    v = v_func.(x)
    u0 = ArrayPartition(h, v)

    if D isa PeriodicUpwindOperators
        # create temporary storage
        h_x = zero(h)
        v_x = zero(h)
        v_x_upwind = zero(h)
        h2_x = zero(h)
        hv_x = zero(h)
        v2_x = zero(h)
        h2_v_vx_x = zero(h)
        h_vx_x = zero(h)
        p_x = zero(h)
        tmp = zero(h)
        M_h = zero(h)
        M_h3_3 = zero(h)

        Dmat_minus = sparse(D.minus)

        # Floating point errors accumulate a bit and the system matrix
        # is not necessarily perfectly symmetric but only up to round-off errors.
        # We wrap it here to avoid issues with the factorization.
        @.. M_h = h
        scale_by_mass_matrix!(M_h, D)
        @.. M_h3_3 = (1/3) * h^3
        scale_by_mass_matrix!(M_h3_3, D)
        system_matrix = Symmetric(Diagonal(M_h)
                                    + Dmat_minus' * Diagonal(M_h3_3) * Dmat_minus)
        factorization = cholesky(system_matrix)

        parameters = (; h_x, v_x, v_x_upwind, h2_x, hv_x, v2_x,
                        h2_v_vx_x, h_vx_x, p_x, tmp,
                        M_h, M_h3_3,
                        g, D, Dmat_minus, factorization)
    else
        # create temporary storage
        h_x = zero(h)
        v_x = zero(h)
        h2_x = zero(h)
        hv_x = zero(h)
        v2_x = zero(h)
        h2_v_vx_x = zero(h)
        h_vx_x = zero(h)
        p_x = zero(h)
        tmp = zero(h)
        M_h = zero(h)
        M_h3_3 = zero(h)

        if D isa FourierDerivativeOperator
            Dmat = Matrix(D)

            parameters = (; h_x, v_x, h2_x, hv_x, v2_x,
                            h2_v_vx_x, h_vx_x, p_x, tmp,
                            M_h, M_h3_3,
                            g, D, Dmat)
        else
            Dmat = sparse(D)

            # Floating point errors accumulate a bit and the system matrix
            # is not necessarily perfectly symmetric but only up to round-off errors.
            # We wrap it here to avoid issues with the factorization.
            @.. M_h = h
            scale_by_mass_matrix!(M_h, D)
            @.. M_h3_3 = (1/3) * h^3
            scale_by_mass_matrix!(M_h3_3, D)
            system_matrix = Symmetric(Diagonal(M_h)
                                        + Dmat' * Diagonal(M_h3_3) * Dmat)
            factorization = cholesky(system_matrix)

            parameters = (; h_x, v_x, h2_x, hv_x, v2_x,
                            h2_v_vx_x, h_vx_x, p_x, tmp,
                            M_h, M_h3_3,
                            g, D, Dmat, factorization)
        end

    end

    return u0, parameters
end

function energy(::typeof(rhs_serre_green_naghdi_flat!),
                u, parameters)
    # unpack physical parameters and SBP operator `D`
    (; g, D, v_x) = parameters

    # `u` is an `ArrayPartition`. It collects the individual arrays for
    # the water height `h` and the velocity `v`.
    h, v = u.x

    # 1/2 g h^2 + 1/2 h v^2 + 1/6 h^3 v_x^2
    if D isa PeriodicUpwindOperators
        mul!(v_x, D.minus, v)
    else
        mul!(v_x, D, v)
    end

    E = parameters.tmp
    @.. E = 1/2 * g * h^2 + 1/2 * h * v^2 + 1/6 * h^3 * v_x^2

    return integrate(E, D)
end

function energy_rate(::typeof(rhs_serre_green_naghdi_flat!),
                     du, u, parameters)
    # unpack physical parameters and SBP operator `D`
    (; g, D, v_x) = parameters

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # E = 1/2 g h^2 + 1/2 h v^2 + 1/6 h^3 v_x^2
    # E_t = (g h + 1/2 v^2 + 1/2 h^2 v_x^2) h_t
    #       + h v v_t + 1/3 h^3 v_x v_xt

    # E_h = g h + 1/2 v^2 + 1/2 h^2 v_x^2
    if D isa PeriodicUpwindOperators
        mul!(v_x, D.minus, v)
    else
        mul!(v_x, D, v)
    end
    E_h = @.. g * h + 1/2 * v^2 + 1/2 * h^2 * v_x^2

    # h v v_t + 1/3 h^3 v_x v_xt
    if D isa PeriodicUpwindOperators
        v_xt = D.minus * dv
    else
        v_xt = D * dv
    end

    uEu = @.. -(E_h * dh + h * v * dv + 1/3 * h^3 * v_x * v_xt)

    return integrate(uEu, D)
end


#####################################################################
# The spatial semidiscretization = ODE RHS
# using nonconservative variables (h, v)
# for the classical Serre-Green-Naghdi equations
# with flat bathymetry
# using artificial viscosity
# Viscosity to be set as a power of domainlength/Nnodes
# perhaps not the best to compare to but the most general
function rhs_serre_green_naghdi_flat_visc!(du, u, parameters, t)
    D = parameters.D
    if D isa PeriodicUpwindOperators
        rhs_sgn_flat_upwind_visc!(du, u, parameters, t)
    else
        rhs_sgn_flat_central_visc!(du, u, parameters, t)
    end
    return nothing
end

function rhs_sgn_flat_central_visc!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D` as well as the
    # SBP operator in sparse matrix form `Dmat`
    (; g, mu, D, Dmat) = parameters

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # Compute all derivatives required below
    (; h_x, v_x, h2_x, hv_x, v2_x,
       h2_v_vx_x, h_vx_x, p_x, tmp, fv_x,
       M_h, M_h3_3) = parameters
    mul!(h_x, D, h)
    mul!(v_x, D, v)
    @.. tmp = h^2
    mul!(h2_x, D, tmp)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)

    @.. tmp = h^2 * v * v_x
    mul!(h2_v_vx_x, D, tmp)
    @.. tmp = h * v_x
    mul!(h_vx_x, D, tmp)
    inv6 = 1 / 6
    @.. tmp = (  0.5 * h^2 * (h * v_x + h_x * v) * v_x
               - inv6 * h * h2_v_vx_x
               - inv6 * h^2 * v * h_vx_x)
    mul!(p_x, D, tmp)

    @.. tmp = h * mu * v_x
    mul!(fv_x, D, tmp)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + ... = 0
    #
    # Split form for energy conservation:
    @.. tmp = -(  g * h2_x - g * h * h_x
                + 0.5 * h * v2_x
                - 0.5 * v^2 * h_x
                + 0.5 * hv_x * v
                - 0.5 * h * v * v_x
                + p_x
                - fv_x)

    # The code below is equivalent to
    #   dv .= (Diagonal(h) - Dmat * Diagonal(1/3 .* h.^3) * Dmat) \ tmp
    # but faster since the symbolic factorization is reused.
    # Floating point errors accumulate a bit and the system matrix
    # is not necessarily perfectly symmetric but only up to round-off errors.
    # We wrap it here to avoid issues with the factorization.
    @.. M_h = h
    scale_by_mass_matrix!(M_h, D)
    inv3 = 1 / 3
    @.. M_h3_3 = inv3 * h^3
    scale_by_mass_matrix!(M_h3_3, D)
    system_matrix = Symmetric(Diagonal(M_h)
                                + Dmat' * Diagonal(M_h3_3) * Dmat)
    if issparse(system_matrix)
        (; factorization) = parameters
        cholesky!(factorization, system_matrix; check = false)
        if issuccess(factorization)
            scale_by_mass_matrix!(tmp, D)
            dv .= factorization \ tmp
        else
            # The factorization may fail if the time step is too large
            # and h becomes negative.
            fill!(dv, Inf)
        end
    else
        factorization = cholesky!(system_matrix)
        scale_by_mass_matrix!(tmp, D)
        ldiv!(dv, factorization, tmp)
    end

    return nothing
end

function rhs_sgn_flat_upwind_visc!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D` as well as the
    # SBP operator in sparse matrix form `Dmat`
    (; g, mu, Dmat_minus) = parameters
    D_upwind = parameters.D
    D = D_upwind.central

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # Compute all derivatives required below
    (; h_x, v_x, v_x_upwind, h2_x, hv_x, v2_x,
       h2_v_vx_x, h_vx_x, p_x, tmp, fv_x,
       M_h, M_h3_3) = parameters
    mul!(h_x, D, h)
    mul!(v_x, D, v)
    mul!(v_x_upwind, D_upwind.minus, v)
    @.. tmp = h^2
    mul!(h2_x, D, tmp)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)

    @.. tmp = h^2 * v * v_x
    mul!(h2_v_vx_x, D, tmp)
    @.. tmp = h * v_x
    mul!(h_vx_x, D, tmp)
    # p_+
    @.. tmp = 0.5 * h^2 * (h * v_x + h_x * v) * v_x_upwind
    mul!(p_x, D_upwind.plus, tmp)
    # p_0
    minv6 = -1 / 6
    @.. tmp = minv6 * h * (h2_v_vx_x + h * v * h_vx_x)
    mul!(p_x, D, tmp, 1.0, 1.0)

    #@.. tmp = h * mu * v_x_upwind
    #below with reference h in mu
    @.. tmp = mu * v_x_upwind
    mul!(fv_x, D_upwind.plus, tmp)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + ... = 0
    #
    # Split form for energy conservation:
    @.. tmp = -(  g * h2_x - g * h * h_x
                + 0.5 * h * v2_x
                - 0.5 * v^2 * h_x
                + 0.5 * hv_x * v
                - 0.5 * h * v * v_x
                + p_x
                - fv_x)
    # The code below is equivalent to
    #   dv .= (Diagonal(h) - Dmat_plus * Diagonal(1/3 .* h.^3) * Dmat_minus) \ tmp
    # but faster since the symbolic factorization is reused.
    # Floating point errors accumulate a bit and the system matrix
    # is not necessarily perfectly symmetric but only up to round-off errors.
    # We wrap it here to avoid issues with the factorization.
    @.. M_h = h
    scale_by_mass_matrix!(M_h, D)
    inv3 = 1 / 3
    @.. M_h3_3 = inv3 * h^3
    scale_by_mass_matrix!(M_h3_3, D)
    system_matrix = Symmetric(Diagonal(M_h)
                                + Dmat_minus' * Diagonal(M_h3_3) * Dmat_minus)
    (; factorization) = parameters
    cholesky!(factorization, system_matrix; check = false)
    if issuccess(factorization)
        scale_by_mass_matrix!(tmp, D)
        dv .= factorization \ tmp
    else
        # The factorization may fail if the time step is too large
        # and h becomes negative.
        fill!(dv, Inf)
    end

    return nothing
end

function setup(::typeof(rhs_serre_green_naghdi_flat_visc!),
               h_func, v_func, b_func::typeof(zero);
               g, mu, λ, D)
    u0, parameters = setup(rhs_serre_green_naghdi_flat!,
                           h_func, v_func, b_func;
                           g, λ, D)
    fv_x = zero(parameters.tmp)

    parameters = (; parameters..., fv_x, mu)

    return u0, parameters
end

function energy(::typeof(rhs_serre_green_naghdi_flat_visc!),
                u, parameters)
    energy(rhs_serre_green_naghdi_flat!, u, parameters)
end

function energy_rate(::typeof(rhs_serre_green_naghdi_flat_visc!),
                     du, u, parameters)
    energy_rate(rhs_serre_green_naghdi_flat!, du, u, parameters)
end


#####################################################################
# The spatial semidiscretization = ODE RHS
# using nonconservative variables (h, v)
# for the classical Serre-Green-Naghdi equations
# with mild-slope approximation
function rhs_serre_green_naghdi_mild!(du, u, parameters, t)
    D = parameters.D
    if D isa PeriodicUpwindOperators
        rhs_sgn_mild_upwind!(du, u, parameters, t)
    else
        rhs_sgn_mild_central!(du, u, parameters, t)
    end
    return nothing
end

function rhs_sgn_mild_central!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D` as well as the
    # SBP operator in sparse matrix form `Dmat`
    (; g, D, Dmat) = parameters

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # Compute all derivatives required below
    (; h_x, v_x, h_hpb_x, b, b_x, hv_x, v2_x,
       h2_v_vx_x, h_vx_x, p_h, p_x, tmp,
       M_h_p_h_bx2, M_h3_3, M_h2_bx) = parameters
    mul!(h_x, D, h)
    mul!(v_x, D, v)
    @.. tmp = h * (h + b)
    mul!(h_hpb_x, D, tmp)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)

    @.. tmp = h^2 * v * v_x
    mul!(h2_v_vx_x, D, tmp)
    @.. tmp = h * v_x
    mul!(h_vx_x, D, tmp)
    inv6 = 1 / 6
    @.. p_h = (  0.5 * h * (h * v_x + h_x * v) * v_x
               - inv6 * h2_v_vx_x
               - inv6 * h * v * h_vx_x)
    @.. tmp = h * b_x * v^2
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * p_x
    @.. tmp = b_x * v
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * h * v * p_x
    @.. p_h = p_h - 0.25 * (h_x * v + h * v_x) * b_x * v
    @.. tmp = p_h * h
    mul!(p_x, D, tmp)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + ... = 0
    #
    # Split form for energy conservation:
    @.. tmp = -(  g * h_hpb_x - g * (h + b) * h_x
                + 0.5 * h * v2_x
                - 0.5 * v^2 * h_x
                + 0.5 * hv_x * v
                - 0.5 * h * v * v_x
                + p_x
                + 1.5 * p_h * b_x)
    # The code below is equivalent to
    #   dv .= (Diagonal(h .+ 0.75 .* h .* b_x.^2) - Dmat * (Diagonal(1/3 .* h.^3) * Dmat - Diagonal(0.5 .* h.^2 .* b_x)) - Diagonal(0.5 .* h.^2 .* b_x) * Dmat) \ tmp
    # but faster since the symbolic factorization is reused.
    # Floating point errors accumulate a bit and the system matrix
    # is not necessarily perfectly symmetric but only up to round-off errors.
    # We wrap it here to avoid issues with the factorization.
    @.. M_h_p_h_bx2 = h + 0.75 * h * b_x^2
    scale_by_mass_matrix!(M_h_p_h_bx2, D)
    inv3 = 1 / 3
    @.. M_h3_3 = inv3 * h^3
    scale_by_mass_matrix!(M_h3_3, D)
    @.. M_h2_bx = 0.5 * h^2 * b_x
    scale_by_mass_matrix!(M_h2_bx, D)
    system_matrix = Symmetric(Diagonal(M_h_p_h_bx2)
                            + Dmat' * (Diagonal(M_h3_3) * Dmat
                                        - Diagonal(M_h2_bx))
                            - Diagonal(M_h2_bx) * Dmat)
    if issparse(system_matrix)
        (; factorization) = parameters
        cholesky!(factorization, system_matrix; check = false)
        if issuccess(factorization)
            scale_by_mass_matrix!(tmp, D)
            dv .= factorization \ tmp
        else
            # The factorization may fail if the time step is too large
            # and h becomes negative.
            fill!(dv, Inf)
        end
    else
        factorization = cholesky!(system_matrix)
        scale_by_mass_matrix!(tmp, D)
        ldiv!(dv, factorization, tmp)
    end

    return nothing
end

function rhs_sgn_mild_upwind!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D` as well as the
    # SBP operator in sparse matrix form `Dmat`
    (; g, Dmat_minus) = parameters
    D_upwind = parameters.D
    D = D_upwind.central

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # Compute all derivatives required below
    (; h_x, v_x, v_x_upwind, h_hpb_x, b, b_x, hv_x, v2_x,
       h2_v_vx_x, h_vx_x, p_h, p_0, p_x, tmp,
       M_h_p_h_bx2, M_h3_3, M_h2_bx) = parameters
    mul!(h_x, D, h)
    mul!(v_x, D, v)
    mul!(v_x_upwind, D_upwind.minus, v)
    @.. tmp = h * (h + b)
    mul!(h_hpb_x, D, tmp)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)

    @.. tmp = h^2 * v * v_x
    mul!(h2_v_vx_x, D, tmp)
    @.. tmp = h * v_x
    mul!(h_vx_x, D, tmp)
    # p_0
    minv6 = -1 / 6
    @.. p_h = minv6 * (  h2_v_vx_x
                       + h * v * h_vx_x)
    @.. tmp = h * b_x * v^2
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * p_x
    @.. tmp = b_x * v
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * h * v * p_x
    @.. p_0 = p_h * h
    mul!(p_x, D, p_0)
    # p_+
    @.. tmp = (  0.5 * h * (h * v_x + h_x * v) * v_x_upwind
               - 0.25 * (h_x * v + h * v_x) * b_x * v)
    @.. p_h = p_h + tmp
    @.. tmp = tmp * h
    mul!(p_x, D_upwind.plus, tmp, 1.0, 1.0)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + ... = 0
    #
    # Split form for energy conservation:
    @.. tmp = -(  g * h_hpb_x - g * (h + b) * h_x
                + 0.5 * h * v2_x
                - 0.5 * v^2 * h_x
                + 0.5 * hv_x * v
                - 0.5 * h * v * v_x
                + p_x
                + 1.5 * p_h * b_x)
    # The code below is equivalent to
    #   dv .= (Diagonal(h .+ 0.75 .* h .* b_x.^2) - Dmat * (Diagonal(1/3 .* h.^3) * Dmat - Diagonal(0.5 .* h.^2 .* b_x)) - Diagonal(0.5 .* h.^2 .* b_x) * Dmat) \ tmp
    # but faster since the symbolic factorization is reused.
    # Floating point errors accumulate a bit and the system matrix
    # is not necessarily perfectly symmetric but only up to round-off errors.
    # We wrap it here to avoid issues with the factorization.
    @.. M_h_p_h_bx2 = h + 0.75 * h * b_x^2
    scale_by_mass_matrix!(M_h_p_h_bx2, D)
    inv3 = 1 / 3
    @.. M_h3_3 = inv3 * h^3
    scale_by_mass_matrix!(M_h3_3, D)
    @.. M_h2_bx = 0.5 * h^2 * b_x
    scale_by_mass_matrix!(M_h2_bx, D)
    system_matrix = Symmetric(Diagonal(M_h_p_h_bx2)
                            + Dmat_minus' * (Diagonal(M_h3_3) * Dmat_minus
                                        - Diagonal(M_h2_bx))
                            - Diagonal(M_h2_bx) * Dmat_minus)
    if issparse(system_matrix)
        (; factorization) = parameters
        cholesky!(factorization, system_matrix; check = false)
        if issuccess(factorization)
            scale_by_mass_matrix!(tmp, D)
            dv .= factorization \ tmp
        else
            # The factorization may fail if the time step is too large
            # and h becomes negative.
            fill!(dv, Inf)
        end
    else
        factorization = cholesky!(system_matrix)
        scale_by_mass_matrix!(tmp, D)
        ldiv!(dv, factorization, tmp)
    end

    return nothing
end

function setup(::typeof(rhs_serre_green_naghdi_mild!),
               h_func, v_func, b_func;
               g, λ, D)
    # Compute the initial state
    x = grid(D)
    h = h_func.(x)
    v = v_func.(x)
    u0 = ArrayPartition(h, v)

    if D isa PeriodicUpwindOperators
        # Compute the bottom topography and its derivative
        b = b_func.(x)
        b_x = D.central * b

        # create temporary storage
        h_x = zero(h)
        v_x = zero(h)
        v_x_upwind = zero(h)
        h_hpb_x = zero(h)
        hv_x = zero(h)
        v2_x = zero(h)
        h2_v_vx_x = zero(h)
        h_vx_x = zero(h)
        p_h = zero(h)
        p_0 = zero(h)
        p_x = zero(h)
        tmp = zero(h)
        M_h_p_h_bx2 = zero(h)
        M_h3_3 = zero(h)
        M_h2_bx = zero(h)

        Dmat_minus = sparse(D.minus)

        # Floating point errors accumulate a bit and the system matrix
        # is not necessarily perfectly symmetric but only up to round-off errors.
        # We wrap it here to avoid issues with the factorization.
        @.. M_h_p_h_bx2 = h + 0.75 * h * b_x^2
        scale_by_mass_matrix!(M_h_p_h_bx2, D)
        inv3 = 1 / 3
        @.. M_h3_3 = inv3 * h^3
        scale_by_mass_matrix!(M_h3_3, D)
        @.. M_h2_bx = 0.5 * h^2 * b_x
        scale_by_mass_matrix!(M_h2_bx, D)
        system_matrix = Symmetric(Diagonal(M_h_p_h_bx2)
                                + Dmat_minus' * (Diagonal(M_h3_3) * Dmat_minus
                                            - Diagonal(M_h2_bx))
                                - Diagonal(M_h2_bx) * Dmat_minus)

        factorization = cholesky(system_matrix)

        parameters = (; h_x, v_x, v_x_upwind, h_hpb_x, b, b_x, hv_x, v2_x,
                        h2_v_vx_x, h_vx_x, p_h, p_0, p_x, tmp,
                        M_h_p_h_bx2, M_h3_3, M_h2_bx,
                        g, D, Dmat_minus, factorization)
    else
        # Compute the bottom topography and its derivative
        b = b_func.(x)
        b_x = D * b

        # create temporary storage
        h_x = zero(h)
        v_x = zero(h)
        h_hpb_x = zero(h)
        hv_x = zero(h)
        v2_x = zero(h)
        h2_v_vx_x = zero(h)
        h_vx_x = zero(h)
        p_h = zero(h)
        p_x = zero(h)
        tmp = zero(h)
        M_h_p_h_bx2 = zero(h)
        M_h3_3 = zero(h)
        M_h2_bx = zero(h)

        if D isa FourierDerivativeOperator
            Dmat = Matrix(D)

            parameters = (; h_x, v_x, h_hpb_x, b, b_x, hv_x, v2_x,
                            h2_v_vx_x, h_vx_x, p_h, p_x, tmp,
                            M_h_p_h_bx2, M_h3_3, M_h2_bx,
                            g, D, Dmat)
        else
            Dmat = sparse(D)

            # Floating point errors accumulate a bit and the system matrix
            # is not necessarily perfectly symmetric but only up to round-off errors.
            # We wrap it here to avoid issues with the factorization.
            @.. M_h_p_h_bx2 = h + 0.75 * h * b_x^2
            scale_by_mass_matrix!(M_h_p_h_bx2, D)
            inv3 = 1 / 3
            @.. M_h3_3 = inv3 * h^3
            scale_by_mass_matrix!(M_h3_3, D)
            @.. M_h2_bx = 0.5 * h^2 * b_x
            scale_by_mass_matrix!(M_h2_bx, D)
            system_matrix = Symmetric(Diagonal(M_h_p_h_bx2)
                                    + Dmat' * (Diagonal(M_h3_3) * Dmat
                                                - Diagonal(M_h2_bx))
                                    - Diagonal(M_h2_bx) * Dmat)

            factorization = cholesky(system_matrix)

            parameters = (; h_x, v_x, h_hpb_x, b, b_x, hv_x, v2_x,
                            h2_v_vx_x, h_vx_x, p_h, p_x, tmp,
                            M_h_p_h_bx2, M_h3_3, M_h2_bx,
                            g, D, Dmat, factorization)
        end
    end

    return u0, parameters
end

function energy(::typeof(rhs_serre_green_naghdi_mild!),
                u, parameters)
    # unpack physical parameters and SBP operator `D`
    (; g, D, b, b_x, v_x) = parameters

    # `u` is an `ArrayPartition`. It collects the individual arrays for
    # the water height `h` and the velocity `v`.
    h, v = u.x

    # 1/2 g (h + b)^2 + 1/2 h v^2 + 1/6 h w^2
    if D isa PeriodicUpwindOperators
        mul!(v_x, D.minus, v)
    else
        mul!(v_x, D, v)
    end
    E = parameters.tmp
    @.. E = 0.5 * g * (h + b)^2 + 0.5 * h * v^2 + 1/6 * h * (-h * v_x + 1.5 * v * b_x)^2

    return integrate(E, D)
end


#####################################################################
# The spatial semidiscretization = ODE RHS
# using nonconservative variables (h, v)
# for the classical Serre-Green-Naghdi equations
# with mild-slope approximation
# using artificial viscosity
function rhs_serre_green_naghdi_mild_visc!(du, u, parameters, t)
    D = parameters.D
    if D isa PeriodicUpwindOperators
        rhs_sgn_mild_upwind_visc!(du, u, parameters, t)
    else
        rhs_serre_green_naghdi_mild_central_visc!(du, u, parameters, t)
    end
    return nothing
end

function rhs_serre_green_naghdi_mild_central_visc!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D` as well as the
    # SBP operator in sparse matrix form `Dmat`
    (; g, mu, D, Dmat) = parameters

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # Compute all derivatives required below
    (; h_x, v_x, h_hpb_x, b, b_x, hv_x, v2_x,
       h2_v_vx_x, h_vx_x, p_h, p_x, tmp, fv_x,
       M_h_p_h_bx2, M_h3_3, M_h2_bx) = parameters
    mul!(h_x, D, h)
    mul!(v_x, D, v)
    @.. tmp = h * (h + b)
    mul!(h_hpb_x, D, tmp)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)

    @.. tmp = h^2 * v * v_x
    mul!(h2_v_vx_x, D, tmp)
    @.. tmp = h * v_x
    mul!(h_vx_x, D, tmp)
    inv6 = 1 / 6
    @.. p_h = (  00.5 * h * (h * v_x + h_x * v) * v_x
               - inv6 * h2_v_vx_x
               - inv6 * h * v * h_vx_x)
    @.. tmp = h * b_x * v^2
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * p_x
    @.. tmp = b_x * v
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * h * v * p_x
    @.. p_h = p_h - 0.25 * (h_x * v + h * v_x) * b_x * v
    @.. tmp = p_h * h
    mul!(p_x, D, tmp)

    @.. tmp = h * mu * v_x
    mul!(fv_x, D, tmp)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + ... = 0
    #
    # Split form for energy conservation:
    @.. tmp = -(  g * h_hpb_x - g * (h + b) * h_x
                + 0.5 * h * v2_x
                - 0.5 * v^2 * h_x
                + 0.5 * hv_x * v
                - 0.5 * h * v * v_x
                + p_x
                + 1.5 * p_h * b_x
                - fv_x)
    # The code below is equivalent to
    # dv .= (Diagonal(h .+ 0.75 .* h .* b_x.^2) - Dmat * (Diagonal(1/3 .* h.^3) * Dmat - Diagonal(0.5 .* h.^2 .* b_x)) - Diagonal(0.5 .* h.^2 .* b_x) * Dmat) \ tmp
    # but faster since the symbolic factorization is reused.
    # Floating point errors accumulate a bit and the system matrix
    # is not necessarily perfectly symmetric but only up to round-off errors.
    # We wrap it here to avoid issues with the factorization.
    @.. M_h_p_h_bx2 = h + 0.75 * h * b_x^2
    scale_by_mass_matrix!(M_h_p_h_bx2, D)
    inv3 = 1 / 3
    @.. M_h3_3 = inv3 * h^3
    scale_by_mass_matrix!(M_h3_3, D)
    @.. M_h2_bx = 0.5 * h^2 * b_x
    scale_by_mass_matrix!(M_h2_bx, D)
    system_matrix = Symmetric(Diagonal(M_h_p_h_bx2)
                            + Dmat' * (Diagonal(M_h3_3) * Dmat
                                        - Diagonal(M_h2_bx))
                            - Diagonal(M_h2_bx) * Dmat)
    if issparse(system_matrix)
        (; factorization) = parameters
        cholesky!(factorization, system_matrix; check = false)
        if issuccess(factorization)
            scale_by_mass_matrix!(tmp, D)
            dv .= factorization \ tmp
        else
            # The factorization may fail if the time step is too large
            # and h becomes negative.
            fill!(dv, Inf)
        end
    else
        factorization = cholesky!(system_matrix)
        scale_by_mass_matrix!(tmp, D)
        ldiv!(dv, factorization, tmp)
    end

    return nothing
end

function rhs_sgn_mild_upwind_visc!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D` as well as the
    # SBP operator in sparse matrix form `Dmat`
    (; g, mu, Dmat_minus) = parameters
    D_upwind = parameters.D
    D = D_upwind.central

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # Compute all derivatives required below
    (; h_x, v_x, v_x_upwind, h_hpb_x, b, b_x, hv_x, v2_x,
       h2_v_vx_x, h_vx_x, p_h, p_0, p_x, tmp, fv_x,
       M_h_p_h_bx2, M_h3_3, M_h2_bx) = parameters
    mul!(h_x, D, h)
    mul!(v_x, D, v)
    mul!(v_x_upwind, D_upwind.minus, v)
    @.. tmp = h * (h + b)
    mul!(h_hpb_x, D, tmp)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)

    @.. tmp = h^2 * v * v_x
    mul!(h2_v_vx_x, D, tmp)
    @.. tmp = h * v_x
    mul!(h_vx_x, D, tmp)
    # p_0
    minv6 = -1 / 6
    @.. p_h = minv6 * (  h2_v_vx_x
                       + h * v * h_vx_x)
    @.. tmp = h * b_x * v^2
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * p_x
    @.. tmp = b_x * v
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * h * v * p_x
    @.. p_0 = p_h * h
    mul!(p_x, D, p_0)
    # p_+
    @.. tmp = (  0.5 * h * (h * v_x + h_x * v) * v_x_upwind
               - 0.25 * (h_x * v + h * v_x) * b_x * v)
    @.. p_h = p_h + tmp
    @.. tmp = tmp * h
    mul!(p_x, D_upwind.plus, tmp, 1.0, 1.0)

    @.. tmp = h * mu * v_x
    mul!(fv_x, D, tmp)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + ... = 0
    #
    # Split form for energy conservation:
    @.. tmp = -(  g * h_hpb_x - g * (h + b) * h_x
                + 0.5 * h * v2_x
                - 0.5 * v^2 * h_x
                + 0.5 * hv_x * v
                - 0.5 * h * v * v_x
                + p_x
                + 1.5 * p_h * b_x
                - fv_x)
    # The code below is equivalent to
    # dv .= (Diagonal(h .+ 0.75 .* h .* b_x.^2) - Dmat * (Diagonal(1/3 .* h.^3) * Dmat - Diagonal(0.5 .* h.^2 .* b_x)) - Diagonal(0.5 .* h.^2 .* b_x) * Dmat) \ tmp
    # but faster since the symbolic factorization is reused.
    # Floating point errors accumulate a bit and the system matrix
    # is not necessarily perfectly symmetric but only up to round-off errors.
    # We wrap it here to avoid issues with the factorization.
    @.. M_h_p_h_bx2 = h + 0.75 * h * b_x^2
    scale_by_mass_matrix!(M_h_p_h_bx2, D)
    inv3 = 1 / 3
    @.. M_h3_3 = inv3 * h^3
    scale_by_mass_matrix!(M_h3_3, D)
    @.. M_h2_bx = 0.5 * h^2 * b_x
    scale_by_mass_matrix!(M_h2_bx, D)
    system_matrix = Symmetric(Diagonal(M_h_p_h_bx2)
                            + Dmat_minus' * (Diagonal(M_h3_3) * Dmat_minus
                                        - Diagonal(M_h2_bx))
                            - Diagonal(M_h2_bx) * Dmat_minus)
    if issparse(system_matrix)
        (; factorization) = parameters
        cholesky!(factorization, system_matrix; check = false)
        if issuccess(factorization)
            scale_by_mass_matrix!(tmp, D)
            dv .= factorization \ tmp
        else
            # The factorization may fail if the time step is too large
            # and h becomes negative.
            fill!(dv, Inf)
        end
    else
        factorization = cholesky!(system_matrix)
        scale_by_mass_matrix!(tmp, D)
        ldiv!(dv, factorization, tmp)
    end

    return nothing
end

function setup(::typeof(rhs_serre_green_naghdi_mild_visc!),
               h_func, v_func, b_func;
               g, mu, λ, D)
    u0, parameters = setup(rhs_serre_green_naghdi_mild!,
                           h_func, v_func, b_func;
                           g, λ, D)
    fv_x = zero(parameters.tmp)

    parameters = (; parameters..., fv_x, mu)

    return u0, parameters
end

function energy(::typeof(rhs_serre_green_naghdi_mild_visc!),
                u, parameters)
    energy(rhs_serre_green_naghdi_mild!, u, parameters)
end

function energy_rate(::typeof(rhs_serre_green_naghdi_mild_visc!),
                     du, u, parameters)
    energy_rate(rhs_serre_green_naghdi_mild!, du, u, parameters)
end


#####################################################################
# The spatial semidiscretization = ODE RHS
# using nonconservative variables (h, v)
# for the classical Serre-Green-Naghdi equations
# without mild-slope approximation
function rhs_serre_green_naghdi_full!(du, u, parameters, t)
    D = parameters.D
    if D isa PeriodicUpwindOperators
        rhs_sgn_full_upwind!(du, u, parameters, t)
    else
        rhs_sgn_full_central!(du, u, parameters, t)
    end
    return nothing
end

function rhs_sgn_full_central!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D` as well as the
    # SBP operator in sparse matrix form `Dmat`
    (; g, D, Dmat) = parameters

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # Compute all derivatives required below
    (; h_x, v_x, h_hpb_x, b, b_x, hv_x, v2_x,
       h2_v_vx_x, h_vx_x, p_h, p_x, ψ, tmp,
       M_h_p_h_bx2, M_h3_3, M_h2_bx) = parameters
    mul!(h_x, D, h)
    mul!(v_x, D, v)
    @.. tmp = h * (h + b)
    mul!(h_hpb_x, D, tmp)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)

    @.. tmp = h^2 * v * v_x
    mul!(h2_v_vx_x, D, tmp)
    @.. tmp = h * v_x
    mul!(h_vx_x, D, tmp)
    inv6 = 1 / 6
    @.. p_h = (  0.5 * h * (h * v_x + h_x * v) * v_x
               - inv6 * h2_v_vx_x
               - inv6 * h * v * h_vx_x)
    @.. tmp = h * b_x * v^2
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * p_x
    @.. ψ = 0.125 * p_x
    @.. tmp = b_x * v
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * h * v * p_x
    @.. ψ += 0.125 * h * v * p_x
    @.. p_h = p_h - 0.25 * (h_x * v + h * v_x) * b_x * v
    @.. ψ = ψ - 0.125 * (h_x * v + h * v_x) * b_x * v
    @.. tmp = p_h * h
    mul!(p_x, D, tmp)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + ... = 0
    #
    # Split form for energy conservation:
    @.. tmp = -(  g * h_hpb_x - g * (h + b) * h_x
                + 0.5 * h * v2_x
                - 0.5 * v^2 * h_x
                + 0.5 * hv_x * v
                - 0.5 * h * v * v_x
                + p_x
                + 1.5 * p_h * b_x
                + ψ * b_x)
    # The code below is equivalent to
    # dv .= (Diagonal(h .+ h .* b_x.^2) - Dmat * (Diagonal(1/3 .* h.^3) * Dmat - Diagonal(0.5 .* h.^2 .* b_x)) - Diagonal(0.5 .* h.^2 .* b_x) * Dmat) \ tmp
    # but faster since the symbolic factorization is reused.
    # Floating point errors accumulate a bit and the system matrix
    # is not necessarily perfectly symmetric but only up to round-off errors.
    # We wrap it here to avoid issues with the factorization.
    @.. M_h_p_h_bx2 = h + h * b_x^2
    scale_by_mass_matrix!(M_h_p_h_bx2, D)
    inv3 = 1 / 3
    @.. M_h3_3 = inv3 * h^3
    scale_by_mass_matrix!(M_h3_3, D)
    @.. M_h2_bx = 0.5 * h^2 * b_x
    scale_by_mass_matrix!(M_h2_bx, D)
    system_matrix = Symmetric(Diagonal(M_h_p_h_bx2)
                            + Dmat' * (Diagonal(M_h3_3) * Dmat
                                        - Diagonal(M_h2_bx))
                            - Diagonal(M_h2_bx) * Dmat)
    if issparse(system_matrix)
        (; factorization) = parameters
        cholesky!(factorization, system_matrix; check = false)
        if issuccess(factorization)
            scale_by_mass_matrix!(tmp, D)
            dv .= factorization \ tmp
        else
            # The factorization may fail if the time step is too large
            # and h becomes negative.
            fill!(dv, Inf)
        end
    else
        factorization = cholesky!(system_matrix)
        scale_by_mass_matrix!(tmp, D)
        ldiv!(dv, factorization, tmp)
    end

    return nothing
end

function rhs_sgn_full_upwind!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D` as well as the
    # SBP operator in sparse matrix form `Dmat`
    (; g, Dmat_minus) = parameters
    D_upwind = parameters.D
    D = D_upwind.central

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # Compute all derivatives required below
    (; h_x, v_x, v_x_upwind, h_hpb_x, b, b_x, hv_x, v2_x,
       h2_v_vx_x, h_vx_x, p_h, p_0, p_x, ψ, tmp,
       M_h_p_h_bx2, M_h3_3, M_h2_bx) = parameters
    mul!(h_x, D, h)
    mul!(v_x, D, v)
    mul!(v_x_upwind, D_upwind.minus, v)
    @.. tmp = h * (h + b)
    mul!(h_hpb_x, D, tmp)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)

    @.. tmp = h^2 * v * v_x
    mul!(h2_v_vx_x, D, tmp)
    @.. tmp = h * v_x
    mul!(h_vx_x, D, tmp)
    # p_0
    minv6 = -1 / 6
    @.. p_h = minv6 * (  h2_v_vx_x
                       + h * v * h_vx_x)
    @.. tmp = h * b_x * v^2
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * p_x
    @.. ψ = 0.125 * p_x
    @.. tmp = b_x * v
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * h * v * p_x
    @.. ψ += 0.125 * h * v * p_x
    @.. p_0 = p_h * h
    mul!(p_x, D, p_0)
    # p_+
    @.. tmp = (  0.5 * h * (h * v_x + h_x * v) * v_x_upwind
               - 0.25 * (h_x * v + h * v_x) * b_x * v)
    @.. ψ = ψ - 0.125 * (h_x * v + h * v_x) * b_x * v
    @.. p_h = p_h + tmp
    @.. tmp = tmp * h
    mul!(p_x, D_upwind.plus, tmp, 1.0, 1.0)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + ... = 0
    #
    # Split form for energy conservation:
    @.. tmp = -(  g * h_hpb_x - g * (h + b) * h_x
                + 0.5 * h * v2_x
                - 0.5 * v^2 * h_x
                + 0.5 * hv_x * v
                - 0.5 * h * v * v_x
                + p_x
                + 1.5 * p_h * b_x
                + ψ * b_x)
    # The code below is equivalent to
    # dv .= (Diagonal(h .+ h .* b_x.^2) - Dmat * (Diagonal(1/3 .* h.^3) * Dmat - Diagonal(0.5 .* h.^2 .* b_x)) - Diagonal(0.5 .* h.^2 .* b_x) * Dmat) \ tmp
    # but faster since the symbolic factorization is reused.
    # Floating point errors accumulate a bit and the system matrix
    # is not necessarily perfectly symmetric but only up to round-off errors.
    # We wrap it here to avoid issues with the factorization.
    @.. M_h_p_h_bx2 = h + h * b_x^2
    scale_by_mass_matrix!(M_h_p_h_bx2, D)
    inv3 = 1 / 3
    @.. M_h3_3 = inv3 * h^3
    scale_by_mass_matrix!(M_h3_3, D)
    @.. M_h2_bx = 0.5 * h^2 * b_x
    scale_by_mass_matrix!(M_h2_bx, D)
    system_matrix = Symmetric(Diagonal(M_h_p_h_bx2)
                            + Dmat_minus' * (Diagonal(M_h3_3) * Dmat_minus
                                        - Diagonal(M_h2_bx))
                            - Diagonal(M_h2_bx) * Dmat_minus)
    if issparse(system_matrix)
        (; factorization) = parameters
        cholesky!(factorization, system_matrix; check = false)
        if issuccess(factorization)
            scale_by_mass_matrix!(tmp, D)
            dv .= factorization \ tmp
        else
            # The factorization may fail if the time step is too large
            # and h becomes negative.
            fill!(dv, Inf)
        end
    else
        factorization = cholesky!(system_matrix)
        scale_by_mass_matrix!(tmp, D)
        ldiv!(dv, factorization, tmp)
    end

    return nothing
end

function setup(::typeof(rhs_serre_green_naghdi_full!),
               h_func, v_func, b_func;
               g, λ, D)
    u0, parameters = setup(rhs_serre_green_naghdi_mild!,
                           h_func, v_func, b_func;
                           g, λ, D)
    ψ = zero(parameters.tmp)

    parameters = (; parameters..., ψ)

    return u0, parameters
end

function energy(::typeof(rhs_serre_green_naghdi_full!),
                u, parameters)
    # unpack physical parameters and SBP operator `D`
    (; g, D, b, b_x, v_x) = parameters

    # `u` is an `ArrayPartition`. It collects the individual arrays for
    # the water height `h` and the velocity `v`.
    h, v = u.x

    # 1/2 g (h + b)^2 + 1/2 h v^2 + 1/6 h w^2
    if D isa PeriodicUpwindOperators
        mul!(v_x, D.minus, v)
    else
        mul!(v_x, D, v)
    end
    E = parameters.tmp
    @.. E = 0.5 * g * (h + b)^2 + 0.5 * h * v^2 + 1/6 * h * (-h * v_x + 1.5 * v * b_x)^2 + 1/8 * h * (v * b_x)^2

    return integrate(E, D)
end


#####################################################################
# The spatial semidiscretization = ODE RHS
# using nonconservative variables (h, v)
# for the classical Serre-Green-Naghdi equations
# without mild-slope approximation
# using artificial viscosity
function rhs_serre_green_naghdi_full_visc!(du, u, parameters, t)
    D = parameters.D
    if D isa PeriodicUpwindOperators
        rhs_sgn_full_upwind_visc!(du, u, parameters, t)
    else
        rhs_sgn_full_central_visc!(du, u, parameters, t)
    end
    return nothing
end

function rhs_sgn_full_central_visc!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D` as well as the
    # SBP operator in sparse matrix form `Dmat`
    (; g, mu, D, Dmat) = parameters

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # Compute all derivatives required below
    (; h_x, v_x, h_hpb_x, b, b_x, hv_x, v2_x,
       h2_v_vx_x, h_vx_x, p_h, p_x, ψ, tmp, fv_x,
       M_h_p_h_bx2, M_h3_3, M_h2_bx) = parameters
    mul!(h_x, D, h)
    mul!(v_x, D, v)
    @.. tmp = h * (h + b)
    mul!(h_hpb_x, D, tmp)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)

    @.. tmp = h^2 * v * v_x
    mul!(h2_v_vx_x, D, tmp)
    @.. tmp = h * v_x
    mul!(h_vx_x, D, tmp)
    inv6 = 1 / 6
    @.. p_h = ( 0.5 * h * (h * v_x + h_x * v) * v_x
              - inv6 * h2_v_vx_x
              - inv6 * h * v * h_vx_x)
    @.. tmp = h * b_x * v^2
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * p_x
    @.. ψ = 0.125 * p_x
    @.. tmp = b_x * v
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * h * v * p_x
    @.. ψ += 0.125 * h * v * p_x
    @.. p_h = p_h - 0.25 * (h_x * v + h * v_x) * b_x * v
    @.. ψ = ψ - 0.125 * (h_x * v + h * v_x) * b_x * v
    @.. tmp = p_h * h
    mul!(p_x, D, tmp)

    @.. tmp = h * mu * v_x
    mul!(fv_x, D, tmp)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + ... = 0
    #
    # Split form for energy conservation:
    @.. tmp = -(  g * h_hpb_x - g * (h + b) * h_x
                + 0.5 * h * v2_x
                - 0.5 * v^2 * h_x
                + 0.5 * hv_x * v
                - 0.5 * h * v * v_x
                + p_x
                + 1.5 * p_h * b_x
                + ψ * b_x
                - fv_x)
    # The code below is equivalent to
    # dv .= (Diagonal(h .+ h .* b_x.^2) - Dmat * (Diagonal(1/3 .* h.^3) * Dmat - Diagonal(0.5 .* h.^2 .* b_x)) - Diagonal(0.5 .* h.^2 .* b_x) * Dmat) \ tmp
    # but faster since the symbolic factorization is reused.
    # Floating point errors accumulate a bit and the system matrix
    # is not necessarily perfectly symmetric but only up to round-off errors.
    # We wrap it here to avoid issues with the factorization.
    @.. M_h_p_h_bx2 = h + h * b_x^2
    scale_by_mass_matrix!(M_h_p_h_bx2, D)
    inv3 = 1 / 3
    @.. M_h3_3 = inv3 * h^3
    scale_by_mass_matrix!(M_h3_3, D)
    @.. M_h2_bx = 0.5 * h^2 * b_x
    scale_by_mass_matrix!(M_h2_bx, D)
    system_matrix = Symmetric(Diagonal(M_h_p_h_bx2)
                            + Dmat' * (Diagonal(M_h3_3) * Dmat
                                        - Diagonal(M_h2_bx))
                            - Diagonal(M_h2_bx) * Dmat)
    if issparse(system_matrix)
        (; factorization) = parameters
        cholesky!(factorization, system_matrix; check = false)
        if issuccess(factorization)
            scale_by_mass_matrix!(tmp, D)
            dv .= factorization \ tmp
        else
            # The factorization may fail if the time step is too large
            # and h becomes negative.
            fill!(dv, Inf)
        end
    else
        factorization = cholesky!(system_matrix)
        scale_by_mass_matrix!(tmp, D)
        ldiv!(dv, factorization, tmp)
    end

    return nothing
end

function rhs_sgn_full_upwind_visc!(du, u, parameters, t)
    # unpack physical parameters and SBP operator `D` as well as the
    # SBP operator in sparse matrix form `Dmat`
    (; g, mu, Dmat_minus) = parameters
    D_upwind = parameters.D
    D = D_upwind.central

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h` and the velocity `v`.
    h, v = u.x
    dh, dv = du.x

    # Compute all derivatives required below
    (; h_x, v_x, v_x_upwind, h_hpb_x, b, b_x, hv_x, v2_x,
       h2_v_vx_x, h_vx_x, p_h, p_0, p_x, ψ, tmp, fv_x,
       M_h_p_h_bx2, M_h3_3, M_h2_bx) = parameters
    mul!(h_x, D, h)
    mul!(v_x, D, v)
    mul!(v_x_upwind, D_upwind.minus, v)
    @.. tmp = h * (h + b)
    mul!(h_hpb_x, D, tmp)
    @.. tmp = h * v
    mul!(hv_x, D, tmp)
    @.. tmp = v^2
    mul!(v2_x, D, tmp)

    @.. tmp = h^2 * v * v_x
    mul!(h2_v_vx_x, D, tmp)
    @.. tmp = h * v_x
    mul!(h_vx_x, D, tmp)
    # p_0
    minv6 = -1 / 6
    @.. p_h = minv6 * (  h2_v_vx_x
                       + h * v * h_vx_x)
    @.. tmp = h * b_x * v^2
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * p_x
    @.. ψ = 0.125 * p_x
    @.. tmp = b_x * v
    mul!(p_x, D, tmp)
    @.. p_h += 0.25 * h * v * p_x
    @.. ψ += 0.125 * h * v * p_x
    @.. p_0 = p_h * h
    mul!(p_x, D, p_0)
    # p_+
    @.. tmp = (  0.5 * h * (h * v_x + h_x * v) * v_x_upwind
               - 0.25 * (h_x * v + h * v_x) * b_x * v)
    @.. ψ = ψ - 0.125 * (h_x * v + h * v_x) * b_x * v
    @.. p_h = p_h + tmp
    @.. tmp = tmp * h
    mul!(p_x, D_upwind.plus, tmp, 1.0, 1.0)

    @.. tmp = h * mu * v_x
    mul!(fv_x, D, tmp)

    # Plain: h_t + (h v)_x = 0
    #
    # Split form for energy conservation:
    # h_t + h_x v + h v_x = 0
    @.. dh = -(h_x * v + h * v_x)

    # Plain: h v_t + ... = 0
    #
    # Split form for energy conservation:
    @.. tmp = -(  g * h_hpb_x - g * (h + b) * h_x
                + 0.5 * h * v2_x
                - 0.5 * v^2 * h_x
                + 0.5 * hv_x * v
                - 0.5 * h * v * v_x
                + p_x
                + 1.5 * p_h * b_x
                + ψ * b_x
                - fv_x)
    # The code below is equivalent to
    # dv .= (Diagonal(h .+ 0.75 .* h .* b_x.^2) - Dmat * (Diagonal(1/3 .* h.^3) * Dmat - Diagonal(0.5 .* h.^2 .* b_x)) - Diagonal(0.5 .* h.^2 .* b_x) * Dmat) \ tmp
    # but faster since the symbolic factorization is reused.
    # Floating point errors accumulate a bit and the system matrix
    # is not necessarily perfectly symmetric but only up to round-off errors.
    # We wrap it here to avoid issues with the factorization.
    @.. M_h_p_h_bx2 = h + h * b_x^2
    scale_by_mass_matrix!(M_h_p_h_bx2, D)
    inv3 = 1 / 3
    @.. M_h3_3 = inv3 * h^3
    scale_by_mass_matrix!(M_h3_3, D)
    @.. M_h2_bx = 0.5 * h^2 * b_x
    scale_by_mass_matrix!(M_h2_bx, D)
    system_matrix = Symmetric(Diagonal(M_h_p_h_bx2)
                            + Dmat_minus' * (Diagonal(M_h3_3) * Dmat_minus
                                        - Diagonal(M_h2_bx))
                            - Diagonal(M_h2_bx) * Dmat_minus)
    if issparse(system_matrix)
        (; factorization) = parameters
        cholesky!(factorization, system_matrix; check = false)
        if issuccess(factorization)
            scale_by_mass_matrix!(tmp, D)
            dv .= factorization \ tmp
        else
            # The factorization may fail if the time step is too large
            # and h becomes negative.
            fill!(dv, Inf)
        end
    else
        factorization = cholesky!(system_matrix)
        scale_by_mass_matrix!(tmp, D)
        ldiv!(dv, factorization, tmp)
    end

    return nothing
end

function setup(::typeof(rhs_serre_green_naghdi_full_visc!),
               h_func, v_func, b_func;
               g, mu, λ, D)
    u0, parameters = setup(rhs_serre_green_naghdi_full!,
                           h_func, v_func, b_func;
                           g, λ, D)
    fv_x = zero(parameters.tmp)

    parameters = (; parameters..., fv_x, mu)

    return u0, parameters
end

function energy(::typeof(rhs_serre_green_naghdi_full_visc!),
                u, parameters)
    energy(rhs_serre_green_naghdi_full!, u, parameters)
end


#####################################################################
# Soliton solution of Serre-Green-Naghdi
function solve_soliton_serre_green_naghdi(rhs!, D;
                                          g = 9.81, λ = 1.0e4,
                                          alg = Tsit5(),
                                          abstol = 1.0e-7,
                                          reltol = 1.0e-7,
                                          kwargs...)
    x = grid(D)
    xmin = SummationByPartsOperators.xmin(D)
    xmax = SummationByPartsOperators.xmax(D)

    # setup initial data
    h1 = 1.0
    h2 = 1.2
    c = sqrt(g * h2)

    function h_analytical(t, x)
        x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
        return h1 + (h2 - h1) * sech(x_t / 2 * sqrt(3 * (h2 - h1) / (h1^2 * h2)))^2
    end

    function v_analytical(t, x)
        return c * (1 - h1 / h_analytical(t, x))
    end

    h_func = x -> h_analytical(0.0, x)
    v_func = x -> v_analytical(0.0, x)
    b_func = zero

    u0, parameters = setup(rhs!, h_func, v_func, b_func;
                           g, λ, D)

    # create and solve ODE
    tspan = (0.0, (xmax - xmin) / c)
    ode = ODEProblem(rhs!, u0, tspan, parameters)

    sol = solve(ode, alg;
                save_everystep = false,
                abstol, reltol, kwargs...)

    # compute errors at the final time
    h = h_analytical.(last(tspan), x)
    v = v_analytical.(last(tspan), x)
    error_h = sqrt(integrate(abs2, sol.u[end].x[1] - h, D))
    error_v = sqrt(integrate(abs2, sol.u[end].x[2] - v, D))

    return (; sol, error_h, error_v)
end


# Soliton solution of Serre-Green-Naghdi
function solve_soliton_serre_green_naghdi_visc(rhs!, D;
                                               g = 9.81, mu, λ = 1.0e4,
                                               alg = Tsit5(),
                                               abstol = 1.0e-7, reltol = 1.0e-7)
    x = grid(D)
    xmin = SummationByPartsOperators.xmin(D)
    xmax = SummationByPartsOperators.xmax(D)

    # setup initial data
    h1 = 1.0
    h2 = 1.2
    c = sqrt(g * h2)

    function h_analytical(t, x)
        x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
        return h1 + (h2 - h1) * sech(x_t / 2 * sqrt(3 * (h2 - h1) / (h1^2 * h2)))^2
    end

    function v_analytical(t, x)
        return c * (1 - h1 / h_analytical(t, x))
    end

    h_func = x -> h_analytical(0.0, x)
    v_func = x -> v_analytical(0.0, x)
    b_func = zero

    u0, parameters = setup(rhs!, h_func, v_func, b_func;
    g, mu, λ, D)

    # create and solve ODE
    tspan = (0.0, (xmax - xmin) / c)
    ode = ODEProblem(rhs!, u0, tspan, parameters)

    sol = solve(ode, alg;
    save_everystep = false,
    abstol, reltol)

    # compute errors at the final time
    h = h_analytical.(last(tspan), x)
    v = v_analytical.(last(tspan), x)
    error_h = sqrt(integrate(abs2, sol.u[end].x[1] - h, D))
    error_v = sqrt(integrate(abs2, sol.u[end].x[2] - v, D))

    return (; sol, error_h, error_v)
end



#####################################################################
# Code to reproduce the results in the paper

function convergence_tests_soliton(; latex = false)
    @info "Soliton of the Serre-Green-Naghdi equations"
    g = 9.81
    xmin = -50.0
    xmax = 50.0
    h0 = 1


    @info "Finite differences (order 2) for original system"
    let rhs! = rhs_serre_green_naghdi_flat!
        accuracy_order = 2
        num_nodes = [200, 400, 600, 800, 1000]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Finite differences (order 2) for original system plus viscosity"
    let rhs! = rhs_serre_green_naghdi_flat_visc!
        accuracy_order = 2
        num_nodes = [200, 400, 600, 800, 1000]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^accuracy_order)/accuracy_order

            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Finite differences (order 4) for original system"
    let rhs! = rhs_serre_green_naghdi_flat!
        accuracy_order = 4
        num_nodes = [100, 200, 300, 400, 500]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Finite differences (order 4) for original system plus viscosity"
    let rhs! = rhs_serre_green_naghdi_flat_visc!
        accuracy_order = 4
        num_nodes = [100, 200, 300, 400, 500]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^accuracy_order)/accuracy_order

            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Finite differences (order 2) for original system, upwind operators"
    let rhs! = rhs_serre_green_naghdi_flat!
        # First-order upwind operators yield second-order central operators
        accuracy_order = 1
        num_nodes = [200, 400, 600, 800, 1000]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1,
                                 accuracy_order,
                                 xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Finite differences (order 2) for original system plus viscosity, upwind operators"
    let rhs! = rhs_serre_green_naghdi_flat_visc!
        # First-order upwind operators yield second-order central operators
        accuracy_order = 1
        order = 2
        num_nodes = [200, 400, 600, 800, 1000]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^order)/order

            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1,
                                 accuracy_order,
                                 xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Finite differences (order 4) for original system, upwind operators"
    let rhs! = rhs_serre_green_naghdi_flat!
        # Third-order upwind operators yield fourth-order central operators
        accuracy_order = 3
        num_nodes = [100, 200, 300, 400, 500]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1,
                                 accuracy_order,
                                 xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Finite differences (order 4) for original system plus viscosity, upwind operators"
    let rhs! = rhs_serre_green_naghdi_flat_visc!
        # Third-order upwind operators yield fourth-order central operators
        accuracy_order = 3
        order = 4
        num_nodes = [200, 400, 600, 800, 1000]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^order)/order

            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1,
                                 accuracy_order,
                                 xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Finite differences (order 2) for hyperbolic approximation"
    let rhs! = rhs_nonconservative!
        accuracy_order = 2
        num_nodes = [200, 400, 600, 800, 1000]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9
        λ = 1.0e4
        @show λ

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Finite differences (order 2) for hyperbolic approximation plus viscosity"
    let rhs! = rhs_nonconservative_visc!
        accuracy_order = 2
        num_nodes = [200, 400, 600, 800, 1000]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9
        λ = 1.0e4
        @show λ

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^accuracy_order)/accuracy_order

            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Finite differences (order 4) for hyperbolic approximation"
    let rhs! = rhs_nonconservative!
        accuracy_order = 4
        num_nodes = [100, 200, 300, 400, 500]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9
        λ = 1.0e6
        @show λ

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Finite differences (order 4) for hyperbolic approximation plus viscosity"
    let rhs! = rhs_nonconservative_visc!
        accuracy_order = 4
        num_nodes = [100, 200, 300, 400, 500]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9
        λ = 1.0e6
        @show λ

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^accuracy_order)/accuracy_order

            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end


    @info "Convergence of the hyperbolic approximation with finite differences"
    let rhs! = rhs_nonconservative!
        accuracy_order = 8
        N = 500
        λs = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for λ in λs
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(λs, errors_h)
        eoc_v = compute_eoc(λs, errors_v)

        data = hcat(λs, errors_h, eoc_h, errors_v, eoc_v)
        header = [L"\lambda", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Convergence of the hyperbolic approximation plus viscosity with finite differences"
    let rhs! = rhs_nonconservative_visc!
        accuracy_order = 8
        N = 500
        mu = ( xmax - xmin )/N
        mu = (mu^accuracy_order)/accuracy_order
        λs = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for λ in λs
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(λs, errors_h)
        eoc_v = compute_eoc(λs, errors_v)

        data = hcat(λs, errors_h, eoc_h, errors_v, eoc_v)
        header = [L"\lambda", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    return nothing
end


function plot_convergence_tests_soliton(; latex = false)
    @info "Soliton of the Serre-Green-Naghdi equations"
    g = 9.81
    xmin = -50.0
    xmax = 50.0


    fig_O2_h = plot(xlim=(-1.1, 0.2) ; xguide = L"log(\Delta x)", yguide = L"log(err_h)", plot_kwargs()...)
    fig_O2_v = plot(xlim=(-1.1, 0.2); xguide = L"log(\Delta x)", yguide = L"log(err_v)", plot_kwargs()...)

    fig_O4_h = plot(xlim=(-0.8, 0.5); xguide = L"log(\Delta x)", yguide = L"log(err_h)", plot_kwargs()...)
    fig_O4_v = plot(xlim=(-0.8, 0.5); xguide = L"log(\Delta x)", yguide = L"log(err_v)", plot_kwargs()...)

    # Plot all O2 convergence

    @info "Finite differences (order 2) for original system"
    let rhs! = rhs_serre_green_naghdi_flat!
        accuracy_order = 2
        num_nodes = [200, 400, 600, 800, 1000]
        logdx = [log10(100/200), log10(100/400), log10(100/600), log10(100/800), log10(100/1000)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_O2_h, logdx, logerror_h; label = "orig-GN, central", color = 1, linestyle = :solid, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_O2_v, logdx, logerror_v; label = "orig-GN, central", color = 1, linestyle = :solid, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences (order 2) for original system plus viscosity"
    let rhs! = rhs_serre_green_naghdi_flat_visc!
        accuracy_order = 2
        num_nodes = [200, 400, 600, 800, 1000]
        logdx = [log10(100/200), log10(100/400), log10(100/600), log10(100/800), log10(100/1000)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^accuracy_order)/accuracy_order

            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_O2_h, logdx, logerror_h; label = "orig-GN, central/AV", color = 1, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_O2_v, logdx, logerror_v; label = "orig-GN, central/AV", color = 1, linestyle = :dot, legend=:bottomright, plot_kwargs()...)
    end


    @info "Finite differences (order 4) for original system"
    let rhs! = rhs_serre_green_naghdi_flat!
        accuracy_order = 4
        num_nodes = [100, 200, 300, 400, 500]
        logdx = [log10(100/100), log10(100/200), log10(100/300), log10(100/400), log10(100/500)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_O4_h, logdx, logerror_h; label = "orig-GN, central", color = 1, linestyle = :solid, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_O4_v, logdx, logerror_v; label = "orig-GN, central", color = 1, linestyle = :solid, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences (order 4) for original system plus viscosity"
    let rhs! = rhs_serre_green_naghdi_flat_visc!
        accuracy_order = 4
        num_nodes = [100, 200, 300, 400, 500]
        logdx = [log10(100/100), log10(100/200), log10(100/300), log10(100/400), log10(100/500)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^accuracy_order)/accuracy_order

            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_O4_h, logdx, logerror_h; label = "orig-GN, central/AV", color = 1, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_O4_v, logdx, logerror_v; label = "orig-GN, central/AV", color = 1, linestyle = :dot, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences (order 2) for original system, upwind operators"
    let rhs! = rhs_serre_green_naghdi_flat!
        # First-order upwind operators yield second-order central operators
        accuracy_order = 1
        num_nodes = [200, 400, 600, 800, 1000]
        logdx = [log10(100/200), log10(100/400), log10(100/600), log10(100/800), log10(100/1000)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1,
                                 accuracy_order,
                                 xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_O2_h, logdx, logerror_h; label = "orig-GN, upwind", color = 2, linestyle = :solid, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_O2_v, logdx, logerror_v; label = "orig-GN, upwind", color = 2, linestyle = :solid, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences (order 2) for original system plus viscosity, upwind operators"
    let rhs! = rhs_serre_green_naghdi_flat_visc!
        # First-order upwind operators yield second-order central operators
        accuracy_order = 1
        order = 2
        num_nodes = [200, 400, 600, 800, 1000]
        logdx = [log10(100/200), log10(100/400), log10(100/600), log10(100/800), log10(100/1000)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^order)/order

            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1,
                                 accuracy_order,
                                 xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_O2_h, logdx, logerror_h; label = "orig-GN, upwind/AV", color = 2, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_O2_v, logdx, logerror_v; label = "orig-GN, upwind/AV", color = 2, linestyle = :dot, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences (order 4) for original system, upwind operators"
    let rhs! = rhs_serre_green_naghdi_flat!
        # Third-order upwind operators yield fourth-order central operators
        accuracy_order = 3
        num_nodes = [100, 200, 300, 400, 500]
        logdx = [log10(100/100), log10(100/200), log10(100/300), log10(100/400), log10(100/500)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1,
                                 accuracy_order,
                                 xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_O4_h, logdx, logerror_h; label = "orig-GN, upwind", color = 2, linestyle = :solid, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_O4_v, logdx, logerror_v; label = "orig-GN, upwind", color = 2, linestyle = :solid, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences (order 4) for original system plus viscosity, upwind operators"
    let rhs! = rhs_serre_green_naghdi_flat_visc!
        # Third-order upwind operators yield fourth-order central operators
        accuracy_order = 3
        order = 4
        num_nodes = [100, 200, 300, 400, 500]
        logdx = [log10(100/100), log10(100/200), log10(100/300), log10(100/400), log10(100/500)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^order)/order

            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1,
                                 accuracy_order,
                                 xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
        logerror_h = log10.(errors_h)
        plot!(fig_O4_h, logdx, logerror_h; label = "orig-GN, upwind/AV", color = 2, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_O4_v, logdx, logerror_v; label = "orig-GN, upwind/AV", color = 2, linestyle = :dot, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences (order 2) for hyperbolic approximation"
    let rhs! = rhs_nonconservative!
        accuracy_order = 2
        num_nodes = [200, 400, 600, 800, 1000]
        logdx = [log10(100/200), log10(100/400), log10(100/600), log10(100/800), log10(100/1000)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9
        λ = 1.0e4
        @show λ

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_O2_h, logdx, logerror_h; label = "hyp-GN, central", color = 3, linestyle = :solid, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_O2_v, logdx, logerror_v; label = "hyp-GN, central", color = 3, linestyle = :solid, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences (order 2) for hyperbolic approximation plus viscosity"
    let rhs! = rhs_nonconservative_visc!
        accuracy_order = 2
        num_nodes = [200, 400, 600, 800, 1000]
        logdx = [log10(100/200), log10(100/400), log10(100/600), log10(100/800), log10(100/1000)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9
        λ = 1.0e4
        @show λ

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^accuracy_order)/accuracy_order

            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
        logerror_h = log10.(errors_h)
        plot!(fig_O2_h, logdx, logerror_h; label = "hyp-GN, central/AV", color = 3, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_O2_v, logdx, logerror_v; label = "hyp-GN, central/AV", color = 3, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        slope_h =  2*logdx .+ errors_h[1].*0.5
        plot!(fig_O2_h, logdx, slope_h; label = "slope 2", color =:black, linestyle = :solid, legend=:bottomright, plot_kwargs()...)

        slope_v =  2*logdx .+ errors_v[1].*0.5
        plot!(fig_O2_v, logdx, slope_v; label = "slope 2", color =:black, linestyle = :solid, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences (order 4) for hyperbolic approximation"
    let rhs! = rhs_nonconservative!
        accuracy_order = 4
        num_nodes = [100, 200, 300, 400, 500]
        logdx = [log10(100/100), log10(100/200), log10(100/300), log10(100/400), log10(100/500)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9
        λ = 1.0e6
        @show λ

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
        logerror_h = log10.(errors_h)
        plot!(fig_O4_h, logdx, logerror_h; label = "hyp-GN, central", color = 3, linestyle = :solid, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_O4_v, logdx, logerror_v; label = "hyp-GN, central", color = 3, linestyle = :solid, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences (order 4) for hyperbolic approximation plus viscosity"
    let rhs! = rhs_nonconservative_visc!
        accuracy_order = 4
        num_nodes = [100, 200, 300, 400, 500]
        logdx = [log10(100/100), log10(100/200), log10(100/300), log10(100/400), log10(100/500)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9
        λ = 1.0e6
        @show λ

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^accuracy_order)/accuracy_order

            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_O4_h, logdx, logerror_h; label = "hyp-GN, central/AV", color = 3, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_O4_v, logdx, logerror_v; label = "hyp-GN, central/AV", color = 3, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        slope_h =  4*logdx .+ errors_h[1].*0.5
        plot!(fig_O4_h, logdx, slope_h; label = "slope 4", color =:black, linestyle = :solid, legend=:bottomright, plot_kwargs()...)

        slope_v =  4*logdx .+ errors_v[1].*0.5
        plot!(fig_O4_v, logdx, slope_v; label = "slope 4", color =:black, linestyle = :solid, legend=:bottomright, plot_kwargs()...)
    end



    savefig(fig_O2_h, joinpath(figdir, "soliton_GN_hconvergence_O2.pdf"))
    savefig(fig_O4_h, joinpath(figdir, "soliton_GN_hconvergence_O4.pdf"))

    savefig(fig_O2_v, joinpath(figdir, "soliton_GN_vconvergence_O2.pdf"))
    savefig(fig_O4_v, joinpath(figdir, "soliton_GN_vconvergence_O4.pdf"))


    @info "Convergence of the hyperbolic approximation with finite differences"
    let rhs! = rhs_nonconservative!
        accuracy_order = 8
        N = 500
        λs = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for λ in λs
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi(
                rhs!, D; g, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(λs, errors_h)
        eoc_v = compute_eoc(λs, errors_v)

        data = hcat(λs, errors_h, eoc_h, errors_v, eoc_v)
        header = [L"\lambda", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Convergence of the hyperbolic approximation plus viscosity with finite differences"
    let rhs! = rhs_nonconservative_visc!
        accuracy_order = 8
        N = 500
        mu = ( xmax - xmin )/N
        mu = (mu^accuracy_order)/accuracy_order
        λs = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9

        errors_h = Float64[]
        errors_v = Float64[]

        for λ in λs
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_soliton_serre_green_naghdi_visc(
                rhs!, D; g, mu, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(λs, errors_h)
        eoc_v = compute_eoc(λs, errors_v)

        data = hcat(λs, errors_h, eoc_h, errors_v, eoc_v)
        header = [L"\lambda", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    return nothing
end



function convergence_tests_manufactured_hyperbolic(; latex = false)
    @info "Manufactured solution of the hyperbolic approximation"
    g = 9.81
    xmin = 0.0
    xmax = 1.0

    fig_h = plot(xlim=(-2.5, 0.0) ; xguide = L"log(\Delta x)", yguide = L"log(err_h)", plot_kwargs()...)
    fig_v = plot(xlim=(-2.5, 0.0); xguide = L"log(\Delta x)", yguide = L"log(err_v)", plot_kwargs()...)

    @info "Finite differences (order 4)"
    let
        accuracy_order = 4
        num_nodes = [50, 100, 150, 200]
        logdx = [log10(1/50), log10(1/100), log10(1/150), log10(1/200)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9
        λ = 1.0e4
        @show λ

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_manufactured_hyperbolic(
                D; g, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_h, logdx, logerror_h; label = "central O4", color = 1, linestyle = :solid, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_v, logdx, logerror_v; label = "central O4", color = 1, linestyle = :solid, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences plus viscosity (order 4)"
    let
        accuracy_order = 4
        num_nodes = [50, 100, 150, 200]
        logdx = [log10(1/50), log10(1/100), log10(1/150), log10(1/200)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9
        λ = 1.0e4
        @show λ

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^accuracy_order)/accuracy_order

            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_manufactured_hyperbolic_visc(
                D; g, mu, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_h, logdx, logerror_h; label = "central O4/AV", color = 1, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_v, logdx, logerror_v; label = "central O4/AV", color = 1, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        slope_h =  4*logdx .+ 4.7
        plot!(fig_h, logdx, slope_h; label = "slope 4", color =:black, linestyle = :solid, legend=:bottomright, plot_kwargs()...)

        slope_v =  4*logdx .+ 4.
        plot!(fig_v, logdx, slope_v; label = "slope 4", color =:black, linestyle = :solid, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences (order 6)"
    let
        accuracy_order = 6
        num_nodes = [25, 50, 75, 100]
        logdx = [log10(1/25), log10(1/50), log10(1/75), log10(1/100)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9
        λ = 1.0e4
        @show λ

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_manufactured_hyperbolic(
                D; g, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_h, logdx, logerror_h; label = "central O6", color = 2, linestyle = :solid, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_v, logdx, logerror_v; label = "central O6", color = 2, linestyle = :solid, legend=:bottomright, plot_kwargs()...)
    end

    @info "Finite differences plus viscosity (order 6)"
    let
        accuracy_order = 6
        num_nodes = [25, 50, 75, 100]
        logdx = [log10(1/25), log10(1/50), log10(1/75), log10(1/100)]
        alg = Tsit5()
        abstol = 1.0e-9
        reltol = 1.0e-9
        λ = 1.0e4
        @show λ

        errors_h = Float64[]
        errors_v = Float64[]

        for N in num_nodes
            mu = ( xmax - xmin )/N
            mu = (mu^accuracy_order)/accuracy_order

            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order,
                                             xmin, xmax, N)
            @time (; error_h, error_v) = solve_manufactured_hyperbolic_visc(
                D; g, mu, λ, alg, abstol, reltol
            )
            push!(errors_h, error_h)
            push!(errors_v, error_v)
        end

        eoc_h = compute_eoc(num_nodes, errors_h)
        eoc_v = compute_eoc(num_nodes, errors_v)

        data = hcat(num_nodes, errors_h, eoc_h, errors_v, eoc_v)
        header = ["#nodes", "L2 error h", "L2 EOC h", "L2 error v", "L2 EOC v"]
        kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                        ft_printf("%.2e", [2, 4]),
                                        ft_printf("%.2f", [3, 5])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end

        logerror_h = log10.(errors_h)
        plot!(fig_h, logdx, logerror_h; label = "central O6/AV", color = 2, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        logerror_v = log10.(errors_v)
        plot!(fig_v, logdx, logerror_v; label = "central O6/AV", color = 2, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        slope_h =  6*logdx .+ 6.85
        plot!(fig_h, logdx, slope_h; label = "slope 6", color =:black, linestyle = :dot, legend=:bottomright, plot_kwargs()...)

        slope_v =  6*logdx .+ 6.75
        plot!(fig_v, logdx, slope_v; label = "slope 6", color =:black, linestyle = :dot, legend=:bottomright, plot_kwargs()...)
    end

    savefig(fig_h, joinpath(figdir, "manufactured_hypGN_hconvergence.pdf"))
    savefig(fig_v, joinpath(figdir, "manufactured_hypGN_vconvergence.pdf"))

    return nothing
end



function rhs_nonconservative_manufactured!(du, u, parameters, t)
    # call standard right-hand side
    rhs_nonconservative!(du, u, parameters, t)

    # unpack physical parameters and SBP operator `D`
    (; g, λ, D) = parameters
    x = grid(D)

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h`, the velocity `v`,
    # and the additional variables `w` and `η`.
    dh, dv, dw, dη = du.x

    # add source terms for the manufactured solution
    # CSE
    (; h_x, v_x, hv_x, h_hpb_x, η_x, η2_h_x, w_x, hvw_x, tmp) = parameters
    a1 = h_x
    a2 = v_x
    a3 = hv_x
    a4 = h_hpb_x
    a5 = η_x
    a6 = η2_h_x
    a7 = w_x
    a8 = hvw_x
    a9 = tmp

    @.. a1 = sinpi(4 * t - 2 * x)
    @.. a2 = cospi(4 * t - 2 * x)
    @.. a3 = sinpi(2 * t - x)
    @.. a4 = cospi(2 * t - x)
    @.. a5 = sinpi(t - 2 * x)
    @.. a6 = cospi(t - 2 * x)
    @.. a7 = sinpi(2 * x)
    @.. a8 = cospi(2 * x)
    @.. a9 = sinpi(2 * t - 4 * x)
    e1 = exp(t)
    e2 = exp(t / 2)

    # Source terms for variable bathymetry
    @.. dh += -4*pi*a1 - a5*(2*pi*a1 - 4*pi*a7) + 2*pi*a6*(a2 + 2*a8 + 7)

    @.. dv += -2*pi*a5*a6 - pi*a6 + 4*pi*a7*g + g*(2*pi*a1 - 4*pi*a7)

    @.. dw += 8*pi^2*a1*a6 - a5*(4*pi^2*a5*(-a2 - 2*a8 - 7) + 2*pi*a6*(-2*pi*a1 + 4*pi*a7)) - 2*pi^2*a5*(-a2 - 2*a8 - 7)

    @.. dη += -4*pi*a1 - 6*pi*a5*a7 - a5*(2*pi*a1 - 4*pi*a7) - 2*pi*a6*(-a2 - 2*a8 - 7)

    return nothing
end

function solve_manufactured_hyperbolic(D;
                                       g = 9.81, λ = 1.0e4,
                                       alg = Tsit5(),
                                       abstol = 1.0e-7, reltol = 1.0e-7)
    x = grid(D)

    # setup initial data
    function h_analytical(t, x)
        h_plus_b = 2 + cospi(2 * (x - 2 * t))
        b = -5 - 2 * cospi(2 * x)
        return h_plus_b - b
    end

    function v_analytical(t, x)
        return sinpi(2 * (x - t / 2))
    end

    h_func = x -> h_analytical(0.0, x)
    v_func = x -> v_analytical(0.0, x)
    b_func = x -> -5 - 2 * cospi(2 * x)
    # b_func = zero

    u0, parameters = setup(rhs_nonconservative!,
                           h_func, v_func, b_func;
                           g, λ, D)

    # create and solve ODE
    tspan = (0.0, 1.0)
    ode = ODEProblem(rhs_nonconservative_manufactured!, u0, tspan, parameters)

    sol = solve(ode, alg;
                save_everystep = false,
                abstol, reltol)

    # compute errors at the final time
    h = h_analytical.(last(tspan), x)
    v = v_analytical.(last(tspan), x)
    error_h = sqrt(integrate(abs2, sol.u[end].x[1] - h, D))
    error_v = sqrt(integrate(abs2, sol.u[end].x[2] - v, D))

    return (; sol, error_h, error_v)
end

function rhs_nonconservative_manufactured_visc!(du, u, parameters, t)
    # call standard right-hand side
    rhs_nonconservative!(du, u, parameters, t)

    # unpack physical parameters and SBP operator `D`
    (; g, λ, D) = parameters
    x = grid(D)

    # `u` and `du` are `ArrayPartition`s. They collect the individual
    # arrays for the water height `h`, the velocity `v`,
    # and the additional variables `w` and `η`.
    dh, dv, dw, dη = du.x

    # add source terms for the manufactured solution
    # CSE
    (; h_x, v_x, hv_x, h_hpb_x, η_x, η2_h_x, w_x, hvw_x, tmp) = parameters
    a1 = h_x
    a2 = v_x
    a3 = hv_x
    a4 = h_hpb_x
    a5 = η_x
    a6 = η2_h_x
    a7 = w_x
    a8 = hvw_x
    a9 = tmp

    @.. a1 = sinpi(4 * t - 2 * x)
    @.. a2 = cospi(4 * t - 2 * x)
    @.. a3 = sinpi(2 * t - x)
    @.. a4 = cospi(2 * t - x)
    @.. a5 = sinpi(t - 2 * x)
    @.. a6 = cospi(t - 2 * x)
    @.. a7 = sinpi(2 * x)
    @.. a8 = cospi(2 * x)
    @.. a9 = sinpi(2 * t - 4 * x)
    e1 = exp(t)
    e2 = exp(t / 2)


    # Source terms for variable bathymetry
    @.. dh += -4*pi*a1 - a5*(2*pi*a1 - 4*pi*a7) + 2*pi*a6*(a2 + 2*a8 + 7)

    @.. dv += -2*pi*a5*a6 - pi*a6 + 4*pi*a7*g + g*(2*pi*a1 - 4*pi*a7)

    @.. dw += 8*pi^2*a1*a6 - a5*(4*pi^2*a5*(-a2 - 2*a8 - 7) + 2*pi*a6*(-2*pi*a1 + 4*pi*a7)) - 2*pi^2*a5*(-a2 - 2*a8 - 7)

    @.. dη += -4*pi*a1 - 6*pi*a5*a7 - a5*(2*pi*a1 - 4*pi*a7) - 2*pi*a6*(-a2 - 2*a8 - 7)

    return nothing
end

function solve_manufactured_hyperbolic_visc(D;
                                            g = 9.81, mu, λ = 1.0e4,
                                            alg = Tsit5(),
                                            abstol = 1.0e-7,
                                            reltol = 1.0e-7)
    x = grid(D)

    # setup initial data
    function h_analytical(t, x)
        h_plus_b = 2 + cospi(2 * (x - 2 * t))
        b = -5 - 2 * cospi(2 * x)
        return h_plus_b - b
    end

    function v_analytical(t, x)
        return sinpi(2 * (x - t / 2))
    end

    h_func = x -> h_analytical(0.0, x)
    v_func = x -> v_analytical(0.0, x)
    b_func = x -> -5 - 2 * cospi(2 * x)
    # b_func = zero

    u0, parameters = setup(rhs_nonconservative_visc!,
                           h_func, v_func, b_func;
                           g, mu, λ, D)

    # create and solve ODE
    tspan = (0.0, 1.0)
    ode = ODEProblem(rhs_nonconservative_manufactured!, u0, tspan, parameters)

    sol = solve(ode, alg;
                save_everystep = false,
                abstol, reltol)

    # compute errors at the final time
    h = h_analytical.(last(tspan), x)
    v = v_analytical.(last(tspan), x)
    error_h = sqrt(integrate(abs2, sol.u[end].x[1] - h, D))
    error_v = sqrt(integrate(abs2, sol.u[end].x[2] - v, D))

    return (; sol, error_h, error_v)
end



#####################################################################
# Qualitative comparison of upwind vs. central

function plot_solution_upwind_vs_central()
    λ = 500.0
    xmin = -150.0
    xmax = 150.0
    N = 500
    abstol = 1.0e-5
    reltol = 1.0e-5

    D_central = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order = 2,
                                             xmin, xmax, N)
    # First-order upwind operators yield second-order central operators
    D_upwind = upwind_operators(periodic_derivative_operator;
                                derivative_order = 1,
                                accuracy_order = 1,
                                xmin, xmax, N)

    @info "Flat topography"
    for (rhs!, D, name) in [(rhs_serre_green_naghdi_flat!,
                             D_central, "original_central"),
                            (rhs_serre_green_naghdi_flat!,
                             D_upwind, "original_upwind")]
        if rhs! === rhs_nonconservative!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        Δx = step(grid(D))
        @info "Running" name Δx
        @time (; sol) = solve_conservation(rhs!, D;
                                           λ, alg, abstol, reltol)

        x = grid(D)
        h0 = sol.u[begin].x[1]
        h = sol.u[end].x[1]
        fig_h = plot(; xguide = L"x", yguide = L"h", plot_kwargs()...)
        plot!(fig_h, x, h0;
              label = L"h^0", color = :gray, linestyle = :dash, plot_kwargs()...)
        plot!(fig_h, x, h;
              label = L"h", color = 1, linestyle = :solid, plot_kwargs()...)

        savefig(fig_h, joinpath(figdir, "upwind_vs_central__$(name).pdf"))
    end


    @info "Variable bathymetry (mild)"
    for (rhs!, D, name) in [(rhs_serre_green_naghdi_mild!,
                             D_central, "original_mild_central"),
                            (rhs_serre_green_naghdi_mild!,
                             D_upwind, "original_mild_upwind")]
        if rhs! === rhs_nonconservative!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        Δx = step(grid(D))
        @info "Running" name Δx
        @time (; sol) = solve_conservation(rhs!, D;
                                           λ, alg, abstol, reltol,
                                           b_func = x -> 0.25 * cospi(x / 75))

        x = grid(D)
        h0 = sol.u[begin].x[1]
        h = sol.u[end].x[1]
        b = sol.prob.p.b
        fig_h = plot(; xguide = L"x", yguide = L"h + b", plot_kwargs()...)
        plot!(fig_h, x, h0 + b;
              label = L"h^0 + b", color = :gray, linestyle = :dash, plot_kwargs()...)
        plot!(fig_h, x, h + b;
              label = L"h + b", color = 1, linestyle = :solid, plot_kwargs()...)
        plot!(fig_h, x, b; label = L"b", color = :gray, linestyle = :dot, plot_kwargs()...)

        savefig(fig_h, joinpath(figdir, "upwind_vs_central__$(name).pdf"))
    end


    @info "Variable bathymetry (full)"
    for (rhs!, D, name) in [(rhs_serre_green_naghdi_full!,
                             D_central, "original_full_central"),
                            (rhs_serre_green_naghdi_full!,
                             D_upwind, "original_full_upwind")]
        if rhs! === rhs_nonconservative!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        Δx = step(grid(D))
        @info "Running" name Δx
        @time (; sol) = solve_conservation(rhs!, D;
                                           λ, alg, abstol, reltol,
                                           b_func = x -> 0.25 * cospi(x / 75))

        x = grid(D)
        h0 = sol.u[begin].x[1]
        h = sol.u[end].x[1]
        b = sol.prob.p.b
        fig_h = plot(; xguide = L"x", yguide = L"h + b", plot_kwargs()...)
        plot!(fig_h, x, h0 + b;
              label = L"h^0 + b", color = :gray, linestyle = :dash, plot_kwargs()...)
        plot!(fig_h, x, h + b;
              label = L"h + b", color = 1, linestyle = :solid, plot_kwargs()...)
        plot!(fig_h, x, b; label = L"b", color = :gray, linestyle = :dot, plot_kwargs()...)

        savefig(fig_h, joinpath(figdir, "upwind_vs_central__$(name).pdf"))
    end


    @info "Results saved in the directory" figdir

    return nothing
end

function plot_solution_upwind_vs_central_visc()
    λ = 500.0
    xmin = -150.0
    xmax = 150.0
    N = 500
    abstol = 1.0e-5
    reltol = 1.0e-5
    order = 2



    D_central = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order = order,
                                             xmin, xmax, N)
    # First-order upwind operators yield second-order central operators
    D_upwind = upwind_operators(periodic_derivative_operator;
                                derivative_order = 1,
                                accuracy_order = order-1,
                                xmin, xmax, N)

    @info "Flat topography"
    mu = ( xmax - xmin )/N
    mu = (mu^order)/order

    for (rhs!, D, name) in [(rhs_serre_green_naghdi_flat_visc!,
                             D_central, "original_central"),
                            (rhs_serre_green_naghdi_flat_visc!,
                             D_upwind, "original_upwind")]
        if rhs! === rhs_nonconservative_visc!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        Δx = step(grid(D))
        @info "Running" name Δx
        @time (; sol) = solve_conservation_visc(rhs!, D; mu,
                                           λ, alg, abstol, reltol)

        x = grid(D)
        h0 = sol.u[begin].x[1]
        h = sol.u[end].x[1]
        fig_h = plot(; xguide = L"x", yguide = L"h", plot_kwargs()...)
        plot!(fig_h, x, h0;
              label = L"h^0", color = :gray, linestyle = :dash, plot_kwargs()...)
        plot!(fig_h, x, h;
              label = L"h", color = 1, linestyle = :solid, plot_kwargs()...)

        savefig(fig_h, joinpath(figdir, "upwind_vs_central_visc_500_$(name).pdf"))
    end


    @info "Variable bathymetry (mild)"
    mu = ( xmax - xmin )/N
    mu = (mu^order)/order

    for (rhs!, D, name) in [(rhs_serre_green_naghdi_mild_visc!,
                             D_central, "original_mild_central"),
                            (rhs_serre_green_naghdi_mild_visc!,
                             D_upwind, "original_mild_upwind")]
        if rhs! === rhs_nonconservative_visc!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        Δx = step(grid(D))
        @info "Running" name Δx
        @time (; sol) = solve_conservation_visc(rhs!, D; mu,
                                           λ, alg, abstol, reltol,
                                           b_func = x -> 0.25 * cospi(x / 75))

        x = grid(D)
        h0 = sol.u[begin].x[1]
        h = sol.u[end].x[1]
        b = sol.prob.p.b
        fig_h = plot(; xguide = L"x", yguide = L"h + b", plot_kwargs()...)
        plot!(fig_h, x, h0 + b;
              label = L"h^0 + b", color = :gray, linestyle = :dash, plot_kwargs()...)
        plot!(fig_h, x, h + b;
              label = L"h + b", color = 1, linestyle = :solid, plot_kwargs()...)
        plot!(fig_h, x, b; label = L"b", color = :gray, linestyle = :dot, plot_kwargs()...)

        savefig(fig_h, joinpath(figdir, "upwind_vs_central_visc_500_$(name).pdf"))
    end


    @info "Variable bathymetry (full)"
    mu = ( xmax - xmin )/N
    mu = (mu^order)/order

    for (rhs!, D, name) in [(rhs_serre_green_naghdi_full_visc!,
                             D_central, "original_full_central"),
                            (rhs_serre_green_naghdi_full_visc!,
                             D_upwind, "original_full_upwind")]
        if rhs! === rhs_nonconservative_visc!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        Δx = step(grid(D))
        @info "Running" name Δx
        @time (; sol) = solve_conservation_visc(rhs!, D; mu,
                                           λ, alg, abstol, reltol,
                                           b_func = x -> 0.25 * cospi(x / 75))

        x = grid(D)
        h0 = sol.u[begin].x[1]
        h = sol.u[end].x[1]
        b = sol.prob.p.b
        fig_h = plot(; xguide = L"x", yguide = L"h + b", plot_kwargs()...)
        plot!(fig_h, x, h0 + b;
              label = L"h^0 + b", color = :gray, linestyle = :dash, plot_kwargs()...)
        plot!(fig_h, x, h + b;
              label = L"h + b", color = 1, linestyle = :solid, plot_kwargs()...)
        plot!(fig_h, x, b; label = L"b", color = :gray, linestyle = :dot, plot_kwargs()...)

        savefig(fig_h, joinpath(figdir, "upwind_vs_central_visc_500_$(name).pdf"))
    end


    @info "Results saved in the directory" figdir

    return nothing
end



#####################################################################
# Conservation of invariants

function plot_solution_conservation_tests()
    D = periodic_derivative_operator(derivative_order = 1,
                                     accuracy_order = 2,
                                     xmin = -150.0,
                                     xmax = 150.0,
                                     N = 3_000)
    λ = 500.0


    @info "Flat topography"
    for (rhs!, name) in [(rhs_nonconservative!, "hyperbolic"),
                         (rhs_serre_green_naghdi_flat!, "original")]
        if rhs! === rhs_nonconservative!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        Δx = step(grid(D))
        @info "Running" name Δx
        @time (; sol) = solve_conservation(rhs!, D; λ, alg)

        x = grid(D)
        h0 = sol.u[begin].x[1]
        h = sol.u[end].x[1]
        fig_h = plot(; xguide = L"x", yguide = L"h", plot_kwargs()...)
        plot!(fig_h, x, h0;
              label = L"h^0", color = :gray, linestyle = :dash, plot_kwargs()...)
        plot!(fig_h, x, h;
              label = "\$h\$", color = 1, linestyle = :solid, plot_kwargs()...)

        savefig(fig_h, joinpath(figdir, "conservation_solution_$(name).pdf"))
    end

    @info "Variable bathymetry"
    for (rhs!, name) in [(rhs_nonconservative!, "variable_hyperbolic"),
                         (rhs_serre_green_naghdi_mild!, "variable_original_mild"),
                         (rhs_serre_green_naghdi_full!, "variable_original_full")]
        if rhs! === rhs_nonconservative!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        Δx = step(grid(D))
        @info "Running" name Δx
        @time (; sol) = solve_conservation(rhs!, D;
                                           λ,  alg,
                                           b_func = x -> 0.25 * cospi(x / 75))

        x = grid(D)
        h0 = sol.u[begin].x[1]
        h = sol.u[end].x[1]
        b = sol.prob.p.b
        fig_h = plot(; xguide = L"x", yguide = L"h + b", plot_kwargs()...)
        plot!(fig_h, x, h0 + b;
              label = L"h^0 + b", color = :gray, linestyle = :dash, plot_kwargs()...)
        plot!(fig_h, x, h + b;
              label = "\$h + b\$", color = 1, linestyle = :solid, plot_kwargs()...)
        plot!(fig_h, x, b; label = L"b", color = :gray, linestyle = :dot, plot_kwargs()...)

        savefig(fig_h, joinpath(figdir, "conservation_solution_$(name).pdf"))
    end


    @info "Results saved in the directory" figdir

    return nothing
end


function plot_solution_conservation_tests_visc()
    D = periodic_derivative_operator(derivative_order = 1,
                                     accuracy_order = 2,
                                     xmin = -150.0,
                                     xmax = 150.0,
                                     N = 3_000)
    λ = 500.0
    xmax = 150.0
    xmin = -150.0
    N = 3_000
    order = 2



    @info "Flat topography"
    mu = ( xmax - xmin )/N
    mu = (mu^order)/order

    for (rhs!, name) in [(rhs_nonconservative_visc!, "hyperbolic"),
                         (rhs_serre_green_naghdi_flat_visc!, "original")]
        if rhs! === rhs_nonconservative_visc!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        Δx = step(grid(D))
        @info "Running" name Δx
        @time (; sol) = solve_conservation_visc(rhs!, D; mu, λ, alg)

        x = grid(D)
        h0 = sol.u[begin].x[1]
        h = sol.u[end].x[1]
        fig_h = plot(; xguide = L"x", yguide = L"h", plot_kwargs()...)
        plot!(fig_h, x, h0;
              label = L"h^0", color = :gray, linestyle = :dash, plot_kwargs()...)
        plot!(fig_h, x, h;
              label = "\$h\$", color = 1, linestyle = :solid, plot_kwargs()...)

        savefig(fig_h, joinpath(figdir, "conservation_solution_visc_$(name).pdf"))
    end

    @info "Variable bathymetry"
    mu = ( xmax - xmin )/N
    mu = (mu^order)/order

    for (rhs!, name) in [(rhs_nonconservative_visc!, "variable_hyperbolic"),
                         (rhs_serre_green_naghdi_mild_visc!, "variable_original_mild"),
                         (rhs_serre_green_naghdi_full_visc!, "variable_original_full")]
        if rhs! === rhs_nonconservative_visc!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        Δx = step(grid(D))
        @info "Running" name Δx
        @time (; sol) = solve_conservation_visc(rhs!, D;
                                           mu, λ,  alg,
                                           b_func = x -> 0.25 * cospi(x / 75))

        x = grid(D)
        h0 = sol.u[begin].x[1]
        h = sol.u[end].x[1]
        b = sol.prob.p.b
        fig_h = plot(; xguide = L"x", yguide = L"h + b", plot_kwargs()...)
        plot!(fig_h, x, h0 + b;
              label = L"h^0 + b", color = :gray, linestyle = :dash, plot_kwargs()...)
        plot!(fig_h, x, h + b;
              label = "\$h + b\$", color = 1, linestyle = :solid, plot_kwargs()...)
        plot!(fig_h, x, b; label = L"b", color = :gray, linestyle = :dot, plot_kwargs()...)

        savefig(fig_h, joinpath(figdir, "conservation_solution_visc_$(name).pdf"))
    end


    @info "Results saved in the directory" figdir

    return nothing
end

function conservation_tests(; latex = true)
    xmin = -150.0
    xmax = 150.0
    N = 1_000

    function analyze_conservation(rhs!, D, dts, b_func, λ = nothing)
        errors_energy = Float64[]
        errors_water_mass = Float64[]
        errors_momentum = Float64[]

        for dt in dts
            @time (; energy, water_mass, momentum) = solve_conservation(
                rhs!, D;
                λ, b_func, dt, adaptive = false
            )
            error_energy = maximum(abs.(energy .- energy[begin]))
            error_water_mass = maximum(abs.(water_mass .- water_mass[begin]))
            error_momentum = maximum(abs.(momentum .- momentum[begin]))
            push!(errors_energy, error_energy)
            push!(errors_water_mass, error_water_mass)
            push!(errors_momentum, error_momentum)
        end

        eoc_momentum = -compute_eoc(dts, errors_momentum)
        eoc_energy = -compute_eoc(dts, errors_energy)

        # the momentum is only conserved for flat bathymetry
        if b_func === zero
            data = hcat(dts, errors_water_mass,
                        errors_momentum, eoc_momentum,
                        errors_energy, eoc_energy)
            header = ["dt", "error water mass",
                      "error momentum", "EOC momentum",
                      "error energy", "EOC energy"]
            kwargs = (; header, formatters=(ft_printf("%.4f", [1]),
                                            ft_printf("%.1e", [2, 3, 5]),
                                            ft_printf("%.2f", [4, 6])))
        else
            data = hcat(dts, errors_water_mass,
                        errors_energy, eoc_energy)
            header = ["dt", "error water mass",
                      "error energy", "EOC energy"]
            kwargs = (; header, formatters=(ft_printf("%.4f", [1]),
                                            ft_printf("%.1e", [2, 3]),
                                            ft_printf("%.2f", [4])))
        end
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    @info "Flat topography"
    b_func = zero

    @info "Hyperbolic approximation"
    let
        D = periodic_derivative_operator(derivative_order = 1;
                                         accuracy_order = 2,
                                         xmin, xmax, N)
        dts = [0.01, 0.005, 0.002, 0.001, 0.0005]
        λ = 500.0
        analyze_conservation(rhs_nonconservative!,
                             D, dts, b_func, λ)
    end

    orders = [2, 4]
    for order in orders
        @info "Hyperbolic approximation plus viscosity" order
        let
            D = periodic_derivative_operator(derivative_order = 1;
                                         accuracy_order = order,
                                         xmin, xmax, N)
            λ = 500.0
            dts = [0.01, 0.005, 0.002, 0.001, 0.0005]
            errors_energy = Float64[]
            errors_water_mass = Float64[]
            errors_momentum = Float64[]
            mu = ( xmax - xmin )/N
            mu = (mu^order)/order


            for dt in dts
                @time (; energy, water_mass, momentum) = solve_conservation_visc(
                    rhs_nonconservative_visc!, D; mu,
                    λ, dt, adaptive = false
                )
                error_energy = maximum(abs.(energy .- energy[begin]))
                error_water_mass = maximum(abs.(water_mass .- water_mass[begin]))
                error_momentum = maximum(abs.(momentum .- momentum[begin]))
                push!(errors_energy, error_energy)
                push!(errors_water_mass, error_water_mass)
                push!(errors_momentum, error_momentum)
            end

            eoc_energy = -compute_eoc(dts, errors_energy)
            eoc_momentum = -compute_eoc(dts, errors_momentum)

            data = hcat(dts, errors_water_mass,
                        errors_momentum, eoc_momentum,
                        errors_energy, eoc_energy)
            header = ["dt", "error water mass",
                      "error momentum", "EOC momentum",
                      "error energy", "EOC energy"]
            kwargs = (; header, formatters=(ft_printf("%.4f", [1]),
                                            ft_printf("%.1e", [2, 3, 5]),
                                            ft_printf("%.2f", [4, 6])))

            pretty_table(data; kwargs...)
            if latex
                pretty_table(data; kwargs..., backend=Val(:latex))
            end
        end
    end

    @info "Original Serre-Green-Naghdi equations with central SBP operators"
    let
        D = periodic_derivative_operator(derivative_order = 1;
                                         accuracy_order = 2,
                                         xmin, xmax, N)
        dts = [0.15, 0.05, 0.02, 0.01, 0.005]
        analyze_conservation(rhs_serre_green_naghdi_flat!,
                             D, dts, b_func)
    end

    orders = [2, 4]
    for order in orders
        @info "Original Serre-Green-Naghdi equations plus viscosity with central SBP operators" order
        let
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order = order,
                                             xmin, xmax, N)
            dts = [0.15, 0.05, 0.02, 0.01, 0.005]
            errors_energy = Float64[]
            errors_water_mass = Float64[]
            errors_momentum = Float64[]
            mu = ( xmax - xmin )/N
            mu = (mu^order)/order

            for dt in dts
                @time (; energy, water_mass, momentum) = solve_conservation_visc(
                    rhs_serre_green_naghdi_flat_visc!, D; mu,

                    dt, adaptive = false
                )
                error_energy = maximum(abs.(energy .- energy[begin]))
                error_water_mass = maximum(abs.(water_mass .- water_mass[begin]))
                error_momentum = maximum(abs.(momentum .- momentum[begin]))
                push!(errors_energy, error_energy)
                push!(errors_water_mass, error_water_mass)
                push!(errors_momentum, error_momentum)
            end

            eoc_energy = -compute_eoc(dts, errors_energy)
            eoc_momentum = -compute_eoc(dts, errors_momentum)

            data = hcat(dts, errors_water_mass,
                        errors_momentum, eoc_momentum,
                        errors_energy, eoc_energy)
            header = ["dt", "error water mass",
                      "error momentum", "EOC momentum",
                      "error energy", "EOC energy"]
            kwargs = (; header, formatters=(ft_printf("%.4f", [1]),
                                            ft_printf("%.1e", [2, 3, 5]),
                                            ft_printf("%.2f", [4, 6])))
            pretty_table(data; kwargs...)
            if latex
                pretty_table(data; kwargs..., backend=Val(:latex))
            end
        end
    end

    @info "Original Serre-Green-Naghdi equations with upwind SBP operators"
    let
        # first-order upwind operators yield a second-order central operator
        D = upwind_operators(periodic_derivative_operator;
                             derivative_order = 1,
                             accuracy_order = 1,
                             xmin, xmax, N)
        dts = [0.15, 0.05, 0.02, 0.01, 0.005]
        analyze_conservation(rhs_serre_green_naghdi_flat!,
                             D, dts, b_func)
    end

    orders = [2, 4]
    for order in orders
        @info "Original Serre-Green-Naghdi equations plus viscosity with upwind SBP operators" order
        let
            # first-order upwind operators yield a second-order central operator
            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1,
                                 accuracy_order = order-1,
                                 xmin, xmax, N)
            dts = [0.15, 0.05, 0.02, 0.01, 0.005]
            errors_energy = Float64[]
            errors_water_mass = Float64[]
            errors_momentum = Float64[]
            mu = ( xmax - xmin )/N
            mu = (mu^order)/order

            for dt in dts

                @time (; energy, water_mass, momentum) = solve_conservation_visc(
                    rhs_serre_green_naghdi_flat_visc!, D;mu,
                    dt, adaptive = false
                )
                error_energy = maximum(abs.(energy .- energy[begin]))
                error_water_mass = maximum(abs.(water_mass .- water_mass[begin]))
                error_momentum = maximum(abs.(momentum .- momentum[begin]))
                push!(errors_energy, error_energy)
                push!(errors_water_mass, error_water_mass)
                push!(errors_momentum, error_momentum)
            end

            eoc_energy = -compute_eoc(dts, errors_energy)
            eoc_momentum = -compute_eoc(dts, errors_momentum)

            data = hcat(dts, errors_water_mass,
                        errors_momentum, eoc_momentum,
                        errors_energy, eoc_energy)
            header = ["dt", "error water mass",
                      "error momentum", "EOC momentum",
                      "error energy", "EOC energy"]
            kwargs = (; header, formatters=(ft_printf("%.4f", [1]),
                                            ft_printf("%.1e", [2, 3, 5]),
                                            ft_printf("%.2f", [4, 6])))

            pretty_table(data; kwargs...)
            if latex
                pretty_table(data; kwargs..., backend=Val(:latex))
            end
        end
    end


    @info "Variable bathymetry"
    b_func = x -> 0.25 * cospi(x / 75)

    @info "Hyperbolic approximation"
    let
        D = periodic_derivative_operator(derivative_order = 1;
                                         accuracy_order = 2,
                                         xmin, xmax, N)
        dts = [0.01, 0.005, 0.002, 0.001, 0.0005]
        λ = 500.0
        analyze_conservation(rhs_nonconservative!,
                             D, dts, b_func, λ)
    end

    orders = [2, 4]
    for order in orders
        @info "Hyperbolic approximation plus viscosity" order
        let
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order = order,
                                             xmin, xmax, N)
            λ = 500.0
            dts = [0.01, 0.005, 0.002, 0.001, 0.0005]
            errors_energy = Float64[]
            errors_water_mass = Float64[]
            mu = ( xmax - xmin )/N
            mu = (mu^order)/order

            for dt in dts
                @time (; energy, water_mass) = solve_conservation_visc(
                    rhs_nonconservative_visc!, D; mu,
                    b_func = x -> 0.25 * cospi(x / 75),
                    λ, dt, adaptive = false
                )
                error_energy = maximum(abs.(energy .- energy[begin]))
                error_water_mass = maximum(abs.(water_mass .- water_mass[begin]))
                push!(errors_energy, error_energy)
                push!(errors_water_mass, error_water_mass)
            end

            eoc_energy = -compute_eoc(dts, errors_energy)

            data = hcat(dts, errors_water_mass, errors_energy, eoc_energy)
            header = ["dt", "error water mass", "error energy", "EOC energy"]
            kwargs = (; header, formatters=(ft_printf("%.4f", [1]),
                                            ft_printf("%.2e", [2, 3]),
                                            ft_printf("%.2f", [4])))
            pretty_table(data; kwargs...)
            if latex
                pretty_table(data; kwargs..., backend=Val(:latex))
            end
        end
    end

    @info "Original Serre-Green-Naghdi equations (mild slope) with central SBP operators"
    let
        D = periodic_derivative_operator(derivative_order = 1;
                                         accuracy_order = 2,
                                         xmin, xmax, N)
        dts = [0.15, 0.05, 0.02, 0.01, 0.005]
        analyze_conservation(rhs_serre_green_naghdi_mild!,
                             D, dts, b_func)
    end

    orders = [2, 4]
    for order in orders
        @info "Original Serre-Green-Naghdi equations (mild slope) plus viscosity with central SBP operators" order
        let
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order = order,
                                             xmin, xmax, N)
            dts = [0.15, 0.05, 0.02, 0.01, 0.005]
            errors_energy = Float64[]
            errors_water_mass = Float64[]
            mu = ( xmax - xmin )/N
            mu = (mu^order)/order

            for dt in dts
                @time (; energy, water_mass) = solve_conservation_visc(
                    rhs_serre_green_naghdi_mild_visc!, D; mu,
                    b_func = x -> 0.25 * cospi(x / 75),
                    dt, adaptive = false
                )
                error_energy = maximum(abs.(energy .- energy[begin]))
                error_water_mass = maximum(abs.(water_mass .- water_mass[begin]))
                push!(errors_energy, error_energy)
                push!(errors_water_mass, error_water_mass)
            end

            eoc_energy = -compute_eoc(dts, errors_energy)

            data = hcat(dts, errors_water_mass, errors_energy, eoc_energy)
            header = ["dt", "error water mass", "error energy", "EOC energy"]
            kwargs = (; header, formatters=(ft_printf("%.3f", [1]),
                                            ft_printf("%.2e", [2, 3]),
                                            ft_printf("%.2f", [4])))
            pretty_table(data; kwargs...)
            if latex
                pretty_table(data; kwargs..., backend=Val(:latex))
            end
        end
    end

    @info "Original Serre-Green-Naghdi equations (mild slope) with upwind SBP operators"
    let
        # first-order upwind operators yield a second-order central operator
        D = upwind_operators(periodic_derivative_operator;
                             derivative_order = 1,
                             accuracy_order = 1,
                             xmin, xmax, N)
        dts = [0.15, 0.05, 0.02, 0.01, 0.005]
        analyze_conservation(rhs_serre_green_naghdi_mild!,
                             D, dts, b_func)
    end

    orders = [2, 4]
    for order in orders
        @info "Original Serre-Green-Naghdi equations (mild slope) plus viscosity with upwind SBP operators" order
        let
            # first-order upwind operators yield a second-order central operator
            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1,
                                 accuracy_order = order-1,
                                xmin, xmax, N)
            dts = [0.15, 0.05, 0.02, 0.01, 0.005]
            errors_energy = Float64[]
            errors_water_mass = Float64[]
            mu = ( xmax - xmin )/N
            mu = (mu^order)/order

            for dt in dts
                @time (; energy, water_mass) = solve_conservation_visc(
                    rhs_serre_green_naghdi_mild_visc!, D; mu,
                    b_func = x -> 0.25 * cospi(x / 75),
                    dt, adaptive = false
                )
                error_energy = maximum(abs.(energy .- energy[begin]))
                error_water_mass = maximum(abs.(water_mass .- water_mass[begin]))
                push!(errors_energy, error_energy)
                push!(errors_water_mass, error_water_mass)
            end

            eoc_energy = -compute_eoc(dts, errors_energy)

            data = hcat(dts, errors_water_mass, errors_energy, eoc_energy)
            header = ["dt", "error water mass", "error energy", "EOC energy"]
            kwargs = (; header, formatters=(ft_printf("%.3f", [1]),
                                            ft_printf("%.2e", [2, 3]),
                                            ft_printf("%.2f", [4])))
            pretty_table(data; kwargs...)
            if latex
                pretty_table(data; kwargs..., backend=Val(:latex))
            end
        end
    end

    @info "Original Serre-Green-Naghdi equations (full bathymetry) with central SBP operators"
    let
        D = periodic_derivative_operator(derivative_order = 1;
                                         accuracy_order = 2,
                                         xmin, xmax, N)
        dts = [0.15, 0.05, 0.02, 0.01, 0.005]
        analyze_conservation(rhs_serre_green_naghdi_full!,
                             D, dts, b_func)
    end

    orders = [2, 4]
    for order in orders
        @info "Original Serre-Green-Naghdi equations (full bathymetry) plus viscosity with central SBP operators" order
        let
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order = order,
                                             xmin, xmax, N)
            dts = [0.15, 0.05, 0.02, 0.01, 0.005]
            errors_energy = Float64[]
            errors_water_mass = Float64[]
            mu = ( xmax - xmin )/N
            mu = (mu^order)/order

            for dt in dts
                @time (; energy, water_mass) = solve_conservation_visc(
                    rhs_serre_green_naghdi_full_visc!, D; mu,
                    b_func = x -> 0.25 * cospi(x / 75),
                    dt, adaptive = false
                )
                error_energy = maximum(abs.(energy .- energy[begin]))
                error_water_mass = maximum(abs.(water_mass .- water_mass[begin]))
                push!(errors_energy, error_energy)
                push!(errors_water_mass, error_water_mass)
            end

            eoc_energy = -compute_eoc(dts, errors_energy)

            data = hcat(dts, errors_water_mass, errors_energy, eoc_energy)
            header = ["dt", "error water mass", "error energy", "EOC energy"]
            kwargs = (; header, formatters=(ft_printf("%.3f", [1]),
                                            ft_printf("%.2e", [2, 3]),
                                            ft_printf("%.2f", [4])))
            pretty_table(data; kwargs...)
            if latex
                pretty_table(data; kwargs..., backend=Val(:latex))
            end
        end
    end

    @info "Original Serre-Green-Naghdi equations (full bathymetry) with upwind SBP operators"
    let
        # first-order upwind operators yield a second-order central operator
        D = upwind_operators(periodic_derivative_operator;
                             derivative_order = 1,
                             accuracy_order = 1,
                             xmin, xmax, N)
        dts = [0.15, 0.05, 0.02, 0.01, 0.005]
        analyze_conservation(rhs_serre_green_naghdi_full!,
                             D, dts, b_func)
    end

    orders = [2, 4]
    for order in orders
        @info "Original Serre-Green-Naghdi equations (full bathymetry) plus viscosity with upwind SBP operators" order
        let
        # first-order upwind operators yield a second-order central operator
            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1,
                                 accuracy_order = order-1,
                                 xmin, xmax, N)
            dts = [0.15, 0.05, 0.02, 0.01, 0.005]
            errors_energy = Float64[]
            errors_water_mass = Float64[]
            mu = ( xmax - xmin )/N
            mu = (mu^order)/order

            for dt in dts
                @time (; energy, water_mass) = solve_conservation_visc(
                    rhs_serre_green_naghdi_full_visc!, D;mu,
                    b_func = x -> 0.25 * cospi(x / 75),
                    dt, adaptive = false
                )
                error_energy = maximum(abs.(energy .- energy[begin]))
                error_water_mass = maximum(abs.(water_mass .- water_mass[begin]))
                push!(errors_energy, error_energy)
                push!(errors_water_mass, error_water_mass)
            end

            eoc_energy = -compute_eoc(dts, errors_energy)

            data = hcat(dts, errors_water_mass, errors_energy, eoc_energy)
            header = ["dt", "error water mass", "error energy", "EOC energy"]
            kwargs = (; header, formatters=(ft_printf("%.3f", [1]),
                                            ft_printf("%.2e", [2, 3]),
                                            ft_printf("%.2f", [4])))
            pretty_table(data; kwargs...)
            if latex
                pretty_table(data; kwargs..., backend=Val(:latex))
            end
        end
    end

    return nothing
end

function solve_conservation(rhs!, D;
                            tspan = (0.0, 35.0),
                            g = 9.81, λ = 500.0,
                            alg = Tsit5(),
                            b_func = zero,
                            kwargs...)
    # setup initial data
    x = grid(D)
    h_func(x) = 1 + exp(-x^2) - b_func(x)
    v_func(x) = 1.0e-2

    if alg === nothing
        if rhs! === rhs_nonconservative!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
    end

    u0, parameters = setup(rhs!,
                           h_func, v_func, b_func;
                           g, λ, D)


    saved_values = SavedValues(Float64, Tuple{Float64, Float64, Float64})
    saving = SavingCallback(saved_values) do u, t, integrator
        parameters = integrator.p
        x = grid(parameters.D)

        total_energy = energy(rhs!, u, parameters)

        total_water_mass = integrate(u.x[1], parameters.D)

        total_momentum = integrate(u.x[1] .* u.x[2], parameters.D)

        return (total_energy, total_water_mass, total_momentum)
    end

    ode = ODEProblem(rhs!, u0, tspan, parameters)

    sol = solve(ode, alg;
                save_everystep = false,
                callback = saving,
                kwargs...)

    return (; sol,
              t = saved_values.t,
              energy = map(x -> x[1], saved_values.saveval),
              water_mass = map(x -> x[2], saved_values.saveval),
              momentum = map(x -> x[3], saved_values.saveval))
end


function solve_conservation_visc(rhs!, D;
                                 tspan = (0.0, 35.0),
                                 g = 9.81, mu, λ = 500.0,
                                 alg = Tsit5(),
                                 b_func = zero,
                                 kwargs...)
    # setup initial data
    x = grid(D)
    h_func(x) = 1 + exp(-x^2) - b_func(x)
    v_func(x) = 1.0e-2

    if alg === nothing
        if rhs! === rhs_nonconservative_visc!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
    end

    u0, parameters = setup(rhs!,
    h_func, v_func, b_func;
    g, mu, λ, D)


    saved_values = SavedValues(Float64, Tuple{Float64, Float64, Float64})
    saving = SavingCallback(saved_values) do u, t, integrator
        parameters = integrator.p
        x = grid(parameters.D)

        total_energy = energy(rhs!, u, parameters)

        total_water_mass = integrate(u.x[1], parameters.D)

        total_momentum = integrate(u.x[1] .* u.x[2], parameters.D)

        return (total_energy, total_water_mass, total_momentum)
    end

    ode = ODEProblem(rhs!, u0, tspan, parameters)

    sol = solve(ode, alg;
                save_everystep = false,
                callback = saving,
                kwargs...)

    return (; sol,
              t = saved_values.t,
              energy = map(x -> x[1], saved_values.saveval),
              water_mass = map(x -> x[2], saved_values.saveval),
              momentum = map(x -> x[3], saved_values.saveval))
end



#####################################################################
# Well-balancedness
function check_well_balancedness(; latex = true)
    g = 9.81


    for (rhs!, λ, name, operator) in (
            (rhs_nonconservative!, 500.0, "hyperbolic", :central),
            (rhs_nonconservative!, 5.0e3, "hyperbolic", :central),
            (rhs_serre_green_naghdi_mild!, 0.0, "SGN, mild slope", :central),
            (rhs_serre_green_naghdi_mild!, 0.0, "SGN, mild slope", :upwind),
            (rhs_serre_green_naghdi_full!, 0.0, "SGN, full", :central),
            (rhs_serre_green_naghdi_full!, 0.0, "SGN, full", :upwind))

        if !iszero(λ)
            @info "System" name λ operator
        else
            @info "System" name operator
        end
        data = Vector{Float64}()
        for accuracy_order in 2:2:6
            push!(data, accuracy_order)
            if operator === :central
                D = periodic_derivative_operator(derivative_order = 1;
                                                 accuracy_order,
                                                 xmin = -150.0,
                                                 xmax = 150.0,
                                                 N = 1000)
            else
                D = upwind_operators(periodic_derivative_operator;
                                     derivative_order = 1,
                                     accuracy_order,
                                     xmin = -150.0,
                                     xmax = 150.0,
                                     N = 1000)
            end
            # setup initial data
            x = grid(D)
            b_func(x) = 0.25 * cospi(x / 75)
            h_func(x) = 1 - b_func(x)
            v_func(x) = 0.0
            u0, parameters = setup(rhs!,
                                   h_func, v_func, b_func;
                                   g, λ, D)
            du = similar(u0)
            rhs!(du, u0, parameters, 0.0)

            for i in eachindex(du.x)
                push!(data, integrate(abs2, du.x[i], D) |> sqrt)
            end
        end

        data = reshape(data, :, 3)'

        if size(data, 2) == 3
            header = ["order", "h", "v"]
        else
            header = ["order", "h", "v", "w", "η"]
        end
        kwargs = (; header, formatters=(ft_printf("%1d", [1]),
                                        ft_printf("%.1e", 2:size(data, 2))))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    for (rhs!, λ, name, operator) in (
        (rhs_nonconservative_visc!, 5.0e2, "hyperbolic plus viscosity", :central),
        (rhs_nonconservative_visc!, 5.0e3, "hyperbolic plus viscosity", :central),
        (rhs_serre_green_naghdi_mild_visc!, 0.0, "SGN, mild slope plus viscosity", :central),
        (rhs_serre_green_naghdi_mild_visc!, 0.0, "SGN, mild slope plus viscosity", :upwind),
        (rhs_serre_green_naghdi_full_visc!, 0.0, "SGN, full plus viscosity", :central),
        (rhs_serre_green_naghdi_full_visc!, 0.0, "SGN, full plus viscosity", :upwind))

        if !iszero(λ)
            @info "System" name λ operator
        else
            @info "System" name operator
        end
        data = Vector{Float64}()
        for accuracy_order in 2:2:6
            push!(data, accuracy_order)
            if operator === :central
                D = periodic_derivative_operator(derivative_order = 1;
                                                 accuracy_order,
                                                 xmin = -150.0,
                                                 xmax = 150.0,
                                                 N = 1000)
            else
                D = upwind_operators(periodic_derivative_operator;
                                     derivative_order = 1,
                                     accuracy_order,
                                     xmin = -150.0,
                                     xmax = 150.0,
                                     N = 1000)
            end
            # setup initial data
            x = grid(D)
            b_func(x) = 0.25 * cospi(x / 75)
            h_func(x) = 1 - b_func(x)
            v_func(x) = 0.0
            xmin = -150.0
            xmax = 150.0
            N = 1000
            mu = ( xmax - xmin )/N
            mu = ( mu^accuracy_order )/accuracy_order

            u0, parameters = setup(rhs!,
                                   h_func, v_func, b_func;
                                   g, mu, λ, D)
            du = similar(u0)
            rhs!(du, u0, parameters, 0.0)

            for i in eachindex(du.x)
                push!(data, integrate(abs2, du.x[i], D) |> sqrt)
            end
        end

        data = reshape(data, :, 3)'

        if size(data, 2) == 3
            header = ["order", "h", "v"]
        else
            header = ["order", "h", "v", "w", "η"]
        end
        kwargs = (; header, formatters=(ft_printf("%1d", [1]),
                                        ft_printf("%.1e", 2:size(data, 2))))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    return nothing
end



#####################################################################
# Error growth with and without relaxation
function plot_error_growth()
    D = fourier_derivative_operator(xmin = -50.0, xmax = 50.0, N = 2^7)

    g = 9.81
    alg = Tsit5()
    abstol = 1.0e-5
    reltol = 1.0e-5
    domain_traversals = 20

    # setup initial data
    x = grid(D)
    xmin = SummationByPartsOperators.xmin(D)
    xmax = SummationByPartsOperators.xmax(D)
    h1 = 1.0
    h2 = 1.2
    c = sqrt(g * h2)

    function h_analytical(t, x)
        x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
        return h1 + (h2 - h1) * sech(x_t / 2 * sqrt(3 * (h2 - h1) / (h1^2 * h2)))^2
    end

    function v_analytical(t, x)
        return c * (1 - h1 / h_analytical(t, x))
    end

    h_func = x -> h_analytical(0.0, x)
    v_func = x -> v_analytical(0.0, x)
    b_func = zero

    u0, parameters = setup(rhs_serre_green_naghdi_flat!,
                           h_func, v_func, b_func;
                           g, λ = 0, D)

    tspan = (0.0, domain_traversals * (xmax - xmin) / c)
    ode = ODEProblem(rhs_serre_green_naghdi_flat!,
                     u0, tspan, parameters)

    for relaxation in (false, true)
        saved_values = SavedValues(Float64, Tuple{Float64, Float64, Float64})
        saving = SavingCallback(saved_values) do u, t, integrator
            parameters = integrator.p
            x = grid(parameters.D)

            h_diff = parameters.tmp
            @.. h_diff = h_analytical(t, x) - u.x[1]
            error_h = sqrt(integrate(abs2, h_diff, parameters.D))

            v_diff = parameters.tmp
            @.. v_diff = v_analytical(t, x) - u.x[2]
            error_v = sqrt(integrate(abs2, v_diff, parameters.D))

            total_energy = energy(rhs_serre_green_naghdi_flat!, u, parameters)

            return (total_energy, error_h, error_v)
        end

        if relaxation
            initial_energy = energy(rhs_serre_green_naghdi_flat!, u0, parameters)
            relaxation_callback = DiscreteCallback(
                (u, t, integrator) -> true,
                Base.Fix2(relaxation!, initial_energy),
                save_positions = (false, false))

            callback = CallbackSet(relaxation_callback, saving)
        else
            callback = saving
        end

        @time sol = solve(ode, alg;
                          save_everystep = false,
                          abstol, reltol, callback)

        let
            t = saved_values.t
            energy = map(x -> x[1], saved_values.saveval)
            error_h = map(x -> x[2], saved_values.saveval)
            error_v = map(x -> x[3], saved_values.saveval)

            label = relaxation ? "relaxation" : "baseline"

            fig_error = plot(; xguide = L"t", yguide = L"Discrete $L^2$ error",
                             margin = 0.3*Plots.cm, plot_kwargs()...)

            plot!(fig_error, t, error_h;
                  label = "water height", linestyle = :solid,
                  plot_kwargs()...)
            plot!(fig_error, t, error_v;
                  label = "velocity", linestyle = :dash,
                  plot_kwargs()...)

            savefig(fig_error, joinpath(figdir, "error_growth_error_$(label).pdf"))


            fig_energy = plot(; xguide = L"t", yguide = "Energy error",
                              margin = 0.3*Plots.cm, plot_kwargs()...)

            plot!(fig_energy, t[1:end-1], energy[1:end-1] .- energy[1];
                  label = "",
                  plot_kwargs()...)

            savefig(fig_energy, joinpath(figdir, "error_growth_energy_$(label).pdf"))
        end
    end

    @info "Results saved in the directory" figdir

    return nothing
end

function relaxation!(integrator::OrdinaryDiffEq.ODEIntegrator, initial_energy)
    told = integrator.tprev
    uold = integrator.uprev
    tnew = integrator.t
    unew = integrator.u
    utmp = first(get_tmp_cache(integrator))
    rhs! = integrator.f.f
    parameters = integrator.p

    next_tstop = top(integrator.opts.tstops)
    if !(tnew ≈ next_tstop)
        # initial_energy = energy(rhs!, uold, parameters)
        function residual(gamma, _)
            @.. utmp = uold + gamma * (unew - uold)
            return energy(rhs!, utmp, parameters) - initial_energy
        end
        bounds = (0.9, 1.1)
        prob = IntervalNonlinearProblem(residual, bounds)
        sol = solve(prob, ITP())
        gamma = sol.u

        @.. unew = uold + gamma * (unew - uold)
        tγ = told + gamma * (tnew - told)

        # We should not step past the final time
        if (tγ > next_tstop || tγ ≈ next_tstop)
            tγ = next_tstop
        end
        set_t!(integrator, tγ)
        set_u!(integrator, unew)
        u_modified!(integrator, true)
    else
        u_modified!(integrator, false)
    end

    return nothing
end



#####################################################################
# Riemann problem leading to a dispersive shock wave
function plot_riemann_problem()
    D = periodic_derivative_operator(derivative_order = 1,
                                     accuracy_order = 2,
                                     xmin = -600.0, xmax = 600.0,
                                     N = 4_000)

    g = 9.81
    λ = 500.0
    abstol = 1.0e-5
    reltol = 1.0e-5
    tspan = (0.0, 47.434)

    # setup initial data
    x = grid(D)
    Δx = step(x)
    α = 2.0
    hL = 1.8
    hR = 1.0
    h_func(x) = hR + (hL - hR) / 2 * (1 - tanh(x  / α))
    v_func(x) = 0.0
    b_func = zero

    for (rhs!, name) in [(rhs_nonconservative!, "hyperbolic"),
                         (rhs_serre_green_naghdi_flat!, "original")]
        u0, parameters = setup(rhs!,
                               h_func, v_func, b_func;
                               g, λ, D)

        ode = ODEProblem(rhs!, u0, tspan, parameters)

        if rhs! == rhs_nonconservative!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        @info "Running" name Δx
        @time sol = solve(ode, alg;
                          save_everystep = false,
                          abstol, reltol)

        x = grid(D)
        idx = @. -300 <= x <= 300
        x = x[idx]

        fig_h = plot(; xguide = L"x", yguide = L"h", plot_kwargs()...)
        h0 = sol.u[begin].x[1][idx]
        h = sol.u[end].x[1][idx]
        plot!(fig_h, x, h0;
              label = L"h^0", color = :gray, linestyle = :dash, plot_kwargs()...)
        plot!(fig_h, x, h;
              label = "\$h\$ ($name)", color = 1, linestyle = :solid, plot_kwargs()...)

        x_annotate = [50.0, 300.0]
        plot!(fig_h, x_annotate, 1.37 * ones(2);
              label = "", color = :gray, linestyle = :dot, plot_kwargs()...)
        annotate!(fig_h, 260, 1.41, text(L"h^* = 1.37"))
        plot!(fig_h, x_annotate, 1.74 * ones(2);
            label = "", color = :gray, linestyle = :dot, plot_kwargs()...)
        annotate!(fig_h, 260, 1.70, text(L"h^m = 1.74"))

        savefig(fig_h, joinpath(figdir, "riemann_problem_$(name).pdf"))
    end

    @info "Results saved in the directory" figdir

    return nothing
end


function plot_riemann_problem_visc()
    D = periodic_derivative_operator(derivative_order = 1,
                                     accuracy_order = 2,
                                     xmin = -600.0, xmax = 600.0,
                                     N = 4_000)

    g = 9.81
    λ = 500.0
    abstol = 1.0e-5
    reltol = 1.0e-5
    tspan = (0.0, 47.434)
    order = 2
    xmin = -600.0
    xmax = 600.0
    N = 4_000
    mu = (xmax-xmin)/N
    mu = (mu^order)/order

    # setup initial data
    x = grid(D)
    α = 2.0
    hL = 1.8
    hR = 1.0
    h_func(x) = hR + (hL - hR) / 2 * (1 - tanh(x  / α))
    v_func(x) = 0.0
    b_func = zero

    for (rhs!, name) in [(rhs_nonconservative_visc!, "hyperbolic"),
                         (rhs_serre_green_naghdi_flat_visc!, "original")]
        u0, parameters = setup(rhs!,
                               h_func, v_func, b_func;
                               g, mu, λ, D)

        ode = ODEProblem(rhs!, u0, tspan, parameters)

        if rhs! == rhs_nonconservative_visc!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        @time sol = solve(ode, alg;
                          save_everystep = false,
                          abstol, reltol)

        x = grid(D)
        idx = @. -300 <= x <= 300
        x = x[idx]

        fig_h = plot(; xguide = L"x", yguide = L"h", plot_kwargs()...)
        h0 = sol.u[begin].x[1][idx]
        h = sol.u[end].x[1][idx]
        plot!(fig_h, x, h0;
              label = L"h^0", color = :gray, linestyle = :dash, plot_kwargs()...)
        plot!(fig_h, x, h;
              label = "\$h\$ ($name)", color = 1, linestyle = :solid, plot_kwargs()...)

        x_annotate = [50.0, 300.0]
        plot!(fig_h, x_annotate, 1.37 * ones(2);
              label = "", color = :gray, linestyle = :dot, plot_kwargs()...)
        annotate!(fig_h, 260, 1.41, text(L"h^* = 1.37"))
        plot!(fig_h, x_annotate, 1.74 * ones(2);
            label = "", color = :gray, linestyle = :dot, plot_kwargs()...)
        annotate!(fig_h, 260, 1.70, text(L"h^m = 1.74"))

        savefig(fig_h, joinpath(figdir, "riemann_problem_visc_$(name).pdf"))
    end

    @info "Results saved in the directory" figdir

    return nothing
end



#####################################################################
# Soliton fission

function plot_soliton_fission()
    D = periodic_derivative_operator(1, 2, -500.0, 500.0, 1_000)

    g = 9.81
    λ = 500.0
    abstol = 1.0e-5
    reltol = 1.0e-5
    tspan = (0.0, 118.0)

    # setup initial data
    x = grid(D)
    Δx = step(x)
    h_func(x) = ifelse(abs(x) < 10, 1.8, 1.0)
    v_func(x) = 0.0
    b_func = zero

    for (rhs!, name) in [(rhs_nonconservative!, "hyperbolic"),
                         (rhs_serre_green_naghdi_flat!, "original")]
        u0, parameters = setup(rhs!,
                               h_func, v_func, b_func;
                               g, λ, D)

        ode = ODEProblem(rhs!, u0, tspan, parameters)

        if rhs! === rhs_nonconservative!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end

        @info "Running" name Δx
        @time sol = solve(ode, alg;
                          save_everystep = false,
                          abstol, reltol)

        x = grid(D)
        h = sol.u[end].x[1]
        h0 = sol.u[begin].x[1]

        idx = @. 390 <= x <= 500
        x = x[idx]
        h = h[idx]
        h0 = h0[idx]

        threshold = 1.001
        waves = detect_waves(h, threshold)

        h_base = let
            idx = setdiff(waves[1][1]:length(h), waves...)
            median(h[idx])
        end

        fig_solution = plot(x, h;
                            xguide = L"x", yguide = L"h",
                            label = "numerical solution", plot_kwargs()...)
        solitons = zero(h)
        for (i, wave) in enumerate(reverse(waves))
            x_wave = x[wave]
            h_wave = h[wave]
            h_soliton = fit_soliton(x_wave, h_wave; h_base)
            @.. solitons += h_soliton(x) - h_base

            label = i == 1 ? "soliton fit" : ""
            plot!(fig_solution, x_wave, h_soliton.(x_wave);
                    label, color = :gray, linestyle = :dot, plot_kwargs()...)
        end
        savefig(fig_solution,
                joinpath(figdir, "soliton_fission_solution_$(name).pdf"))

        fig_differences = plot(x, solitons .- (h .- h_base);
                                xguide = L"x", yguide = "error of fit",
                                label = "", plot_kwargs()...)
        savefig(fig_differences,
                joinpath(figdir, "soliton_fission_differences_$(name).pdf"))
    end

    @info "Results saved in the directory" figdir

    return nothing
end

function plot_soliton_fission_visc()
    D = periodic_derivative_operator(1, 2, -500.0, 500.0, 1_000)

    g = 9.81
    λ = 500.0
    abstol = 1.0e-5
    reltol = 1.0e-5
    tspan = (0.0, 118.0)
    xmax = 500.
    xmin = -500.
    N = 1_000
    order = 2
    mu = ( xmax - xmin )/N
    mu = ( mu^order )/order

    # setup initial data
    x = grid(D)
    Δx = step(x)
    h_func(x) = ifelse(abs(x) < 10, 1.8, 1.0)
    v_func(x) = 0.0
    b_func = zero

    for (rhs!, name) in [(rhs_nonconservative_visc!, "hyperbolic"),
                         (rhs_serre_green_naghdi_flat_visc!, "original")]
        u0, parameters = setup(rhs!,
                               h_func, v_func, b_func;
                               g, mu, λ, D)

        ode = ODEProblem(rhs!, u0, tspan, parameters)

        if rhs! === rhs_nonconservative_visc!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end

        @info "Running" name Δx
        @time sol = solve(ode, alg;
                          save_everystep = false,
                          abstol, reltol)

        x = grid(D)
        h = sol.u[end].x[1]
        h0 = sol.u[begin].x[1]

        idx = @. 390 <= x <= 500
        x = x[idx]
        h = h[idx]
        h0 = h0[idx]

        threshold = 1.001
        waves = detect_waves(h, threshold)

        h_base = let
            idx = setdiff(waves[1][1]:length(h), waves...)
            median(h[idx])
        end

        fig_solution = plot(x, h;
                            xguide = L"x", yguide = L"h",
                            label = "numerical solution", plot_kwargs()...)
        solitons = zero(h)
        for (i, wave) in enumerate(reverse(waves))
            x_wave = x[wave]
            h_wave = h[wave]
            h_soliton = fit_soliton(x_wave, h_wave; h_base)
            @.. solitons += h_soliton(x) - h_base

            label = i == 1 ? "soliton fit" : ""
            plot!(fig_solution, x_wave, h_soliton.(x_wave);
                    label, color = :gray, linestyle = :dot, plot_kwargs()...)
        end
        savefig(fig_solution,
                joinpath(figdir, "soliton_fission_visc_solution_$(name).pdf"))

        fig_differences = plot(x, solitons .- (h .- h_base);
                                xguide = L"x", yguide = "error of fit",
                                label = "", plot_kwargs()...)
        savefig(fig_differences,
                joinpath(figdir, "soliton_fission_visc_differences_$(name).pdf"))
    end

    @info "Results saved in the directory" figdir

    return nothing
end

function detect_waves(h, threshold)
    idx = h .> threshold

    idx1 = findfirst(idx)
    wave_indices = Vector{UnitRange{Int}}()

    while idx1 !== nothing
        idx2 = findnext(!, idx, idx1) - 1
        idx2 === nothing && break
        push!(wave_indices, idx1:idx2)

        idx1 = findnext(idx, idx2 + 1)
    end

    return wave_indices
end

function fit_soliton(x, h; h_base)
    h1 = h_base
    idx_max = argmax(h)
    h2 = h[idx_max]
    x0 = x[idx_max]

    function h_analytical(x; h1, h2, x0)
        return h1 + (h2 - h1) * sech((x - x0) / 2 * Base.sqrt_llvm(3 * (h2 - h1) / (h1^2 * h2)))^2
    end

    prob = OptimizationProblem([h2, x0]) do h2_x0, parameters
        h2, x0 = h2_x0
        return sum(abs2, h - h_analytical.(x; h1, h2, x0))
    end
    sol = solve(prob, NelderMead())
    h2, x0 = sol.u

    return x -> h_analytical(x; h1, h2, x0)
end



#####################################################################
# Favre waves aka undular bores

function plot_favre_waves_solutions()
    g = 9.81
    λ = 500.0
    abstol = 1.0e-5
    reltol = 1.0e-5

    h0 = 0.2
    α = 5 * h0

    for (rhs!, name) in [(rhs_nonconservative!, "hyperbolic"),
                         (rhs_serre_green_naghdi_flat!, "original")]
        xmin = -50.0
        xmax = 50.0
        N = 800
        if rhs! === rhs_nonconservative!
            D = periodic_derivative_operator(derivative_order = 1,
                                             accuracy_order = 4;
                                             xmin, xmax, N)
        else
            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1, accuracy_order = 3,
                                 xmin, xmax, N)
        end
        for ε in (0.1, 0.2, 0.3)
            x = grid(D)
            Δx = step(x)
            @info "Running" name ε Δx

            # setup initial data
            x0 = 0.0
            jump_h = ε * h0
            h1 = h0 + jump_h
            v0 = 0.0
            jump_v = sqrt(g * (h1 + h0) / (2 * h0 * h1)) * jump_h
            h_func(x) = h0 + 0.5 * jump_h * (1 - tanh((x - x0) / α))
            v_func(x) = v0 + 0.5 * jump_v * (1 - tanh((x - x0) / α))
            b_func = zero

            u0, parameters = setup(rhs!,
                                   h_func, v_func, b_func;
                                   g, λ, D)

            # create and solve ODE
            if rhs! === rhs_nonconservative!
                alg = RDPK3SpFSAL35()
            else
                alg = Tsit5()
            end
            saveat = (0:10:70) .* sqrt(h0 / g) # non-dimensional time
            ode = ODEProblem(rhs!, u0, extrema(saveat), parameters)

            @time sol = solve(ode, alg;
                              save_everystep = false, saveat, abstol, reltol)

            fig = plot(xguide = L"x / h_0", yguide = L"(h - h_0) / h_0")

            for (i, t) in enumerate((70, 60, 50))
                plot!(fig, x / h0, (sol.u[end - i + 1].x[1] .- h0) ./ h0;
                      label = "", color = i, plot_kwargs()...)
                data = readdlm(joinpath(@__DIR__, "Wei_data",
                                        "dh$(ε)t$(t)_FNPF.txt"))
                plot!(fig, data[:, 1], data[:, 2];
                     label = "", color = i, linestyle = :dot, plot_kwargs()...)

                idx = argmax(data[:, 2])
                annotate!(fig, data[idx, 1], data[idx, 2],
                          text("\$\\tilde t = $t\$", :center, :bottom))
            end

            ylims = Plots.ylims(fig)
            plot!(fig; xlims = (45, 90), ylims = (-0.01, ylims[2]),
                  plot_kwargs()...)
            savefig(fig,
                    joinpath(figdir, "favre_waves_eps$(round(Int, 10ε))_$(name).pdf"))
        end
    end

    @info "Results saved in the directory" figdir
end

function plot_favre_waves_solutions_visc()
    g = 9.81
    λ = 500.0
    abstol = 1.0e-5
    reltol = 1.0e-5

    h0 = 0.2
    α = 5 * h0

    for (rhs!, name) in [(rhs_nonconservative_visc!, "hyperbolic_visc"),
                         (rhs_serre_green_naghdi_flat_visc!, "original_visc")]
        xmin = -50.0
        xmax = 50.0
        N = 800
        order = 4

        if rhs! === rhs_nonconservative_visc!
            D = periodic_derivative_operator(derivative_order = 1,
                                             accuracy_order = 4;
                                             xmin, xmax, N)
        else
            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1, accuracy_order = 3,
                                 xmin, xmax, N)
        end
        for ε in (0.1, 0.2, 0.3)
            x = grid(D)
            Δx = step(x)
            @info "Running" name ε Δx

            # setup initial data
            x0 = 0.0
            jump_h = ε * h0
            h1 = h0 + jump_h
            v0 = 0.0
            jump_v = sqrt(g * (h1 + h0) / (2 * h0 * h1)) * jump_h
            h_func(x) = h0 + 0.5 * jump_h * (1 - tanh((x - x0) / α))
            v_func(x) = v0 + 0.5 * jump_v * (1 - tanh((x - x0) / α))
            b_func = zero

            mu = ( xmax - xmin )/N
            mu = (mu^order)/order


            u0, parameters = setup(rhs!,
                                   h_func, v_func, b_func;
                                   g, mu, λ, D)

            # create and solve ODE
            if rhs! === rhs_nonconservative_visc!
                alg = RDPK3SpFSAL35()
            else
                alg = Tsit5()
            end
            saveat = (0:10:70) .* sqrt(h0 / g) # non-dimensional time
            ode = ODEProblem(rhs!, u0, extrema(saveat), parameters)

            @time sol = solve(ode, alg;
                              save_everystep = false, saveat, abstol, reltol)

            fig = plot(xguide = L"x / h_0", yguide = L"(h - h_0) / h_0")

            for (i, t) in enumerate((70, 60, 50))
                plot!(fig, x / h0, (sol.u[end - i + 1].x[1] .- h0) ./ h0;
                      label = "", color = i, plot_kwargs()...)
                data = readdlm(joinpath(@__DIR__, "Wei_data",
                                        "dh$(ε)t$(t)_FNPF.txt"))
                plot!(fig, data[:, 1], data[:, 2];
                     label = "", color = i, linestyle = :dot, plot_kwargs()...)

                idx = argmax(data[:, 2])
                annotate!(fig, data[idx, 1], data[idx, 2],
                          text("\$\\tilde t = $t\$", :center, :bottom))
            end

            ylims = Plots.ylims(fig)
            plot!(fig; xlims = (45, 90), ylims = (-0.01, ylims[2]),
                  plot_kwargs()...)
            savefig(fig,
                    joinpath(figdir, "favre_waves_eps$(round(Int, 10ε))_$(name).pdf"))
        end
    end

    @info "Results saved in the directory" figdir
end



function plot_favre_waves_amplitudes_over_time()
    g = 9.81
    λ = 500.
    abstol = 1.0e-5
    reltol = 1.0e-5

    h0 = 1.


    file_num = 0
    dx = 0

    for (rhs!, name) in [#(rhs_nonconservative!, "hyperbolic"),
                         (rhs_serre_green_naghdi_flat!, "original")]
        xmin = -3000.0
        xmax = +3000.0
        N_values = ( 12_000, 24_000 )
        tend = 1500.0

        ε_values = (0.1,0.2,0.3)

        for ε in ε_values, order in (2, 4)
            fig_h = plot(; xguide = L"x", yguide = L"h", plot_kwargs()...)
            fig = plot(xguide = L"\widetilde{t} = t \sqrt{g / h_0}", yguide = L"a_\mathrm{max} / h_0")

            for N in N_values
                if rhs! === rhs_nonconservative!
                    D = periodic_derivative_operator(derivative_order = 1,
                                                     accuracy_order = order;
                                                     xmin, xmax, N)
                else
                    D = upwind_operators(periodic_derivative_operator;
                                         derivative_order = 1, accuracy_order = order -1,
                                         xmin, xmax, N)
                end

                x = grid(D)

                # setup initial data
                delta_x0 = 2000.0
                x0 = -3000.0 + delta_x0
                jump_h = ε * h0
                h1 = h0 + jump_h
                α = 2.5*ε/0.1
                v0 = 0.0
                jump_v = sqrt(g * (h1 + h0) / (2 * h0 * h1)) * jump_h
                h_func(x) = h0 + 0.5 * jump_h * (1 - tanh((x - x0) / α)) #- 0.5 * jump_h * (1 - tanh((x - xmax + 10 ) / α))
                v_func(x) = v0 + 0.5 * jump_v * (1 - tanh((x - x0) / α)) #- 0.5 * jump_v * (1 - tanh((x - xmax + 10 ) / α))
                b_func = zero
                speed =  h1*jump_v/jump_h

                u0, parameters = setup(rhs!,
                                       h_func, v_func, b_func;
                                       g, λ, D)

                # create and solve ODE
                if rhs! === rhs_nonconservative!
                    alg = RDPK3SpFSAL35()
                else
                    alg = Tsit5()
                end
                tspan = (0.0, tend) .* sqrt(h0 / g) # non-dimensional time
                ode = ODEProblem(rhs!, u0, tspan, parameters)

                saved_values = SavedValues(Float64, Float64)
                saving_callback = SavingCallback(saved_values) do u, t, integrator
                    x = grid(integrator.p.D)
                    h = u.x[1]
                    idx = @. x0 + 25.0  <= x <= x0 + speed*tspan[end]  + 50.
                    h = @view h[idx]
                    return maximum(h)
                end

                color_value = Int(N/N_values[1])

                file_num = Int(10*ε)


                Δx = step(x)
                dx = Δx
                @info "running next" name ε Δx order N color_value

                @time sol = solve(ode, alg;
                                  save_everystep = false, callback = saving_callback,
                                  abstol, reltol)

                shift = sqrt(1.0*(file_num-1))*30.0
                idx = @. x0 + speed*tspan[end] - 120.   + shift  <= x <= x0 + speed*tspan[end] + 50. + shift
                h = sol.u[end].x[1]
                xv = @view x[idx]
                hv = @view h[idx]

                plot!(fig_h, xv, hv; label = "\$\\Delta x=$dx\$", color = color_value , linestyle = :solid, legend=:bottomleft, plot_kwargs()...)

                plot!(fig, saved_values.t * sqrt(g / h0),
                      (saved_values.saveval .- h0) ./ h0; color = color_value,
                    label = "\$\\Delta x=$dx\$",  legend=:bottomright, plot_kwargs()...)
            end

            savefig(fig_h, joinpath(figdir, "favre_wave_solution_eps$(file_num)_$(name)_O$order.pdf"))

            savefig(fig,
                joinpath(figdir, "favre_waves_amax_eps$(file_num)_$(name)_O$order.pdf"))
        end
    end

    @info "Results saved in the directory" figdir
end



function plot_favre_waves_amplitudes_over_time_visc()
    g = 9.81
    λ = 500.0
    abstol = 1.0e-5
    reltol = 1.0e-5

    file_num = 0
    dx = 0

    for (rhs!, name) in [#(rhs_nonconservative_visc!, "hyperbolic_visc"),
                         (rhs_serre_green_naghdi_flat_visc!, "original_visc")]
        xmin = -3000.0
        xmax = 3000.0
        N_values = (12_000, 24_000, 48_000 )
        tend = 1500.0
        ε_values =  (0.1,0.2,0.3)

        for ε in ε_values, order in (2, 4)
            fig_h = plot(; xguide = L"x", yguide = L"h", plot_kwargs()...)
            fig = plot(xguide = L"\widetilde{t} = t \sqrt{g / h_0}", yguide = L"a_\mathrm{max} / h_0")

            for N in N_values
                 if rhs! === rhs_nonconservative_visc!
                    D = periodic_derivative_operator(derivative_order = 1,
                                                     accuracy_order = order;
                                                     xmin, xmax, N)
                 else
                     D = upwind_operators(periodic_derivative_operator;
                                          derivative_order = 1,
                                          accuracy_order = order - 1,
                                          xmin, xmax, N)
                    @info "Upwind operator"
                 end

                 x = grid(D)

                # setup initial data
                delta_x0 = 2000.0
                x0 = -3000.0 + delta_x0
                h0 = 1.
                jump_h = ε * h0
                h1 = h0 + jump_h
                v0 = 0.0
                α = 2.5*ε/0.1
                jump_v = sqrt(g * (h1 + h0) / (2 * h0 * h1)) * jump_h
                h_func(x) = h0 + 0.5 * jump_h * (1 - tanh((x - x0) / α)) #- 0.5 * jump_h * (1 - tanh((x - xmax + 500. ) / (10*α) ))
                v_func(x) = v0 + 0.5 * jump_v * (1 - tanh((x - x0) / α)) #- 0.5 * jump_v * (1 - tanh((x - xmax + 500. ) / (10*α)))
                b_func = zero
                speed =  h1*jump_v/jump_h

                mu = ( xmax - xmin )/N
                mu = (mu^order)/order

                u0, parameters = setup(rhs!,
                                       h_func, v_func, b_func;
                                       g, mu, λ, D)

                # create and solve ODE
                if rhs! === rhs_nonconservative!
                    alg = RDPK3SpFSAL35()
                else
                    alg = Tsit5()
                end
                tspan = (0.0, tend ) .* sqrt(h0 / g) # non-dimensional time
                ode = ODEProblem(rhs!, u0, tspan, parameters)

                saved_values = SavedValues(Float64, Float64)
                saving_callback = SavingCallback(saved_values) do u, t, integrator
                    x = grid(integrator.p.D)
                    h = u.x[1]
                    idx = @. x0 + 25.0  <= x <= x0 + speed*tspan[end]  + 50.
                    h = @view h[idx]
                    return maximum(h)
                end

                color_value = Int(N/N_values[1])

                Δx = step(x)
                dx = Δx
                @info "running next" name ε Δx order mu speed*tspan[end] α


                file_num = Int(10*ε)

                @time sol = solve(ode, alg;
                                  save_everystep = false, callback = saving_callback,
                                  abstol, reltol)

                shift = sqrt(1.0*(file_num-1))*30.0
                idx = @. x0 + speed*tspan[end] - 120.   + shift  <= x <= x0 + speed*tspan[end] + 50. + shift
                h = sol.u[end].x[1]

                xv =  @view x[idx]
                hv =  @view h[idx]

                plot!(fig_h, xv, hv; label = "\$\\Delta x=$dx\$", color = color_value , linestyle = :solid, legend=:bottomleft, plot_kwargs()...)

                plot!(fig, saved_values.t * sqrt(g / h0),
                          (saved_values.saveval .- h0) ./ h0; color = color_value,
                        label = "\$\\Delta x=$dx\$",  legend=:bottomright, plot_kwargs()...)

            end

            savefig(fig_h, joinpath(figdir, "favre_wave_solution_eps$(file_num)_$(name)_O$order.pdf"))

            savefig(fig,
                    joinpath(figdir, "favre_waves_amax_eps$(file_num)_$(name)_O$order.pdf"))
        end
    end

    @info "Results saved in the directory" figdir
end

function plot_favre_waves_amplitudes_over_froude()
    g = 9.81
    λ = 500.0
    abstol = 1.0e-5
    reltol = 1.0e-5

    h0 = 0.2
    α = 5 * h0

    for (rhs!, name) in [(rhs_nonconservative!, "hyperbolic"),
                         (rhs_serre_green_naghdi_flat!, "original")]
        xmin = -150.0
        xmax = +150.0
        N = 15_000
        if rhs! === rhs_nonconservative!
            D = periodic_derivative_operator(derivative_order = 1,
                                             accuracy_order = 4;
                                             xmin, xmax, N)
        else
            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1, accuracy_order = 3,
                                 xmin, xmax, N)
        end

        froude = Vector{Float64}()
        a_max = Vector{Float64}()

        for ε in 0.02:0.04:0.3
            x = grid(D)

            # setup initial data
            x0 = 0.0
            jump_h = ε * h0
            h1 = h0 + jump_h
            v0 = 0.0
            jump_v = sqrt(g * (h1 + h0) / (2 * h0 * h1)) * jump_h
            h_func(x) = h0 + 0.5 * jump_h * (1 - tanh((x - x0) / α))
            v_func(x) = v0 + 0.5 * jump_v * (1 - tanh((x - x0) / α))
            b_func = zero

            u0, parameters = setup(rhs!,
                                   h_func, v_func, b_func;
                                   g, λ, D)

            # create and solve ODE
            if rhs! === rhs_nonconservative!
                alg = RDPK3SpFSAL35()
            else
                alg = Tsit5()
            end
            tspan = (0.0, 350.0) .* sqrt(h0 / g) # non-dimensional time
            ode = ODEProblem(rhs!, u0, tspan, parameters)

            callback = let idx = findfirst(>(63.5), x), threshold = h0 + 1.1 * jump_h
                DiscreteCallback(terminate!; save_positions = (true, false)) do u, t, integrator
                    h = u.x[1]
                    return h[idx] > threshold
                end
            end

            Δx = step(x)
            @info "running next" name ε Δx
            @time sol = solve(ode, alg;
                              save_everystep = false, abstol, reltol,
                              isoutofdomain = (u, p, t) -> any(isinf, u),
                              callback = callback)

            @show sol.t[end]
            h = sol.u[end].x[1]
            idx = @. 10 <= x <= 70
            h = @view h[idx]
            h_max = maximum(h)
            # Froude number σ / √(g h0)
            push!(froude, sqrt((1 + ε) * (1 + ε / 2)))
            push!(a_max, (h_max - h0) / h0)
        end

        fig = plot(; xguide = L"Froude number $\mathrm{Fr}$",
                     yguide = L"Max. amplitude $a_{\mathrm{max}} / h_0$",
                     plot_kwargs()...)

        for x in (100, 200)
            data = readdlm(joinpath(@__DIR__, "Favre_data/Favre_amplmax_$(x).txt"))
            plot!(data[:, 1], data[:, 2];
                  linecolor = :transparent, mark = :utriangle, markercolor = :transparent,
                  markerstrokecolor = :black, label = x == 100 ? "Favre (1935)" : "",
                  plot_kwargs()...)
        end

        for x in (80, 160)
            data = readdlm(joinpath(@__DIR__, "Favre_data/Treske_amplmax_$(x).txt"))
            plot!(data[:, 1], data[:, 2];
                  linecolor = :transparent, mark = :dtriangle, markercolor = :transparent,
                  markerstrokecolor = :black, label = x == 80 ? "Treske (1994)" : "",
                  plot_kwargs()...)
        end

        plot!(fig, froude, a_max;
              label = "numerical", color = 1, plot_kwargs()...)

        savefig(fig,
                joinpath(figdir, "favre_waves_amax_Froude_$(name).pdf"))
    end

    @info "Results saved in the directory" figdir
end


function plot_favre_waves_amplitudes_over_froude_visc()
    g = 9.81
    λ = 500.0
    abstol = 1.0e-5
    reltol = 1.0e-5

    h0 = 0.2
    α = 5 * h0

    for (rhs!, name) in [(rhs_nonconservative_visc!, "hyperbolic_visc"),
                         (rhs_serre_green_naghdi_flat_visc!, "original_visc")]
        xmin = -150.0
        xmax = +150.0
        N = 15_000
        order = 4
        mu = (xmax-xmin)/N
        mu=(h0*mu^order)/order

        if rhs! === rhs_nonconservative_visc!
            D = periodic_derivative_operator(derivative_order = 1,
                                             accuracy_order = 4;
                                             xmin, xmax, N)
        else
            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1, accuracy_order = 3,
                                 xmin, xmax, N)
        end

        froude = Vector{Float64}()
        a_max = Vector{Float64}()

        for ε in 0.02:0.04:0.3
            x = grid(D)

            # setup initial data
            x0 = 0.0
            jump_h = ε * h0
            h1 = h0 + jump_h
            v0 = 0.0
            jump_v = sqrt(g * (h1 + h0) / (2 * h0 * h1)) * jump_h
            h_func(x) = h0 + 0.5 * jump_h * (1 - tanh((x - x0) / α))
            v_func(x) = v0 + 0.5 * jump_v * (1 - tanh((x - x0) / α))
            b_func = zero

            u0, parameters = setup(rhs!,
                                   h_func, v_func, b_func;
                                   g, mu, λ, D)

            # create and solve ODE
            if rhs! === rhs_nonconservative!
                alg = RDPK3SpFSAL35()
            else
                alg = Tsit5()
            end
            tspan = (0.0, 350.0) .* sqrt(h0 / g) # non-dimensional time
            ode = ODEProblem(rhs!, u0, tspan, parameters)

            callback = let idx = findfirst(>(63.5), x), threshold = h0 + 1.1 * jump_h
                DiscreteCallback(terminate!; save_positions = (true, false)) do u, t, integrator
                    h = u.x[1]
                    return h[idx] > threshold
                end
            end

            Δx = step(x)
            @info "running next" name ε Δx
            @time sol = solve(ode, alg;
                              save_everystep = false, abstol, reltol,
                              isoutofdomain = (u, p, t) -> any(isinf, u),
                              callback = callback)

            @show sol.t[end]
            h = sol.u[end].x[1]
            idx = @. 10 <= x <= 70
            h = @view h[idx]
            h_max = maximum(h)
            # Froude number σ / √(g h0)
            push!(froude, sqrt((1 + ε) * (1 + ε / 2)))
            push!(a_max, (h_max - h0) / h0)
        end

        fig = plot(; xguide = L"Froude number $\mathrm{Fr}$",
                     yguide = L"Max. amplitude $a_{\mathrm{max}} / h_0$",
                     plot_kwargs()...)

        for x in (100, 200)
            data = readdlm(joinpath(@__DIR__, "Favre_data/Favre_amplmax_$(x).txt"))
            plot!(data[:, 1], data[:, 2];
                  linecolor = :transparent, mark = :utriangle, markercolor = :transparent,
                  markerstrokecolor = :black, label = x == 100 ? "Favre (1935)" : "",
                  plot_kwargs()...)
        end

        for x in (80, 160)
            data = readdlm(joinpath(@__DIR__, "Favre_data/Treske_amplmax_$(x).txt"))
            plot!(data[:, 1], data[:, 2];
                  linecolor = :transparent, mark = :dtriangle, markercolor = :transparent,
                  markerstrokecolor = :black, label = x == 80 ? "Treske (1994)" : "",
                  plot_kwargs()...)
        end

        plot!(fig, froude, a_max;
              label = "numerical", color = 1, plot_kwargs()...)

        savefig(fig,
                joinpath(figdir, "favre_waves_amax_Froude_$(name).pdf"))
    end

    @info "Results saved in the directory" figdir
end



#####################################################################
# Dingemans experiment

function plot_dingemans_solution(rhs! = rhs_serre_green_naghdi_mild!;
                                 λ = 500.0,
                                 tol = 1.0e-5,
                                 N = 1_000,
                                 tspan = (0.0, 40.0),
                                 offset = 2.4,
                                 accuracy_order = 4)
    g = 9.81
    abstol = tol
    reltol = tol

    xmin = -140.0
    xmax = 100.0

    if rhs! === rhs_nonconservative!
        D = periodic_derivative_operator(derivative_order = 1;
                                         accuracy_order,
                                         xmin, xmax, N)
    else
        D = upwind_operators(periodic_derivative_operator;
                             derivative_order = 1,
                             accuracy_order = accuracy_order - 1,
                             xmin, xmax, N)
    end

    # setup initial data
    h0 = 0.8
    A = 0.02
    # omega = 2*pi/(2.02*sqrt(2))
    k = 0.8406220896381442 # precomputed result of find_zero(k -> omega^2 - g * k * tanh(k * h0), 1.0) using Roots.jl
    h_plus_b_func(x) = begin
        if x < offset - 34.5 * pi / k || x > offset - 8.5 * pi / k
            h = h0
        else
            h = h0 + A * cos(k * x - offset)
        end
    end
    v_func(x) = sqrt(g / k * tanh(k * h0)) * (h_plus_b_func(x) - h0) / h0
    b_func(x) = begin
        if 11.01 <= x && x < 23.04
            b = 0.6 * (x - 11.01) / (23.04 - 11.01)
        elseif 23.04 <= x && x < 27.04
            b = 0.6
        elseif 27.04 <= x && x < 33.07
            b = 0.6 * (33.07 - x) / (33.07 - 27.04)
        else
            b = 0.0
        end
    end
    h_func(x) = h_plus_b_func(x) - b_func(x)

    u0, parameters = setup(rhs!,
                           h_func, v_func, b_func;
                           g, λ, D)

    # create and solve ODE
    if rhs! === rhs_nonconservative!
        alg = RDPK3SpFSAL35()
    else
        alg = Tsit5()
    end
    ode = ODEProblem(rhs!, u0, tspan, parameters)

    Δx = step(grid(D))
    @info "Running" Δx
    @time sol = solve(ode, alg;
                      save_everystep = false, abstol, reltol)

    fig = plot(; xguide = L"x", yguide = L"h + b", plot_kwargs()...)
    x = grid(sol.prob.p.D)
    b = sol.prob.p.b
    h0 = sol.u[begin].x[1]
    h = sol.u[end].x[1]
    plot!(fig, x, h0 + b; label = L"h^0 + b", color = :gray, plot_kwargs()...)
    plot!(fig, x, h + b; label = L"h + b", color = 1, plot_kwargs()...)
    plot!(fig, x, b; label = L"b", color = :gray, linestyle = :dot, plot_kwargs()...)
    plot!(fig; legend = :left)

    savefig(fig,
            joinpath(figdir, "dingemans_solution.pdf"))
    @info "Results saved in the directory" figdir

    return nothing
end


function plot_dingemans_solutions_at_gauges(; λ = 500.0,
                                              tol = 1.0e-5,
                                              N = 1_000,
                                              offset = 2.4,
                                              accuracy_order = 4)
    g = 9.81
    abstol = tol
    reltol = tol

    xmin = -140.0
    xmax = 100.0

    # setup initial data
    h0 = 0.8
    A = 0.02
    # omega = 2*pi/(2.02*sqrt(2))
    k = 0.8406220896381442 # precomputed result of find_zero(k -> omega^2 - g * k * tanh(k * h0), 1.0) using Roots.jl
    h_plus_b_func(x) = begin
        if x < offset - 34.5 * pi / k || x > offset - 8.5 * pi / k
            h = h0
        else
            h = h0 + A * cos(k * x - offset)
        end
    end
    v_func(x) = sqrt(g / k * tanh(k * h0)) * (h_plus_b_func(x) - h0) / h0
    b_func(x) = begin
        if 11.01 <= x && x < 23.04
            b = 0.6 * (x - 11.01) / (23.04 - 11.01)
        elseif 23.04 <= x && x < 27.04
            b = 0.6
        elseif 27.04 <= x && x < 33.07
            b = 0.6 * (33.07 - x) / (33.07 - 27.04)
        else
            b = 0.0
        end
    end
    h_func(x) = h_plus_b_func(x) - b_func(x)

    x_values = (3.04, 9.44, 20.04, 26.04, 30.44, 37.04)
    fig = plot(layout = (3, 2), size = (1_000, 800))
    ylim = (0.765, 0.865)
    tlims = [
        (20.0, 30.0),
        (25.0, 35.0),
        (35.0, 45.0),
        (40.0, 50.0),
        (45.0, 55.0),
        (50.0, 60.0),
    ]
    let
        experimental_data, header = readdlm(
            joinpath(@__DIR__, "Dingemans_data", "data_dingemans.csv"), ',';
            header = true)
        for (j, x) in enumerate(x_values)
            plot!(fig, experimental_data[:, 1], experimental_data[:, j + 1];
                  label = "experiment", color = :gray,
                  subplot = j, xlim = tlims[j], ylim = ylim,
                  legend = nothing, plot_kwargs()...)
            annotate!(fig, 0.5 * sum(tlims[j]), ylim[2],
                      text("\$x = $x\$", :center, :top),
                      subplot = j)
        end
    end

    for (i, (rhs!, label)) in enumerate([
            (rhs_nonconservative!, "hyperbolic"),
            (rhs_serre_green_naghdi_mild!, "mild slope"),
            (rhs_serre_green_naghdi_full!, "full system"),])

        if rhs! === rhs_nonconservative!
            D = periodic_derivative_operator(derivative_order = 1;
                                             accuracy_order ,
                                             xmin, xmax, N)
        else
            D = upwind_operators(periodic_derivative_operator;
                                 derivative_order = 1,
                                 accuracy_order = accuracy_order - 1,
                                 xmin, xmax, N)
        end

        Δx = step(grid(D))
        @info "running" label Δx

        u0, parameters = setup(rhs!,
                               h_func, v_func, b_func;
                               g, λ, D)

        # create and solve ODE
        if rhs! === rhs_nonconservative!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end
        tspan = (0.0, 70.0)
        ode = ODEProblem(rhs!, u0, tspan, parameters)

        # save water height at gauges
        saved_values = SavedValues(Float64, NTuple{6, Float64})
        saving_callback = SavingCallback(saved_values) do u, t, integrator
            x = grid(integrator.p.D)
            h = u.x[1]
            map(x_values) do xi
                idx = findfirst(>(xi), x)
                x_p = x[idx]
                x_m = x[idx - 1]
                factor = (xi - x_m) / (x_p - x_m)
                h_p = h[idx] + integrator.p.b[idx]
                h_m = h[idx - 1] + integrator.p.b[idx - 1]
                return factor * h_p + (1 - factor) * h_m
            end
        end

        @time sol = solve(ode, alg;
                          save_everystep = false, abstol, reltol,
                          callback = saving_callback)

        for j in eachindex(x_values)
            plot!(fig, saved_values.t, getindex.(saved_values.saveval, j);
                label = label, color = i,
                subplot = j,
                plot_kwargs()...)
        end
    end

    # adjust last x tick label to avoid overlap
    for j in eachindex(x_values)
        ticks, labels = Plots.xticks(fig)[j]
        labels[end] = first(split(labels[end], "."))
        Plots.xticks!(fig, ticks, labels, subplot = j)
    end

    plot!(fig; subplot = 5, legend = (-0.1, -0.9),
          legend_column = 4, bottom_margin = 13 * Plots.mm,
          xguide = L"Time $t$",
          plot_kwargs()...)
    plot!(fig; subplot = 6, right_margin = 2 * Plots.mm,
          xguide = L"Time $t$",
          plot_kwargs()...)

    savefig(fig,
            joinpath(figdir, "dingemans_solutions_at_gauges.pdf"))
    @info "Results saved in the directory" figdir

    return nothing
end



#####################################################################
# Small benchmarks based on the setup of the conservation tests
# and Favre waves

function benchmarks_conservation(; latex = false)
    b_func = x -> 0.25 * cospi(x / 75)

    setups = [(rhs_nonconservative!, 500.0, "hyperbolic_500"),
              (rhs_nonconservative!, 1000.0, "hyperbolic_1000"),
              (rhs_serre_green_naghdi_mild!, nothing, "original_mild"),
              (rhs_serre_green_naghdi_full!, nothing, "original_full")]
    Ns = (1_000, 2_000, 3_000, 4_000, 5_000)

    header = ["N"]
    for (_, _, name) in setups
        push!(header, name)
    end
    data = Vector{Any}()

    for N in Ns
        push!(data, N)
    end

    for (rhs!, λ, name) in setups
        if rhs! === rhs_nonconservative!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end

        for N in Ns
            @info "running" name N
            D = periodic_derivative_operator(derivative_order = 1,
                                             accuracy_order = 2,
                                             xmin = -150.0,
                                             xmax = 150.0,
                                             N = N)

            res = @benchmark solve_conservation($rhs!, $D;
                                                λ = $λ, alg = $alg,
                                                b_func = $b_func)
            display(res)
            push!(data, prettytime(res))
            sleep(2)
        end
    end

    data = reshape(data, (:, length(header)))
    pretty_table(data; header)
    if latex
        pretty_table(data; header, backend=Val(:latex))
    end

    return nothing
end

function benchmarks_favre_waves(; latex = false)
    setups = [(rhs_nonconservative!, 500.0, "hyperbolic_500"),
              (rhs_nonconservative!, 1000.0, "hyperbolic_1000"),
              (rhs_serre_green_naghdi_flat!, nothing, "original_flat"),
              (rhs_serre_green_naghdi_mild!, nothing, "original_mild"),
              (rhs_serre_green_naghdi_full!, nothing, "original_full")]
    Ns = (1_000, 2_000, 3_000, 4_000, 5_000)

    header = ["N"]
    for (_, _, name) in setups
        push!(header, name)
    end
    data = Vector{Any}()

    for N in Ns
        push!(data, N)
    end

    for (rhs!, λ, name) in setups
        if rhs! === rhs_nonconservative!
            alg = RDPK3SpFSAL35()
        else
            alg = Tsit5()
        end

        for N in Ns
            @info "running" name N
            xmin = -50.0
            xmax = 50.0
            if rhs! === rhs_nonconservative!
                D = periodic_derivative_operator(derivative_order = 1,
                                                 accuracy_order = 4;
                                                 xmin, xmax, N)
            else
                D = upwind_operators(periodic_derivative_operator;
                                     derivative_order = 1, accuracy_order = 3,
                                     xmin, xmax, N)
            end

            res = @benchmark solve_favre_waves($rhs!, $D;
                                               λ = $λ, alg = $alg)
            display(res)
            push!(data, prettytime(res))
            sleep(2)
        end
    end

    data = reshape(data, (:, length(header)))
    pretty_table(data; header)
    if latex
        pretty_table(data; header, backend=Val(:latex))
    end

    return nothing
end

function solve_favre_waves(rhs!, D; λ, alg)
    ε = 0.2
    h0 = 0.2
    α = 5 * h0
    g = 9.81

    abstol = 1.0e-5
    reltol = 1.0e-5

    # setup initial data
    x0 = 0.0
    jump_h = ε * h0
    h1 = h0 + jump_h
    v0 = 0.0
    jump_v = sqrt(g * (h1 + h0) / (2 * h0 * h1)) * jump_h
    h_func(x) = h0 + 0.5 * jump_h * (1 - tanh((x - x0) / α))
    v_func(x) = v0 + 0.5 * jump_v * (1 - tanh((x - x0) / α))
    b_func = zero

    u0, parameters = setup(rhs!,
                           h_func, v_func, b_func;
                           g, λ, D)

    # create and solve ODE
    saveat = (0:10:70) .* sqrt(h0 / g) # non-dimensional time
    ode = ODEProblem(rhs!, u0, extrema(saveat), parameters)

    sol = solve(ode, alg; save_everystep = false, abstol, reltol)
end

function prettytime(trial::BenchmarkTools.Trial)
    t = time(mean(trial))
    factur, unit = factor_and_unit(t)
    (t ± time(std(trial))) / factur * unit
end

function factor_and_unit(t)
    if t < 1e3
        1e0, u"ns"
    elseif t < 1e6
        1e3, u"μs"
    elseif t < 1e9
        1e6, u"ms"
    else
        1e9, u"s"
    end
end
