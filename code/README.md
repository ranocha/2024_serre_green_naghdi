# Structure-preserving approximations of the Serre-Green-Naghdi equations in standard and hyperbolic form

To run the code, please start Julia in this directory and execute

```julia
julia> include("code.jl")

```

Afterwards, you can perform the following numerical experiments.


## Convergence tests for Serre-Green-Naghdi soliton

```julia
julia> convergence_tests_soliton()   # grid convergence with nice tables

julia> plot_convergence_tests_soliton() # grid convergence with nice tables and convergence plots

```


## Convergence tests for a manufactured solution

```julia
julia> convergence_tests_manufactured_hyperbolic()

```


## Qualitative comparison of upwind, central, and stabilized (AV) methods

```julia
julia> plot_solution_conservation_tests()

julia> plot_solution_conservation_tests_visc()

julia> plot_solution_upwind_vs_central()

julia> plot_solution_upwind_vs_central_visc()

```


## Conservation of invariants

```julia
julia> conservation_tests() # may take some time

```


## Well-balancedness

```julia
julia> check_well_balancedness()

```


## Error growth

```julia
julia> plot_error_growth()

```


## Riemann problem

```julia
julia> plot_riemann_problem()

```


## Soliton fission

```julia
julia> plot_soliton_fission()

julia> plot_soliton_fission_visc()

```


## Favre waves

```julia
julia> plot_favre_waves_solutions()

julia> plot_favre_waves_solutions_visc()

julia> plot_favre_waves_amplitudes_over_time() # this can take long time

julia> plot_favre_waves_amplitudes_over_time_visc() # this can take long time

julia> plot_favre_waves_amplitudes_over_froude() # this can take long time

julia> plot_favre_waves_amplitudes_over_froude_visc() # this can take long time

```


## Dingemans experiment

```julia
julia> plot_dingemans_solution()

julia> plot_dingemans_solutions_at_gauges()

```


## Performance benchmarks

```julia
julia> benchmarks_conservation() # may take some time

julia> benchmarks_favre_waves() # may take some time

```
