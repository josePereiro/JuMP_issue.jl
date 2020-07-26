# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: jl,ipynb
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Julia 1.1.0
#     language: julia
#     name: julia-1.1
# ---

import Pkg
import JuMP
import GLPK
import Clp
import MatrixOptInterface
import JSON
using SparseArrays
using BenchmarkTools
using InteractiveUtils

# ---
# # Tools
# ---

function load_model(file)
    model_dict = JSON.parsefile(file)
    
    # Fix types
    model_dict["S"] = model_dict["S"] .|> Vector{Float64}
    model_dict["S"] = sparse(model_dict["S"]...) |> Matrix
    for k in ["b", "lb", "ub"]
        model_dict[k] = model_dict[k] |> Vector{Float64}
    end
    for k in ["rxns", "mets"]
        model_dict[k] = model_dict[k] |> Vector{String}
    end
    model_dict
end

rxnindex(model, ider::Int) = 1 <= ider <= size(model["S"], 2) ? ider : nothing
rxnindex(model, ider::AbstractString) = findfirst(isequal(ider), model["rxns"])

# ---
# # FBA
# ---

# ### MatrixOptInterface
function fba_MatrixOptInterface(S, b, lb, ub, obj_idx::Integer; 
    solver = Clp.Optimizer, sense = JuMP.MOI.MAX_SENSE)

    c = zeros(size(S, 2))
    c[obj_idx] = 1.0
    lp = MatrixOptInterface.LPForm(sense, c, S, b, b, lb, ub)

    lp_model = JuMP.Model()
    JuMP.MOI.copy_to(lp_model, lp)
    JuMP.set_optimizer(lp_model, Clp.Optimizer)
    JuMP.set_silent(lp_model)
    JuMP.optimize!(lp_model)
    xs = JuMP.value.(JuMP.all_variables(lp_model))
    return (sol = xs, obj_val = xs[obj_idx], obj_idx = obj_idx)
end

function fba_MatrixOptInterface(model; kwargs...)
    obj_idx = rxnindex(model, model["obj_ider"])
    return fba_MatrixOptInterface(model["S"], model["b"], model["lb"], model["ub"], obj_idx; kwargs...)
end

# ### JuMP

# +
function fba_JuMP(S, b, lb, ub, obj_idx::Integer; 
        sense = JuMP.MOI.MAX_SENSE, 
        solver = GLPK.Optimizer)
    

    model = JuMP.Model(solver)
    JuMP.set_silent(model)

    M,N = size(S)

    # Variables
    JuMP.@variable(model, lb[i] <= x[i = 1:N] <= ub[i])

    # Constraints
    JuMP.@constraint(model, S * x .== b)

    # objective
    JuMP.@objective(model, sense, x[obj_idx])

    # optimize
    JuMP.optimize!(model)

    # FBAout    
    return (sol = JuMP.value.(x), obj_val = JuMP.value(x[obj_idx]), obj_idx = obj_idx)
end

function fba_JuMP(model; kwargs...) 
    obj_idx = rxnindex(model, model["obj_ider"])
    return fba_JuMP(model["S"], model["b"], model["lb"], model["ub"], obj_idx; kwargs...)
end
# -

# ---
# # Tests
# ---

model_files = ["toy_model.json", "e_coli_core.json", "iJR904.json", "HumanGEM.json"] # HumanGEM could take some time
@assert all(isfile.(model_files))

# # versioninfo
println("\nvesioninfo -------------------")
versioninfo()
println()
flush(stdout)

# # Project
println("\nProject.toml -------------------")
Pkg.status(mode = Pkg.PKGMODE_PROJECT)
println()
flush(stdout)

# precompaling
_model = load_model("toy_model.json")
fba_MatrixOptInterface(_model; solver = GLPK.Optimizer)
fba_MatrixOptInterface(_model; solver = Clp.Optimizer);
fba_JuMP(_model; solver = GLPK.Optimizer);
fba_JuMP(_model; solver = Clp.Optimizer);


tests = [
        ("fba_MatrixOptInterface-GLPK.Optimizer", fba_MatrixOptInterface, GLPK.Optimizer),
        ("fba_MatrixOptInterface-Clp.Optimizer", fba_MatrixOptInterface, Clp.Optimizer),
        ("fba_JuMP-GLPK.Optimizer", fba_JuMP, GLPK.Optimizer),
        ("fba_JuMP-Clp.Optimizer", fba_JuMP, Clp.Optimizer),
    ]

for model_file in model_files
    
    obj_vals = []
    model = load_model(model_file)
    println("\nModel: $(basename(model_file)) size: ", size(model["S"]), " -------------------")

    for (tname, fba_method, solver_) in tests
        println("\n$tname")
        sol = @btime $fba_method($model; solver = $solver_)
        println("obj_val: ", sol.obj_val)
        flush(stdout); 
        push!(obj_vals, sol.obj_val)
    end
    
    @assert all(isapprox.(obj_vals[1], obj_vals, atol = 1e-3))
end
