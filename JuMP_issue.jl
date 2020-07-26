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
import MathProgBase
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

# ### MathProgBase

# +
function fba_MathProgBase(S, b, lb, ub, obj_idx::Integer; 
        sense = -1.0, 
        solver = Clp.ClpSolver())
        
    sv = zeros(size(S, 2));
    sv[obj_idx] = sense
    sol = MathProgBase.HighLevelInterface.linprog(
        sv, # Opt sense vector 
        S, # Stoichiometric matrix
        b, # row lb
        b, # row ub
        lb, # column lb
        ub, # column ub
        solver);
    isempty(sol.sol) && error("FBA failed, empty solution returned!!!")
    return (sol = sol.sol, obj_val = sol.sol[obj_idx], obj_idx = obj_idx)
end

function fba_MathProgBase(model; kwargs...)
    obj_idx = rxnindex(model, model["obj_ider"])
    return fba_MathProgBase(model["S"], model["b"], model["lb"], model["ub"], obj_idx; kwargs...)
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
fba_JuMP(_model; solver = GLPK.Optimizer)
fba_JuMP(_model; solver = Clp.Optimizer);
fba_MathProgBase(_model; solver = Clp.ClpSolver());

obj_vals = []
for model_file in model_files

    model = load_model(model_file)
    println("\nModel: $(basename(model_file)) size: ", size(model["S"]), " -------------------")
    
    println("\fba_MatrixOptInterface-GLPK.Optimizer")
    sol = @btime fba_MatrixOptInterface($model; solver = GLPK.Optimizer)
    println("obj_val: ", sol.obj_val)
    flush(stdout); 
    push!(obj_vals, sol.obj_val)

    println("\fba_MatrixOptInterface-Clp.Optimizer")
    sol = @btime fba_MatrixOptInterface($model; solver = Clp.Optimizer)
    println("obj_val: ", sol.obj_val)
    flush(stdout); 
    push!(obj_vals, sol.obj_val)

    println("\nfba_JuMP-GLPK.Optimizer")
    sol = @btime fba_JuMP($model; solver = GLPK.Optimizer)
    println("obj_val: ", sol.obj_val)
    flush(stdout); 
    push!(obj_vals, sol.obj_val)

    
    println("\nfba_JuMP-Clp.Optimizer")
    sol = @btime fba_JuMP($model; solver = Clp.Optimizer);
    println("obj_val: ", sol.obj_val)
    flush(stdout);
    push!(obj_vals, sol.obj_val)
    
    println("\nfba_MathProgBase-ClpSolver")
    sol = @btime fba_MathProgBase($model; solver = Clp.ClpSolver());
    println("obj_val: ", sol.obj_val)
    flush(stdout); 
    push!(obj_vals, sol.obj_val)
    
    @assert all(isapprox.(obj_vals[1], obj_vals))
end
