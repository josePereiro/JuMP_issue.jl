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

import JuMP
import GLPK
import Clp
import MathProgBase
import JSON
using SparseArrays
using BenchmarkTools

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

# ### JuMP

# +
function fba_JuMP(S, b, lb, ub, obj_idx::Integer; 
        sense = JuMP.MOI.MAX_SENSE, err = [], 
        solver = GLPK.Optimizer)
    

    lp_model = JuMP.Model(solver)
    JuMP.set_silent(lp_model)

    M,N = size(S)

    # Variables
    JuMP.@variable(lp_model, lp_x[1:N])

    # Constraints
    JuMP.@constraint(lp_model, balance, S * lp_x .== b)
    JuMP.@constraint(lp_model, bounds, lb .<= lp_x .<= ub)

    # objective
    JuMP.@objective(lp_model, sense, lp_x[obj_idx])

    # optimize
    JuMP.optimize!(lp_model)
    
    #FBAout
    obj_val = JuMP.value(lp_x[obj_idx])
    obj_val in err && error("FBA failed, error value returned, obj_val[$(obj_idx)] = $(obj_val)!!!")
    
    return (sol = JuMP.value.(lp_x), obj_val = obj_val, obj_idx = obj_idx)
end

function fba_JuMP(model; kwargs...) 
    obj_idx = rxnindex(model, model["obj_ider"])
    return fba_JuMP(model["S"], model["b"], model["lb"], model["ub"], obj_idx; kwargs...)
end
# -

# ### MathProgBase

# +
function fba_MathProgBase(S, b, lb, ub, obj_idx::Integer; 
        sense = -1.0, solver = Clp.ClpSolver())
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

# precompaling
model = load_model("toy_model.json")
fba_JuMP(model; solver = GLPK.Optimizer)
fba_JuMP(model; solver = Clp.Optimizer);
fba_MathProgBase(model; solver = Clp.ClpSolver());

for model_file in model_files
    model = load_model(model_file)
    println("\nModel: $(basename(model_file)) size: ", size(model["S"]))
    
    println("fba_JuMP-GLPK.Optimizer")
    sol1 = @btime fba_JuMP(model; solver = GLPK.Optimizer)
    flush(stdout); 
    
    println("fba_JuMP-Clp.Optimizer")
    sol2 = @btime fba_JuMP(model; solver = Clp.Optimizer);
    flush(stdout); 
    
    println("fba_MathProgBase-ClpSolver")
    sol3 = @btime fba_MathProgBase(model; solver = Clp.ClpSolver());
    flush(stdout); 
    
    @assert isapprox(sol1.obj_val, sol2.obj_val) && isapprox(sol1.obj_val, sol3.obj_val)
end
