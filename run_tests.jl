# This must be run from root
!isfile("run_tests.jl") && error("Run this script from JuMP_issue dir!!!")

try
    julia = Base.julia_cmd()
    
    println("\nRunning test with Clp0.7.1 ------------------ \n")
    # This must bring a Manifest with Clp v0.7.1
    run(`git checkout clp_0.7.1`)
    run(`$julia --project JuMP_issue.jl`)
    flush(stdout)
    println()

    println("\nRunning test with Clp up tu day ------------------ \n")
    # This must bring a Manifest with Clp v0.8
    run(`git checkout clp_up_to_date`) 
    run(`$julia --project JuMP_issue.jl`)
    flush(stdout)
    println()
finally
    run(`git checkout master`) 
end