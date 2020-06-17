using Flux
using Flux.Optimise: update!
using DelimitedFiles
using Random
using Distributions
using LinearAlgebra
using ArgParse

include("NADE.jl")
include("postprocess.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
    "--Nh"
        help = "number of hidden units"
        arg_type=Int
    "--train_path"
        help = "training data path"
        arg_type=String
    "--psi_path"
        help = "true psi path"
        arg_type=String
    end
    return parse_args(s)
end

parsed_args = parse_commandline()

Nh = parsed_args["Nh"]
train_path = parsed_args["train_path"]
psi_path = parsed_args["psi_path"]

train_data = Int.(readdlm(train_path))
true_psi = readdlm(psi_path)[:,1]

N = size(train_data,2)
NADE_ID = rand(0:10000) 

# names of files to save things to
fidelity_path = "fidelities/fidelity_N=$N"*"_Nh=$Nh"*"_ID=$NADE_ID"
parameter_path = "params/parameters_N=$N"*"_Nh=$Nh"*"_ID=$NADE_ID"

function fidelity_stopping(current_fid, desired_fid)
    if current_fid >= desired_fid
        return true
    else
        return false
    end
end 

# Change these hyperparameters to your liking 
η = 0.01
batch_size = 100
epochs = 10000
log_every = 100
opt = ADAM(η)

desired_fid = 0.995
initialize_parameters(seed=9999)

args = train(
    train_data, 
    batch_size=batch_size, 
    opt=opt, 
    epochs=epochs,
    calc_fidelity=true,
    target=true_psi, 
    early_stopping=fidelity_stopping,
    early_stopping_args=desired_fid,
    log_every=log_every
)

fidelities = args[1]

if fidelities[size(fidelities,1)] >= desired_fid
    println("Reached desired fidelity")
    open(fidelity_path, "w") do io
        writedlm(io, fidelities)
    end
    @save parameter_path θ  
else
    println("Increasing Nh by 5")
    Nh += 5
    submit_new_job(Nh, train_path, psi_path) 
end
