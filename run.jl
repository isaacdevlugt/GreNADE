using Flux
using Flux.Optimise: update!
using DelimitedFiles
using Random
using Distributions
using LinearAlgebra

include("NADE.jl")

train_path = "tfim1D_samples"
psi_path = "tfim1D_psi"

train_data = Int.(readdlm(train_path))
psi = readdlm(psi_path)[:,1]

N = size(train_data,2)

# Change these hyperparameters to your liking
Nh = 20 # number of hidden units 

η = 0.001
batch_size = 100
epochs = 100
log_every = 1
opt = ADAM(η)

initialize_parameters(seed=9999)

train(
    train_data, 
    batch_size=batch_size,
    opt=opt,
    epochs=epochs, 
    calc_fidelity=true,
    target=psi,
    log_every=log_every
)
