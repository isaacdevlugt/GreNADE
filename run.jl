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

function fidelity_stopping(current_fid, desired_fid)
    if current_fid >= desired_fid
        return true
    else
        return false
    end
end 

function observable_stopping(current_obs_stats, desired_obs)
    if abs(current_obs_stats[1] - desired_obs[1]) / desired_obs[1] <= desired_obs[2] 
        return true
    else
        return false
    end
end

function true_magnetization()
    magnetization = 0
    for Ket = 0:2^N-1
        SumSz = 0.
        for SpinIndex = 0:N-1
            Spin1 = 2*((Ket>>SpinIndex)&1) - 1
            SumSz += Spin1
        end
        magnetization += SumSz*SumSz*psi[Ket+1]^2
    end
    return magnetization / N
end

function magnetization(sample)
    sample = (sample .* 2) .- 1
    return sum(sample)*sum(sample) / N
end 

# Change these hyperparameters to your liking 
Nh = 20 # number of hidden units 

η = 0.001
batch_size = 100
epochs = 100
log_every = 1
opt = ADAM(η)

#desired_fid = 0.99

tolerance = 0.05
# arguments for early_stopping function
desired_obs = (true_magnetization(), tolerance)

initialize_parameters(seed=9999)

train(
    train_data, 
    batch_size=batch_size, 
    opt=opt, 
    epochs=epochs, 
    calc_fidelity=true,
    target=psi,
    calc_observable=true,
    num_samples=10000, 
    observable=magnetization,
    log_every=1,
    early_stopping=observable_stopping,
    early_stopping_args=desired_obs
)