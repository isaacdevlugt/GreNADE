using Flux
using Flux.Optimise: update!
using DelimitedFiles
using Random
using Distributions
using LinearAlgebra
using Statistics
using JLD2

function activation(v, idx)
    if idx == 1
        if length(size(v)) == 1
            return ones(Nh)
        else
            return ones(Nh, size(v,1))
        end

    else
        if length(size(v)) == 1
            return σ.(θ[2] + θ[4][:,1:idx-1] * v[1:idx-1])
        else
            return σ.(θ[2] .+ θ[4][:,1:idx-1] * transpose(v[:,1:idx-1]))
        end
    end

end

function Flux.Optimise.update!(opt, xs::Tuple, gs)
    for (x, g) in zip(xs, gs)
        update!(opt, x, g)
    end
end

function prob_v_given_vlt(vlt, idx)
    h = activation(vlt, idx)
    return σ.(θ[1][idx] .+ transpose(h) * θ[3][idx,:])
end

function probability(v)

    if length(size(v)) == 1
        prob = 1
        a = θ[2]
        for i in 1:N
            h = σ.(a)
            p = σ.(θ[1][i] .+ transpose(h) * θ[3][i,:])
            prob *= ( p^(v[i]) * (1 - p)^(1 - v[i]) )
            a += θ[4][:,i] * v[i]
        end

    else
        prob = ones(size(v,1))
        a = θ[2]
        for i in 2:size(v,1)
            a = hcat(a,θ[2])
        end

        for i in 1:N
            h = σ.(a)
            p = σ.(θ[1][i] .+ transpose(h) * θ[3][i,:])
            prob .*= ( p .^ (v[:,i]) .* (1 .- p) .^ (1 .- v[:,i]) )
            a .+= θ[4][:,i] .* transpose(v[:,i])
        end

    end

    return prob
end

function sample(num_samples)
    v = rand(0:1, num_samples) # initialize sample
    v = reshape(v, (num_samples,1))
    for idx in 2:N
        prob = prob_v_given_vlt(v, idx)
        v = hcat(v, rand.(Bernoulli.(prob)))
    end
    return v
end

function NLL(v)
    if length(size(v)) == 1
        nll = 0
        for idx in 1:N
            nll -= prob_v_given_vlt(v, idx)
        end 
    else
        nll = zeros(size(v,1))
        for idx in 1:N
            nll .-= prob_v_given_vlt(v, idx)
        end 
        nll = sum(nll) / size(v,1)
    end
                                                                        
    return nll                                                                  
end

function gradients(v)
    # please make 'v' a batch
    grads = [ 
        zeros(size(b,1),batch_size), 
        zeros(size(c,1), batch_size), 
        zeros(size(U,1), size(U,2), batch_size), 
        zeros(size(W,1), size(W,2), batch_size) 
    ]
    da = zeros(Nh, batch_size)
    
    for i = 1:N
        p = prob_v_given_vlt(v, i)
        h = activation(v, i)
        dh = transpose((p .- v[:,i]) * transpose(θ[3][i,:])) .* h .* (ones(size(h)) .- h)
        
        grads[1][i,:] = p .- v[:,i]
        grads[2] .+= dh
        grads[3][i, :, :] = transpose((p .- v[:,i]) .* transpose(h))
        grads[4][:,i,:] = transpose(v[:,i] .* transpose(da))
    
        da .+= dh    
    end
    
    for i in 1:size(grads,1)
        grads[i] = reshape(
            sum(grads[i],dims=length(size(grads[i]))), 
            size(θ[i])
        ) / batch_size
    end
   
    # must reteurn a tuple 
    return (grads[1], grads[2], grads[3], grads[4])
end

function fidelity(space, target)
    return dot(target, sqrt.(probability(space)))
end

function statistics_from_observable(observable, samples)
    obs = observable.(samples)
    mean = sum(obs) / size(samples,1)
    variance = var(obs)
    std_error = std(obs) / sqrt(size(samples,1))

    return mean, variance, std_error
end 

function train(
    train_data;
    batch_size=100, 
    opt=ADAM(), 
    epochs=1000, 
    log_every=100,
    calc_fidelity=false,
    target=nothing,
    calc_observable=false,
    num_samples=nothing,
    obs_name=nothing,
    observable=nothing
)

    # TODO: early stopping

    # batch the training data
    train_data = [view(train_data, k:k+batch_size-1, :) 
        for k in 1:batch_size:size(train_data,1)
    ]
    num_batches = size(train_data, 1)

    # allocate space for monitoring metrics
    if calc_fidelity
        space = generate_hilbert_space()
        fidelities = zeros(Int(epochs / log_every))
    end

    if calc_observable
        # observable value (mean), variance, std error
        observable_stats = zeros(Int(epochs / log_every), 3)
    end

    # TODO
    #if calc_NLL
    #    NLLs = zeros(epochs / log_every)
    #end

    count = 1
    for ep in 1:epochs
        for n in 1:num_batches
            # pass through train_data
            batch = train_data[n]
            grads = gradients(batch)
            update!(opt, θ, grads)
        end 
        
        if ep%log_every == 0
            println("epoch: ", ep)
            
            if calc_fidelity
                fidelities[count] = fidelity(space, target)
                println("Fidelity = ",fidelities[count])
            end

            if calc_observable
                samples = sample(num_samples)
                mean, variance, std_error = statistics_from_observable(
                    observable, samples
                )
                observable_stats[count,1] = mean
                observable_stats[count,2] = variance
                observable_stats[count,3] = std_error
                
                println("$obs_name"*" = ", mean)
            end

            count += 1
        end

    end

    # TODO: better file naming conventions
    # save NADE parameters
    @save "NADE_params.jld2" θ

    # save metrics
    if calc_fidelity    
        open("training_fidelities", "w") do io
            writedlm(io, fidelities)
        end
    end
    
    if calc_observable
        open("training_$obs_name", "w") do io
            writedlm(io, observable_stats)
        end
    end
 
end

function generate_hilbert_space()
    dim = [i for i in 0:2^N-1] 
    space = space = parse.(Int64, split(bitstring(dim[1])[end-N+1:end],""))

    for i in 2:length(dim)
        tmp = parse.(Int64, split(bitstring(dim[i])[end-N+1:end],""))
        space = hcat(space, tmp)
    end

    return transpose(space)
end
