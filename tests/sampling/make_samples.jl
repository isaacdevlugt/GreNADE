using DelimitedFiles

include("../../NADE.jl")

# arbitraty parameters
N = 5
Nh = 5
initialize_parameters()

space = generate_hilbert_space()
prob = probability(space)

num_samples = 10000
samples = convert(Array{Int,2}, sample(num_samples))

open("NADE_probability", "w") do io
    writedlm(io, prob)
end

open("NADE_samples", "w") do io
    writedlm(io, samples)
end
