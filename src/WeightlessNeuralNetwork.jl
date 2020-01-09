#=
    Module: Weightless Neural Network

    Implements Weightless Neural Network based on Weightless Perceptrons

    2019,2020 (@) Diego Carvalho - d.carvalho@ieee.org
=#
module WeightlessNeuralNetwork

# Only export WNN data types and train and classify methods
export Retina, Wisard, Drasiw,
       train!, classify,
       classify_parallel,
       train_standard_discriminator!, train_drasiw_discriminator!,
       sigma_standard_discriminator, sigma_drasiw_discriminator,
       get_activated, get_activation, get_confidence

include("retina.jl")
include("tools.jl")
include("discriminator.jl")
include("wisard.jl")
include("drasiw.jl")

end  # module WeightlessNeuralNetwork
