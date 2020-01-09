#=
    Module: Weightless Neural Network
    Data Type: tools

    Implements methods used by other WNN's files.

    2019,2020 (@) Diego Carvalho - d.carvalho@ieee.org
=#

import Base.show
import Base.getindex
import Base.length

using Random
using Printf

wnn_RNG = MersenneTwister(12770);

struct Classification
    class::String           # class classifyed by the WNN
    activated::Int          # number of RAM nodes activated by the retina of the highest response
    activation::Float64     # ratio of activated RAM nodes to the total ammount of RAM nodes in the Discriminator
    confidence::Float64     # the relative confidence of the highest response with the respect of the second one.
end

get_activated(v::Array{Classification,1}) = getproperty.(v,:activated)
get_activation(v::Array{Classification,1}) = getproperty.(v,:activation)
get_confidence(v::Array{Classification,1}) = getproperty.(v,:confidence)

function Base.show(io::IO, c::Classification)
     @printf(io, "WNN Classification(class=\"%s\", RAM Nodes Activated=%d, Activation=%f, Confidence=%f)\n",
             c.class,
             c.activated,
             c.activation,
             c.confidence
            )
end

function build_mapping(address_size::Int, n::Int, rng=wnn_RNG)
    mapping = reshape(Random.randperm(rng,Int(n)),(n ÷ address_size,address_size))
    return mapping
end

function get_RAM_address(mem::Array{Int8,1},
                         mapping::Array{Int64,2},
                         address_size::Int,
                         ramid::Int)::Int
    val::Int = 1
    @inbounds for i in 1:address_size
        val =  val + mem[mapping[ramid,i]] << (address_size-i)
    end
    return val
end

function get_winners(classes::Dict{String,Int})::Array{Pair{String,Int},1}
    ordered_discriminators = sort!(collect(classes), by=x->x[2], rev=true)
    winners = Array{Pair{String,Int},1}()
    max_value = ordered_discriminators[1][2]
    for (k,v) in ordered_discriminators
        push!(winners,Pair(k,v)) 
        if max_value != v
            break
        end
    end
    return winners
end