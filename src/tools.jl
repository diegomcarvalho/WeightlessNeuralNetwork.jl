#=
    Module: Weightless Neural Network
    Data Type: tools

    Implements methods used by other WNN's files.

    2019,2020 (@) Diego Carvalho - d.carvalho@ieee.org
=#

const tools_VERSION = "1.0";

using Random

wnn_RNG = MersenneTwister(12770);

function build_mapping(address_size::Int, n::Int, rng=wnn_RNG)
    mapping = reshape(Random.randperm(Int(n)),(n ÷ address_size,address_size))
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
