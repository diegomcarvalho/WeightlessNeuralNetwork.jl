#=
    Module: Weightless Neural Network
    Data Type: Retina

    Implements the Retina. It is memory vector that hold the data representation
    used by WWNs

    2019,2020 (@) Diego Carvalho - d.carvalho@ieee.org
=#

const retina_VERSION = "1.0"

# Retina represents a memory retina
struct Retina
    mem::Array{Int8,1}

    function Retina(n::Int)
        mem = zeros(Int,n)
        return new(mem)
    end

    function Retina(v::Array{Int64,1})
        return new(v)
    end

    function Retina(v::Array{Signed,1})
        return new(v)
    end
end
