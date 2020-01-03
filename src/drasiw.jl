#=
    Module: Weightless Neural Network
    Data Type: Drasiw

    Implements the Drasiw. It is composed of a set of discriminators, often
    called as classifiers[1] with bleaching. The bleaching is implemented
    based on the description provided by [2].

    This specific implementation is based on the description provided by [1],
    and the zero_skip and padded_retina were based on the IAZero implementation.

    [1] M. De Gregorio and M. Giordano, “Cloning DRASiW systems via memory
        transfer,” Neurocomputing, vol. 192, pp. 115–127, 2016.
    [2] Carneiro, Hugo Cesar de Castro, "Função do Índice de Síntese das
        Linguagens na Classificação Gramatical com Redes Neurais Sem Peso",
        Rio de Janeiro, Dissertação (mestrado), UFRJ/COPPE, 2012.

    2019,2020 (@) Diego Carvalho - d.carvalho@ieee.org
=#
const drasiw_VERSION = "1.0"

# Drasiw Class
struct Drasiw
    address::Int           # Number of bits per RAM node
    size::Int              # Number of pixels on the retina
    zero_skip::Bool        # Flags to ignore zero address lines on RAM nodes
    padded_retina::Bool    # pad the retina's remainder pixels (not coverd)

    net::Dict{String,Discriminator} # Set of discriminators
    train!::Function
    classify::Function

    Drasiw(a::Int64, s::Int64, z::Bool, p::Bool,
           d::Dict{String,Discriminator},
           tr::Function,
           cf::Function) = new(a,s,z,p,d,tr,cf)

    function Drasiw(a::Int, s::Int,
                    z::Bool=false, p::Bool=false)
        tr::Function = train!
        cl::Function = classify
        return new(a,s,z,p,Dict{String,Discriminator}(),tr,cl)
    end
end

function train!(w::Drasiw, x::Array{Retina,1}, y::Array{String,1})
    lx = length(x)
    ly = length(y)

    if lx != ly
        return
    end

    class_keys = keys(w.net)

    @inbounds for i in 1:lx
        k = y[i]
        if !(k ∈ class_keys)
            # create a new class discriminator
            w.net[k] = Discriminator(w.address,w.size,
                                     w.zero_skip, w.padded_retina,
                                     train_drasiw_discriminator!,
                                     sigma_drasiw_discriminator)
        end
        d = w.net[k]
        d.Γ!(d,x[i])
    end
end

function classify(w::Drasiw, x::Array{Retina,1})
    m = size(collect(values(w.net))[1].ram)[1] # TODO can crash if net is not initialized
    y = Array{String,1}()
    ckeys = keys(w.net)
    classes = Dict(zip(ckeys,zeros(Int,length(ckeys))))
    @inbounds for i in x
        b = 0
        while true
            b += 1
            for (k,v) in classes
                d = w.net[k]
                classes[k] = d.Σ(d,i,b)
            end
            winners = findall(x->x==maximum(values(classes)), classes)
            if (length(winners) == 1) || (b >= m)
                push!(y,winners[1])
                break
            end
        end
    end
    return y
end
