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
const drasiw_VERSION = "1.1"

using Base.Threads

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
                    z::Bool=false, p::Bool=false, par::Bool=false)
        tr::Function = train!
        cl::Function = par ? classify_parallel : classify
        return new(a,s,z,p,Dict{String,Discriminator}(),tr,cl)
    end
end

function train!(w::Drasiw, x::Array{Retina,1}, y::Array{String,1})
    lx = length(x)
    ly = length(y)

    # check if they have the same size
    if lx != ly
        println("The Drasiw cannot be trained: length of Retinas != length of Cases ")
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
# Drasiw classification method
function classify(w::Drasiw, x::Array{Retina,1})
    n_of_retinas = size(x)[1]
    n_of_ram_nodes = size(collect(values(w.net))[1].ram)[1] 

    y = Array{Classification,1}(undef, n_of_retinas)

    ckeys = collect(keys(w.net))
    n_of_classes = length(ckeys)

    classes = Dict(zip(ckeys,zeros(Int,length(ckeys))))

    # loop over all retinas
    @inbounds for i = 1:n_of_retinas
        # init the b-bleaching
        b = 0
        while true
            # raise the bleaching bar 
            b += 1

            # calculate each class discriminator value
            for k in 1:n_of_classes
                d = w.net[ckeys[k]]
                classes[ckeys[k]] = d.Σ(d,x[i],b)
            end

            # find winners
            # and the Oscar goes to?
            winners = get_winners(classes)

            if length(winners) <= 2
                # if only one winners, it's done. Let's get the next one
                winner = winners[1]
                y[i] = Classification(winner[1], winner[2], (winner[2]/n_of_ram_nodes), (winner[2] - winners[end][2])/winner[2])
                break
            elseif b >= n_of_ram_nodes
                # so check if it needs a hard stop. If it's a tie, then random select
                winner = winners[Random.randperm(Int(length(winners)-1))[1]]
                y[i] = Classification(winner[1], winner[2], (winner[2]/n_of_ram_nodes), (winner[2] - winners[end][2])/winner[2])
                break
            end
        end
    end

    # return the classification result
    return y
end

# Drasiw parallel classification method
function classify_parallel(w::Drasiw, x::Array{Retina,1})
    nth = nthreads()
    # determine who many retinas it has to classify
    n_of_retinas = size(x)[1]

    # to stop the bleaching algorithm, compare with the number of ram nodes
    # which indicates the maximum value returned by Σ     
    n_of_ram_nodes = size(collect(values(w.net))[1].ram)[1] 

    # create an array to hold the Drasiw's guesses
    y = Array{Classification,1}(undef, n_of_retinas)

    # get all known classes and the set size
    ckeys = collect(keys(w.net))
    n_of_classes = length(ckeys)

    # The code needs one class vector and a discriminator for each thread
    classes = [ Dict(zip(ckeys,zeros(Int,length(ckeys)))) for i = 1:nth ]
    disc = Array{Discriminator,1}(undef, nth)

    # parallel loop on the retina vector
    Threads.@threads for i = 1:n_of_retinas
        # init the b-bleaching
        b = 0
        thid = threadid()
        @inbounds while true
            # raise the bleaching bar
            b += 1
            
            # calculate each class discriminator value
            for k in 1:n_of_classes
                disc[thid] = w.net[ckeys[k]]
                classes[thid][ckeys[k]] = disc[thid].Σ(disc[thid],x[i],b)
            end

            # and the Oscar goes to?
            winners = get_winners(classes[thid])

            if length(winners) <= 2
                # if only one winners, it's done. Let's get the next one
                winner = winners[1]
                y[i] = Classification(winner[1], winner[2], (winner[2]/n_of_ram_nodes), (winner[2] - winners[end][2])/winner[2])
                break
            elseif b >= n_of_ram_nodes
                # so check if it needs a hard stop. If it's a tie, then random select
                winner = winners[Random.randperm(Int(length(winners)-1))[1]]
                y[i] = Classification(winner[1], winner[2], (winner[2]/n_of_ram_nodes), (winner[2] - winners[end][2])/winner[2])
                break
            end
        end
    end

    # return the classification result
    return y
end
