#=
    Module: Weightless Neural Network
    Data Type: Wisard

    2019,2020 (@) Diego Carvalho - d.carvalho@ieee.org
=#

using Base.Threads

# Wisard Class
"""
    w = Wisard(address,size,zero_skip,padded_retina,train!,classify,parallel)

    address     - ::Int Number of bits per RAM node
    size        - ::Int Number of pixels on the retina
    zero_skip   - ::Boolean pad the retina's remainder pixels (not coverd)
    parallel    - ::Boolean enables parallel classification
    
Implements the WiSARD Neural Network (Wilkes, Stonham and Aleksander
Recognition Device). It is composed of a set of discriminators, often
called as classifiers[^1].

This specific implementation is based on the description provided by [^2],
and the *zero skip* and *padded retina* were based on the IAZero implementation.

[^1]: I. Aleksander, W. Thomas, P. Bowden, WISARD a radical step forward in
    image recognition, Sens. Rev. 4 (1984) 120–124.
[^2]: M. De Gregorio and M. Giordano, “Cloning DRASiW systems via memory
    transfer,” Neurocomputing, vol. 192, pp. 115–127, 2016.
"""
struct Wisard
    address::Int           # Number of bits per RAM node
    size::Int              # Number of pixels on the retina
    zero_skip::Bool        # Flags to ignore zero address lines on RAM nodes
    padded_retina::Bool    # pad the retina's remainder pixels (not coverd)

    net::Dict{String,Discriminator} # Set of discriminators
    train!::Function                # Train function
    classify::Function              # Classify function

    #Ctors
    Wisard(a::Int64, s::Int64, z::Bool, p::Bool,
           d::Dict{String,Discriminator},
           tr::Function,
           cf::Function) = new(a,s,z,p,d,tr,cf)

    function Wisard(a::Int, s::Int, z::Bool=false, p::Bool=false, par::Bool=false)
        tr::Function = train!
        cl::Function = par ? classify_parallel : classify
        return new(a,s,z,p,Dict{String,Discriminator}(),tr,cl)
    end
end

# Wisard training method
function train!(w::Wisard, x::Array{Retina,1}, y::Array{String,1})
    # get the number of Retinas (lx) and cases (ly)
    lx = length(x)
    ly = length(y)

    # check if they have the same size
    if lx != ly
        println("The WiSARD cannot be trained: length of Retinas != length of Cases ")
        return
    end

    # get the current trained set of classes
    class_keys = keys(w.net)

    # loop over each retina and train the corresponding discriminator
    @inbounds for i in 1:lx
        k = y[i]      # get the class name
        if !(k ∈ class_keys)    # check if it's already been trained in the WiSARD
            # nope, so create a new class discriminator
            w.net[k] = Discriminator(w.address,w.size)
        end
        d = w.net[k] # get the discriminator for k class
        d.Γ!(d,x[i]) # train the discriminator with this retina
    end

    return # nothing more to do, go back to the user
end

# Wisard classification method
function classify(w::Wisard, x::Array{Retina,1})
    # create an array to hold the WiSARD's guesses
    y = Array{Classification,1}()

    # get the number of RAM nodes on each discriminator
    n_of_ram_nodes = size(collect(values(w.net))[1].ram)[1]

    # get the set of keys known by this Wisard
    ckeys = keys(w.net)
    # create a dictionary to hold the answer of each discriminator
    classes = Dict(zip(ckeys,zeros(Int,length(ckeys))))
    # loop over all retinas
    @inbounds for i in x
        # for the i retina, loop over each discriminator and stor its answer
        #d = Array{Discriminator,1}(undef, nthreads())
        for (k,v) in classes
            d = w.net[k]
            classes[k] = d.Σ(d,i)
        end
        # and the Oscar goes to?
        winners = get_winners(classes) 
        # get_winners returns at least more then one (the winner and the 2nd)
        # so, if more then 2, the winners may have more than one class (ties)
        if (length(winners) > 2)
            # in case of ties, randomize one answer and push into the guess vector
            winner = winners[Random.randperm(Int(length(winners)-1))[1]]
        else
            # only one class, so push into the guess vector
            winner = winners[1]
        end
        push!(y,Classification(winner[1], winner[2], (winner[2]/n_of_ram_nodes), (winner[2] - winners[end][2])/winner[2]))
    end

    return y # nothing more to do, return the guess vector
end

# Wisard parallel classification method
function classify_parallel(w::Wisard, x::Array{Retina,1})
    # create an array to hold the WiSARD's guesses
    nx = size(x)[1]
    y = Array{Classification,1}(undef, nx)

    # get the number of RAM nodes on each discriminator
    n_of_ram_nodes = size(collect(values(w.net))[1].ram)[1]

    # get the key set known by this Wisard
    ckeys = keys(w.net)
    k = collect(ckeys)

    # create a dictionary to hold the answer of each discriminator
    # loop over all retinas
    Threads.@threads for i in 1:nx
        classes = Dict(zip(ckeys,zeros(Int,length(ckeys))))
        # for the i retina, loop over each discriminator and stor its answer
        d = Array{Discriminator,1}(undef, nthreads())
        @inbounds for c in 1:length(classes)
            d[threadid()] = w.net[k[c]]
            classes[k[c]] = d[threadid()].Σ(d[threadid()],x[i])
        end
        # and the Oscar goes to?
        winners = get_winners(classes) 
        # get_winners returns at least more then one (the winner and the 2nd)
        # so, if more then 2, the winners may have more than one class (ties)
        if (length(winners) > 2)
            # in case of ties, randomize one answer and push into the guess vector
            winner = winners[Random.randperm(Int(length(winners)-1))[1]]
        else
            # only one class, so push into the guess vector
            winner = winners[1]
        end
        y[i] = Classification(winner[1], winner[2], (winner[2]/n_of_ram_nodes), (winner[2] - winners[end][2])/winner[2])
    end

    return y # nothing more to do, return the guess vector
end
