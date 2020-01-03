#=
    Module: Weightless Neural Network
    Data Type: Drasiw

    Implements the Discriminator the perception for WNN. It provides two
    functions Γ!() and Σ().

    Γ!() implements the training of a perceptron and Σ() implements the
    readout of a perceptron.

    This specific implementation is based on the description provided by [1],
    [2] and the zero_skip and padded_retina were based on the IAZero
    implementation.

    [1] M. De Gregorio and M. Giordano, “Cloning DRASiW systems via memory
        transfer,” Neurocomputing, vol. 192, pp. 115–127, 2016.
    [2] Carneiro, Hugo Cesar de Castro, "Função do Índice de Síntese das
        Linguagens na Classificação Gramatical com Redes Neurais Sem Peso",
        Rio de Janeiro, Dissertação (mestrado), UFRJ/COPPE, 2012.

    2019,2020 (@) Diego Carvalho - d.carvalho@ieee.org
=#

# TODO - Implement the show() and performance improvements.

const discriminator_VERSION = "0.9"

# Discriminator represents one Σ in an Weightless Neural Network
struct Discriminator
    address::Int           # Number of bits per RAM node
    size::Int              # Number of pixels on the retina
    zero_skip::Bool        # Flags to ignore zero address lines on RAM nodes
    padded_retina::Bool    # pad the retina's remainder pixels (not coverd)
    map::Array{Int64,2}    # Current mapping between the retina and the RAM nodes
    ram::Array{Int16,2}    # RAM Nodes
    Γ!::Function           # Train the discriminator
    Σ::Function            # Read the discriminator

    function Discriminator(address::Int, size::Int,
                           zero_skip::Bool=false, padded_retina::Bool=false,
                           gamma::Function=train_standard_discriminator!,
                           sigma::Function=sigma_standard_discriminator)
        mod_size = mod(size,address)
        pad = (padded_retina && (mod_size != 0)) ? mod_size : address

        this_rns = (size + (address-pad)) ÷ address
        this_map = build_mapping(address,this_rns*address)

        this_adl = 1 << address
        this_ram = zeros(Int16,(this_rns,this_adl))

        return new( Int(address), Int(size),
                    zero_skip, padded_retina,
                    this_map, this_ram,
                    gamma, sigma)
    end
end

function train_standard_discriminator!(d::Discriminator, r::Retina)
    n = length(d.map) ÷ d.address
    @inbounds for i in 1:n
        j = get_RAM_address(r.mem, d.map, d.address, i)
        d.ram[i,j] = 1
    end
end

function train_drasiw_discriminator!(d::Discriminator, r::Retina)
    n = length(d.map) ÷ d.address
    @inbounds for i in 1:n
        j = get_RAM_address(r.mem, d.map, d.address, i)
        d.ram[i,j] += 1
    end
end

function sigma_standard_discriminator(d::Discriminator, r::Retina)
    val = 0
    start_line = d.zero_skip ? 2 : 1
    end_line = length(d.map) ÷ d.address
    @inbounds for i in start_line:end_line
        j = get_RAM_address(r.mem, d.map, d.address, i)
        val += d.ram[i,j] > 0 ? 1 : 0
    end
    return val
end

function sigma_drasiw_discriminator(d::Discriminator, r::Retina, b::Int)
    val = 0
    start_line = d.zero_skip ? 2 : 1
    end_line = length(d.map) ÷ d.address
    @inbounds for i in start_line:end_line
        j = get_RAM_address(r.mem, d.map, d.address, i)
        v = d.ram[i,j]
        val += (v >= b) ? 1 : 0
    end
    return val
end

# import Base.show
# using Printf
#
# function Base.show(io::IO, d::Discriminator)
#     @printf(io, "Discriminator(address=%d,size=%d)\n\n",
#             d.address,
#             d.size)
#     @printf(io, "mapping\n")
#     show(io,d.map)
#     @printf(io, "\nram\n")
#     show(io,d.ram)
#
# end
