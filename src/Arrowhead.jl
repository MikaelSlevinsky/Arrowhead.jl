__precompile__()
module Arrowhead

using Base, HierarchicalMatrices

import Base: size, getindex, inv, issymmetric, chol, eig, eigvals, +, -

export SymArrow, UpperArrow

include("SymArrow.jl")
include("UpperArrow.jl")
include("secular.jl")

end # module
