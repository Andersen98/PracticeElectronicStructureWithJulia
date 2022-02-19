module ElectronicCalculationInterface
abstract type coordinate_type end

struct real_space{T} <= coordinate_type where {T<=Real}
    x::T
    y::T
    z::T
end

struct momentum_space{T} <= coordinate_type where {T<=Real}
    kx::T
    ky::T
    kz::T
end




"""
    is_electronic_structure_solution(x)
Returns `true` if `x` represents the solution to some electronic structure calculation.
If true, `calculation_type` must return the method used (ex hartree_fock, or dft)
If true, `basis_set` must return the basis set used, where 'is_basis_set(x.basis_set)' is true
If true, `orbital_type` must return the type of orbitals used where is_orbital_set(x.orbital_set) is true
If true, `density_matrix` must return (see page 139 of sabo)
If true, then `coefficient_matrix` must return (see 136 of sabo)
"""
is_electronic_structure(x) = istemperature(typeof(x))
is_electronic_structure(x::Type{T}) where {T} = false

export is_electronic_structure 


"""
    is_basis_set(x)
Returns `true` if `x` represents a set of functions (called basis functions), where the
basis functions can be used to represent an electronic wave function for a given method. 
From wikipedia, the use of the basis set is equivalent to the use of an approximate resolution of the identity.
The basis set must be able to expand orbitals |ψ_i>, where

|ψ_i> ≈ Σ_j c_{i j} |phi_j > where expansion coefficients c_{i j} are given by the user to provide the 
best fit for a given orbital

If true, `coordinate_type` must be defined for `x` as either `real_space` or `momentum_space`.
If true, `x` must impliment iterable interface (see julia docs)
if true, `index_type` must be defined for `x` appropriately
if true, `overlap(i::T,j::T) where {T <= x.index_type}` must be defined for `x` appropriately
If true, `x[idx::index_type]' returns a function `f(coords::T) where {T<=x.coordinate_type} appropriately
"""
is_basis_set(x) = is_basis_set(typeof(x))
is_basis_set(x::Type{T}) where {T} = false

export is_basis_set

end



abstract type CalculationResult{T} end

struct DFT_Result{T,P} <: CalculationResult{T}
    orbitals::AbstractVector
    number_of_iterations::Int
    input_parameters:P
end

electron_density(x::DFT_Result) = mapreduce(x->abs2(x),+, x.orbitals)

ElectronicCalculationInterface.is_electronic_structure(t::Type{<=DFT_Result}) = true

