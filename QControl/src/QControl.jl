module QControl

using QuantumOptics
using PyPlot
using Altro
using ForwardDiff
using LinearAlgebra
using SparseArrays
using RobotDynamics
using TrajectoryOptimization
using StaticArrays
using FiniteDiff
const TO = TrajectoryOptimization
const RD = RobotDynamics

RobotDynamics.@autodiff struct QuantumState <: RobotDynamics.ContinuousDynamics end
# TODO: store appropriate information in QuantumState, e.g. state size

function complex_to_real_isomorphism(M::Union{Matrix,SparseMatrixCSC}) #TODO Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64, Int64}})
    """
    M
    => 
    Mᵣ -Mᵢ
    Mᵢ  Mᵣ
    """
    return [real(M) -imag(M); imag(M) real(M)]
end

function real_to_complex_isomorphism(M_full::Union{Matrix,SparseMatrixCSC}) #TODO: {Float64})
    """
    Mᵣ -Mᵢ
    Mᵢ  Mᵣ
    => 
    M
    """
    n_rows, n_cols = size(M_full)
    M = M_full[1:n_rows÷2, 1:n_cols÷2] + im * M_full[n_rows÷2+1:end, 1:n_cols÷2]
    return M
end

function im_times_isomorphism(M_full::Union{Matrix,SparseMatrixCSC}) #TODO: {Float64})
    """
    i(Mᵣ + iMᵢ) -> -Mᵢ + iMᵣ


    im
    *
     Mᵣ -Mᵢ
     Mᵢ  Mᵣ
    => 
    -Mᵢ -Mᵣ
     Mᵣ -Mᵢ   
    """
    n_rows_full, n_cols_full = size(M_full)
    n_rows = n_rows_full ÷ 2
    n_cols = n_cols_full ÷ 2
    Mᵣ = M_full[1:n_rows, 1:n_cols]
    Mᵢ = M_full[n_rows+1:end, 1:n_cols]
    return [-Mᵢ -Mᵣ; Mᵣ -Mᵢ]
end

function conj_isomorphism(M_full::Union{Matrix,SparseMatrixCSC}) #TODO: {Float64})
    """
    Mᵣ + iMᵢ -> Mᵣ + i (-Mᵢ)


     Mᵣ -Mᵢ
     Mᵢ  Mᵣ
    => 
     Mᵣ  Mᵢ
    -Mᵢ  Mᵣ 
    """
    n_rows_full, n_cols_full = size(M_full)
    n_rows = n_rows_full ÷ 2
    n_cols = n_cols_full ÷ 2
    Mᵣ = M_full[1:n_rows, 1:n_cols]
    Mᵢ = M_full[n_rows+1:end, 1:n_cols]

    return [Mᵣ Mᵢ; -Mᵢ Mᵣ]
end

function complex_to_real_isomorphism(v::Vector) #TODO: {ComplexF64})
    """
    v
    => 
    vᵣ
    vᵢ
    """
    return [real(v); imag(v)]
end

function real_to_complex_isomorphism(v_full::Vector) #TODO: {Float64})
    """
    vᵣ
    vᵢ
    => 
    v
    """
    n_rows = size(v_full)[1]
    v = v_full[1:n_rows÷2] + im * v_full[n_rows÷2+1:end]
    return v
end

function im_times_isomorphism(v_full::Vector) #TODO:{Float64})
    """
    i(vᵣ + ivᵢ) -> -vᵢ + ivᵣ

    im 
    *
     vᵣ
     vᵢ
    => 
    -vᵢ
     vᵣ
    """
    n_rows_full = size(v_full)[1]
    n_rows = n_rows_full ÷ 2
    vᵣ = v_full[1:n_rows]
    vᵢ = v_full[n_rows+1:end]
    return [-vᵢ; vᵣ]
end

function conj_isomorphism(v_full::Vector) # TODO: {Float64})
    """
    vᵣ + ivᵢ -> vᵣ + i(-vᵢ)

     vᵣ
     vᵢ
    => 
     vᵣ
    -vᵢ
    """
    n_rows_full = size(v_full)[1]
    n_rows = n_rows_full ÷ 2
    vᵣ = v_full[1:n_rows]
    vᵢ = v_full[n_rows+1:end]
    return [vᵣ; -vᵢ]
end

function gen_LQR_params(bfull::Basis, H₀::Operator, Hcs::Vector{<:Operator}, ψi::Vector{<:Ket}, ψt::Vector{<:Ket})
    """

    Args:
        bfull: full basis
        H₀ (Operator)
        Hcs (Vector{Ket}): vector of control Hamiltonians 
        ψi (Vector): vector of initial states
        ψt (Vector): vector of target states
    """
    H₀_full = complex_to_real_isomorphism(H₀.data)
    Hcs_full = complex_to_real_isomorphism.(map(H -> H.data, Hcs))
    ψi_full = complex_to_real_isomorphism.(map(ψ -> ψ.data, ψi))
    ψt_full = complex_to_real_isomorphism.(map(ψ -> ψ.data, ψt))


    ψi_combined = reduce(vcat, ψi_full)
    ψt_combined = reduce(vcat, ψt_full)

    num_states = size(ψi_full)[1]
    state_size = size(ψi_full[1])[1]
    @assert size(ψi_full)[1] == size(ψt_full)[1]
    state_dim = num_states * state_size

    num_controls = size(Hcs_full)[1]
    control_dim = 2 * num_controls # Factor of 2 comes from complex -> real


    function dynamics_func(::QuantumState, x, u)
        ψ_full = map(i -> x[(i-1)*state_size+1:i*state_size], 1:num_states)
        Ht_full = H₀_full
        for control_indx = 1:num_controls
            uᵣ = u[control_indx]
            uᵢ = u[control_indx+num_controls]
            Ht_full += uᵣ * Hcs_full[control_indx] + uᵢ * im_times_isomorphism(Hcs_full[control_indx])
        end

        # TODO: density matrices, loss, etc
        dψ = reduce(vcat, map(i -> -im_times_isomorphism(Ht_full * ψ_full[i]), 1:num_states))
        return dψ
    end


    return state_dim, control_dim, dynamics_func, ψi_combined, ψt_combined
end



end # module
