
RobotDynamics.@autodiff struct QuantumState <: RobotDynamics.ContinuousDynamics end
# TODO: store appropriate information in QuantumState, e.g. state size

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