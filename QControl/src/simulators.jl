function split_state(x, state_size::Int, num_states::Int)
    ψ_full = map(i -> x[(i-1)*state_size+1:i*state_size], 1:num_states)
    return ψ_full
end

function schrodinger_dψ(x, u, H₀_full, Hcs_full; num_states::Int=1)
    state_size = size(x)[1] ÷ num_states
    num_controls = size(Hcs_full)[1]
    ψ_full = split_state(x, state_size, num_states)
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

function solve_me(bfull::Basis, H₀::Operator, Hcs::Vector{<:Operator}, ψi::Vector{<:Ket}, u::Vector{Vector{ComplexF64}})
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
    ψi_combined = reduce(vcat, ψi_full)
    num_states = size(ψi_full)[1]

    f(x, p, t) = schrodinger_dψ(x, p[t], H₀_full, Hcs_full; num_states=num_states)
    x0 = ψi_combined

end