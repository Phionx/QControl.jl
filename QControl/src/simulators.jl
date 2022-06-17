function split_state(x, num_states::Int)
    state_size = size(x)[1] ÷ num_states
    ψ_full = map(i -> x[(i-1)*state_size+1:i*state_size], 1:num_states)
    return ψ_full
end

function tanh_envelope(slope, ts)
    ts = ts - minimum(ts)
    ys = (tanh.(slope * ts) + tanh.(slope * (maximum(ts) .- ts))) / 2
    ys = (ys .- minimum(ys)) / (maximum(ys) - minimum(ys))
    return ys
end

function schrodinger_dψ(astate, acontrol, H₀_full, Hcs_full; num_states::Int=1)
    """
    Calculates the differential on the augmented state.

    Here we use the augmented state (`astate`) and augmented control (`acontrol``), as defined below. 

    ```
    astate = [ψ_state_1, ψ_state_2, ..., ψ_state_n, ∫(controls), controls, d(controls)]
    acontrol = [d²(controls)]
    ```

    where `controls = [uᵣ₁, uᵣ₂, ⋯ , uᵣₙ, uᵢ₁, uᵢ₂, ⋯ , uᵢₙ]`

    This augmented state and control technique is based on work 
    in Propson, T. et al. Physical Review Applied 17 (2022). 

    Args:
        astate: augmented state vector
        acontrol: augmented control vector
        H₀_full: base hamiltonian written in real matrix form
        Hcs_full: list of control hamiltonians written in real matrix form
        num_states: number of states 
    """

    # calculate sizes
    num_controls = size(Hcs_full)[1]


    # extract states and controls
    state_indices, icontrol_indices, control_indices, dcontrol_indices = generate_astate_indices(size(astate)[1], num_controls)
    states = astate[state_indices]
    # icontrols = astate[icontrol_indices]
    controls = astate[control_indices]
    dcontrols = astate[dcontrol_indices]
    ddcontrols = acontrol

    ψs = split_state(states, num_states)

    Ht_full = H₀_full
    for control_indx = 1:num_controls
        uᵣ = controls[control_indx]
        uᵢ = controls[control_indx+num_controls]
        Ht_full += uᵣ * Hcs_full[control_indx] + uᵢ * im_times_isomorphism(Hcs_full[control_indx])
    end

    # TODO: density matrices, loss, etc
    dψs = reduce(vcat, map(i -> -im_times_isomorphism(Ht_full * ψs[i]), 1:num_states))
    dastate = [dψs; controls; dcontrols; ddcontrols]
    return dastate
end

function solve_me(bfull::Basis, H₀::Operator, Hcs::Vector{<:Operator}, ψi::Vector{<:Ket}, u::Vector{Vector{ComplexF64}})
    """
    TODO: complete this!

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
    # x0 = ψi_combined

end