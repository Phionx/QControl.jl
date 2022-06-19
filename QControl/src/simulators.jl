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

function schrodinger_dψ(astate, acontrol, H₀_full, Hcs_full; num_states::Int=1, control_derivative_range::Tuple{Int,Int}=(-1, 2))
    """
    Calculates the differential on the augmented state.

    Args:
        astate: augmented state vector
        acontrol: augmented control vector
        H₀_full: base hamiltonian written in real matrix form
        Hcs_full: list of control hamiltonians written in real matrix form
        num_states: number of states 
        control_derivative_range: 
            indicates which control derivatives (integrals) we are including
            e.g. (-1, 2) implies that ∫(controls), (controls), d(controls) are used in astate and acontrol = d²(controls)
    """

    # calculate sizes
    num_controls = size(Hcs_full)[1]

    # extract states and controls
    states, icontrols, controls, dcontrols, ddcontrols = extract_state_controls(astate, acontrol, num_controls, control_derivative_range=control_derivative_range)
    ψs = split_state(states, num_states)

    Ht_full = H₀_full
    for control_indx = 1:num_controls
        uᵣ = controls[control_indx]
        uᵢ = controls[control_indx+num_controls]
        Ht_full += uᵣ * Hcs_full[control_indx] + uᵢ * im_times_isomorphism(Hcs_full[control_indx])
    end

    # TODO: density matrices, loss, etc
    dψs = reduce(vcat, map(i -> -im_times_isomorphism(Ht_full * ψs[i]), 1:num_states))

    dastate = dψs

    # choose which control derivatives to add to dastate
    if control_derivative_range[2] > control_derivative_range[1]
        dastate_controls = reduce(vcat, [controls, dcontrols, ddcontrols][control_derivative_range[1]+2:control_derivative_range[2]+1])
        dastate = [dastate; dastate_controls]
    end

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