
RobotDynamics.@autodiff struct QuantumState <: RobotDynamics.ContinuousDynamics end
# TODO: store appropriate information in QuantumState, e.g. state size

function gen_default_QR(astate_dim::Int, acontrol_dim::Int; N::Int=1001, tf::Float64=10.0, amp_scale::Float64=0.0001, control_derivative_range::Tuple{Int,Int}=(-1, 2))
    dt = tf / (N - 1) # time step
    num_controls = acontrol_dim ÷ 2 # acontrol stores [d²(controls)], factor of 2 comes from real -> complex isomorphism
    state_indices, icontrol_indices, control_indices, dcontrol_indices = generate_astate_indices(astate_dim, num_controls, control_derivative_range=control_derivative_range)

    Q_diag = zeros(astate_dim)
    if !isnothing(icontrol_indices)
        Q_diag[icontrol_indices] = amp_scale * ones(size(icontrol_indices)[1])
    end

    if !isnothing(control_indices)
        Q_diag[control_indices] = amp_scale * ones(size(control_indices)[1])
    end

    if !isnothing(dcontrol_indices)
        Q_diag[dcontrol_indices] = amp_scale * ones(size(dcontrol_indices)[1])
    end

    # n_op = sigmap(bq_single)*sigmam(bq_single)
    # Q = complex_to_real_isomorphism(n_op.data)
    # Q = complex_to_real_isomorphism(one(bq_single).data)

    Q = Diagonal(Q_diag)

    # # R = complex_to_real_isomorphism(reshape([1.0 + 0.0*im],(1,1)))
    R = complex_to_real_isomorphism(amp_scale * Matrix(I(acontrol_dim ÷ 2)) .+ 0.0 * im)

    return Q, R
end

function gen_default_Qf(state_dim::Int, ψt_state::Vector{Float64})
    I_matrix = complex_to_real_isomorphism(Matrix(I(state_dim ÷ 2)) .+ 0.0 * im)
    # Qf = (I_matrix - ψt_state * transpose(conj_isomorphism(ψt_state)))
    # # ⟨ψf|Qf|ψf⟩ = ⟨ψf|(I - |ψt⟩⟨ψt|)|ψf⟩ = 1 - |⟨ψf|ψt⟩|^2
    Qf = complex_to_real_isomorphism(zeros(state_dim ÷ 2, state_dim ÷ 2) .+ 0.0 * im)
    return Qf
end

function gen_default_objective(astate_dim::Int, acontrol_dim::Int, ψt_state::Vector{Float64}; N::Int=1001, tf::Float64=10.0, amp_scale::Float64=0.0001, control_derivative_range::Tuple{Int,Int}=(-1, 2))
    Q, R = gen_default_QR(astate_dim, acontrol_dim; N=N, tf=tf, amp_scale=amp_scale, control_derivative_range=control_derivative_range)
    Qf = gen_default_Qf(astate_dim, ψt_state)
    obj = LQRObjective(Q, R, Qf, ψt_state, N)
    return obj
end

function gen_LQR_params(bfull::Basis, H₀::Operator, Hcs::Vector{<:Operator}, ψi::Vector{<:Ket}, ψt::Vector{<:Ket}; control_derivative_range::Tuple{Int,Int}=(-1, 2))
    """
    Here we use the augmented state (`astate`) and augmented control (`acontrol``), as defined below. 

    ```
    astate = [ψ_state_1, ψ_state_2, ..., ψ_state_n, ∫(controls), controls, d(controls)]
    acontrol = [d²(controls)]
    ```

    where `controls = [uᵣ₁, uᵣ₂, ⋯ , uᵣₙ, uᵢ₁, uᵢ₂, ⋯ , uᵢₙ]`

    This augmented state and control technique is based on work 
    in Propson, T. et al. Physical Review Applied 17 (2022). 

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


    num_states = size(ψi_full)[1]
    state_size = size(ψi_full[1])[1]
    @assert size(ψi_full)[1] == size(ψt_full)[1]
    num_controls = size(Hcs_full)[1]
    num_control_derivatives_in_astate = (control_derivative_range[2] - control_derivative_range[1])
    controls_astate_size = num_control_derivatives_in_astate * 2 * num_controls # Factor of 2 comes from complex -> real isomorphism, other factor comes from how many derivatives we want to include
    astate_dim = num_states * state_size + controls_astate_size
    acontrol_dim = 2 * num_controls


    ψi_combined = reduce(vcat, ψi_full)
    ψt_combined = reduce(vcat, ψt_full)

    astate_initial = [ψi_combined; fill(0, controls_astate_size)]
    astate_target = [ψt_combined; fill(0, controls_astate_size)]

    dynamics_func(::QuantumState, x, u) = schrodinger_dψ(x, u, H₀_full, Hcs_full; num_states=num_states, control_derivative_range=control_derivative_range)

    return astate_dim, acontrol_dim, dynamics_func, astate_initial, astate_target
end