
RobotDynamics.@autodiff struct QuantumState <: RobotDynamics.ContinuousDynamics end
# TODO: store appropriate information in QuantumState, e.g. state size

function gen_default_QR(state_dim::Int, control_dim::Int; N::Int=1001, tf::Float64=10.0, R_scale::Float64=0.0001)
    dt = tf / (N - 1) # time step

    # n_op = sigmap(bq_single)*sigmam(bq_single)
    # Q = complex_to_real_isomorphism(n_op.data)
    # Q = complex_to_real_isomorphism(one(bq_single).data)
    Q = complex_to_real_isomorphism(zeros(state_dim ÷ 2, state_dim ÷ 2) .+ 0.0 * im)

    # # R = complex_to_real_isomorphism(reshape([1.0 + 0.0*im],(1,1)))
    R = complex_to_real_isomorphism(R_scale * Matrix(I(control_dim ÷ 2)) .+ 0.0 * im)

    return Q, R
end

function gen_default_Qf(state_dim::Int, ψt_state::Vector{Float64})
    I_matrix = complex_to_real_isomorphism(Matrix(I(state_dim ÷ 2)) .+ 0.0 * im)
    Qf = (I_matrix - ψt_state * transpose(conj_isomorphism(ψt_state)))
    # # ⟨ψf|Qf|ψf⟩ = ⟨ψf|(I - |ψt⟩⟨ψt|)|ψf⟩ = 1 - |⟨ψf|ψt⟩|^2
    return Qf
end

function gen_default_objective(state_dim::Int, control_dim::Int, ψt_state::Vector{Float64}; N::Int=1001, tf::Float64=10.0, R_scale::Float64=0.0001)
    Q, R = gen_default_QR(state_dim, control_dim; N=N, tf=tf, R_scale=R_scale)
    Qf = gen_default_Qf(state_dim, ψt_state)
    obj = LQRObjective(Q, R, Qf, ψt_state, N)
    return obj
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

    dynamics_func(::QuantumState, x, u) = schrodinger_dψ(x, u, H₀_full, Hcs_full; num_states=num_states)

    return state_dim, control_dim, dynamics_func, ψi_combined, ψt_combined
end