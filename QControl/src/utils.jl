# Optimizer Params



# Complex <-> Real Isomorphism Helpers
# ======================================================================
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


# Save & Load Data
# ======================================================================
function save_solver_data(solver::ALTROSolver; label::String="solver")
    data = Dict("astates" => states(solver), "acontrols" => controls(solver))
    save_object(string(label, ".jld2"), data)
end

function load_solver_data(filename::String)
    data = load_object(filename)
    return data
end

# Parse Data
# ======================================================================
function states_to_kets(X::Vector, b::Basis)
    Xv = Vector.(X)
    Xcv = real_to_complex_isomorphism.(Xv)
    Xqv = map(cv -> normalize(Ket(b, cv)), Xcv)
    return Xqv
end

function controls_to_amplitudes(U::Vector)
    Uv = Vector.(U)
    Ucv = real_to_complex_isomorphism.(Uv)
    return Ucv
end

function generate_astate_indices(astate_size::Int, num_controls::Int; control_derivative_range::Tuple{Int,Int}=(-1, 2))
    """
    Here we use the augmented state (`astate`) and augmented control (`acontrol``), as defined below. 

    ```
    astate = [ψ_state_1, ψ_state_2, ..., ψ_state_n, ∫(controls), controls, d(controls)]
    acontrol = [d²(controls)]
    ```

    where `controls = [uᵣ₁, uᵣ₂, ⋯ , uᵣₙ, uᵢ₁, uᵢ₂, ⋯ , uᵢₙ]`
    when `control_derivative_range = (-1,2)`

    This augmented state and control technique is based on work 
    in Propson, T. et al. Physical Review Applied 17 (2022). 

    Args:
        astate_size: size of astate
        num_controls: number of control fields
        control_derivative_range: 
            indicates which control derivatives (integrals) we are including
            e.g. (-1, 2) implies that ∫(controls), (controls), d(controls) are used in astate and acontrol = d²(controls)

    Returns:
        state_indices: astate indices for [[ψ_state_1, ψ_state_2, ..., ψ_state_n]
        icontrol_indices: astate indicies for [∫(controls)]
        control_indices: astate indices for [controls]
        dcontrol_indices: astate indices for [d(controls)]
    """
    # sizes
    real_control_size = 2 * num_controls # Factor of 2 comes from complex -> real isomorphism
    total_control_derivatives_in_astate = (control_derivative_range[2] - control_derivative_range[1])
    @assert total_control_derivatives_in_astate >= 0

    real_state_size = astate_size - total_control_derivatives_in_astate * real_control_size # Factor comes from inclusion of control derivatives

    # state
    state_indices = 1:real_state_size

    # controls
    icontrol_indices, control_indices, dcontrol_indices = nothing, nothing, nothing
    if total_control_derivatives_in_astate == 0
        return state_indices, icontrol_indices, control_indices, dcontrol_indices
    end

    num_controls_in_astate = 0
    if control_derivative_range[1] <= -1 && control_derivative_range[2] > -1
        start_index = (real_state_size + num_controls_in_astate * real_control_size)
        icontrol_indices = start_index+1:start_index+real_control_size
        num_controls_in_astate += 1
    end

    if control_derivative_range[1] <= 0 && control_derivative_range[2] > 0
        start_index = (real_state_size + num_controls_in_astate * real_control_size)
        control_indices = start_index+1:start_index+real_control_size
        num_controls_in_astate += 1
    end

    if control_derivative_range[1] <= 1 && control_derivative_range[2] > 1
        start_index = (real_state_size + num_controls_in_astate * real_control_size)
        dcontrol_indices = start_index+1:start_index+real_control_size
        num_controls_in_astate += 1
    end

    return state_indices, icontrol_indices, control_indices, dcontrol_indices
end

function extract_state_controls(astate, acontrol, num_controls::Int; control_derivative_range::Tuple{Int,Int}=(-1, 2))
    state_indices, icontrol_indices, control_indices, dcontrol_indices = generate_astate_indices(size(astate)[1], num_controls; control_derivative_range=control_derivative_range)

    state = astate[state_indices]
    icontrols, controls, dcontrols, ddcontrols = nothing, nothing, nothing, nothing

    # extract controls in astate
    if !isnothing(icontrol_indices)
        icontrols = astate[icontrol_indices]
    end

    if !isnothing(control_indices)
        controls = astate[control_indices]
    end

    if !isnothing(dcontrol_indices)
        dcontrols = astate[dcontrol_indices]
    end

    # extract control in acontrol
    if control_derivative_range[2] == -1
        icontrols = acontrol
    elseif control_derivative_range[2] == 0
        controls = acontrol
    elseif control_derivative_range[2] == 1
        dcontrols = acontrol
    elseif control_derivative_range[2] == 2
        ddcontrols = acontrol
    end

    return state, icontrols, controls, dcontrols, ddcontrols
end