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

function generate_astate_indices(astate_size::Int, num_controls::Int)
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
        astate_size: size of astate
        num_controls: number of control fields

    Returns:
        state_indices: astate indices for [[ψ_state_1, ψ_state_2, ..., ψ_state_n]
        icontrol_indices: astate indicies for [∫(controls)]
        control_indices: astate indices for [controls]
        dcontrol_indices: astate indices for [d(controls)]
    """
    full_control_size = 2 * num_controls # Factor of 2 comes from complex -> real isomorphism
    full_state_size = astate_size - 3 * full_control_size # Factor of 3 comes from ∫(controls), controls, d(controls)

    state_indices = 1:full_state_size
    icontrol_indices = full_state_size+1:full_state_size+full_control_size
    control_indices = (full_state_size+full_control_size)+1:(full_state_size+full_control_size)+full_control_size
    dcontrol_indices = (full_state_size+2*full_control_size)+1:(full_state_size+2*full_control_size)+full_control_size

    return state_indices, icontrol_indices, control_indices, dcontrol_indices
end