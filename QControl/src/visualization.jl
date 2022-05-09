function plot_wigner(state; wigner_pts::Vector=[-5:0.1:5;], trace_idx=nothing)
    """
    Function to plot quantum state of a bosonic mode in phase space via its Wigner function. 
    If a trace_idx is provided, optionally performs a partial trace of the input states along 
    the specified dimension. 
    """

    if isnothing(trace_idx)
        W = (state) -> π * wigner(state, wigner_pts, wigner_pts)
    else
        W = (state) -> π * wigner(ptrace(state, trace_idx), wigner_pts, wigner_pts)
    end

    heatmap(wigner_pts,
        wigner_pts,
        W(state),
        c=:seismic,
        clim=(-1, 1),
        aspect_ratio=:equal,
        framestyle=:box,
        titlefontsize=15,
        tickfontsize=12,
        tick_direction=:out,
        colorbar_tickfontsize=10
    )
    title!("Wigner Quasi-Probability Dist.")
end


function animate_wigner(states; wigner_pts::Vector=[-5:0.1:5;], trace_idx::Int=nothing, save_interval::Int=5)
    """
    Function to animate the trajectory of a quantum state in phase space. Plots Wigner functions 
    via plot_wigner(). If a trace_idx is provided, optionally performs a partial trace of the 
    quantum states along the dimension specified. 
    """

    @gif for state in states
        plot_wigner(state; wigner_pts, trace_idx)
    end every save_interval
end