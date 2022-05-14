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
        W(state)',
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

# Bloch Plots

function plot_bloch(ρs; plotstyle="point")
    """
    Function to plot set of qubit state vectors or density matrices on the Bloch sphere.
    Renders/plots Bloch sphere via PyCall/qutip and returns Bloch sphere PyObject. 
    """

    qt = pyimport("qutip")
    b = qt.Bloch()

    # Bloch sphere formatting
    b.vector_color = ["r"]
    b.point_color = ["b"]
    b.point_marker = ["o"]
    b.zlpos = [1.25, -1.35]

    pos(ρ) = [real(expect(f(SpinBasis(1 // 2)), ρ)) for f in [sigmax, sigmay, sigmaz]]

    for ρ in ρs
        if plotstyle == "vector"
            b.add_vectors(pos(ρ))
        elseif plotstyle == "point"
            b.add_points(pos(ρ))
        end
    end

    b.render()
    b
end

function animate_bloch(ρs; duration=0.03, save_all=false)
    """
    Function to animate a trajectory of a qubit state vector or density matrix on the Bloch 
    sphere. Renders Bloch sphere via PyCall/qutip, saves via PyCall/imageio. Adapted from 
    https://sites.google.com/site/tanayroysite/articles/bloch-sphere-animation-using-qutip
    """

    io = pyimport("imageio")
    qt = pyimport("qutip")
    b = qt.Bloch()

    # Bloch sphere formatting
    b.vector_color = ["r"]
    b.point_color = ["b"]
    b.point_marker = ["o"]
    b.zlpos = [1.2, -1.35]

    pos(ρ) = [real(expect(f(SpinBasis(1 // 2)), ρ)) for f in [sigmax, sigmay, sigmaz]]

    images = []
    for idx in 1:length(ρs)
        b.clear()
        b.add_vectors(pos(ρs[idx]))
        b.add_points(pos(ρs[1:idx]))
        if save_all
            b.save(dirc="tmp") #saving images to tmp directory
            filename = string("tmp/bloch_", idx - 1, ".png")
        else
            filename = "temp_file.png"
            b.save(filename)
        end
        push!(images, io.imread(filename))
    end

    io.mimsave("bloch_anim.gif", images, duration=duration)

    b.render()
    b
end