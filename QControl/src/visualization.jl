function plot_wigner(state::Ket; wigner_pts::Vector=[-5:0.1:5;])
    W = π * wigner(state, wigner_pts, wigner_pts)
    pcolor(wigner_pts, wigner_pts, W, cmap="seismic", vmin=-1, vmax=1)
    colorbar()
    title("Wigner Quasi-Probability Dist.")
    tight_layout()
    # savefig("fock.png")
end