module QControl

using QuantumOptics
using PyPlot
using Plots
pyplot()
using PyPlot: PyPlot
using Altro
using ForwardDiff
using LinearAlgebra
using SparseArrays
using RobotDynamics
using TrajectoryOptimization
using StaticArrays
using FiniteDiff
using JLD2
using PyCall
using FFTW
const TO = TrajectoryOptimization
const RD = RobotDynamics

include("utils.jl")
export save_solver_data, load_solver_data, states_to_kets, controls_to_amplitudes, generate_astate_indices, extract_state_controls

include("visualization.jl")
export plot_wigner, animate_wigner, plot_bloch, animate_bloch, plot_fft, testf

include("simulators.jl")
export solve_me, tanh_envelope

include("altro_solver.jl")
export QuantumState, gen_default_QR, gen_default_Qf, gen_default_objective, gen_LQR_params

end # module
