module QControl

using QuantumOptics
using PyPlot
using Altro
using ForwardDiff
using LinearAlgebra
using SparseArrays
using RobotDynamics
using TrajectoryOptimization
using StaticArrays
using FiniteDiff
using JLD2
const TO = TrajectoryOptimization
const RD = RobotDynamics

include("utils.jl")
export save_solver_data, load_solver_data, states_to_kets, controls_to_amplitudes

include("visualization.jl")
export plot_wigner

include("simulators.jl")
export solve_me

include("altro_solver.jl")
export QuantumState, gen_LQR_params

end # module
