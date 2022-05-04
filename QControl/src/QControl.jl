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
const TO = TrajectoryOptimization
const RD = RobotDynamics

include("utils.jl")
include("visualization.jl")
include("altro_solver.jl")
export QuantumState, gen_LQR_params, plot_wigner

end # module
