# CUDA Fluid Simulation
2D fluid simulator built in CUDA and using OpenGL for visualisation. Based on the paper 'Real-Time Fluid Dynamics for Games' included in this repository. Please refer to the paper for a description of the parameters.

## Build Instructions
As this project uses CMake, simply use your favourite CLI/IDE/code editor and compile the `SolverExec` target. A modern C++ and CUDA compilers are needed. All required libraries and resources are provided in this repository

## Implementation Details
- Each pixel in the window is considered a cell
- Shared memory is heavily utilised to accelerate computation within thread blocks
- Simulations are done in CUDA memory buffers and their results are copied over to OpenGL textures for display
