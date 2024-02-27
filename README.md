# CUDA Fluid Simulation
2D fluid simulator built in CUDA and using OpenGL for visualisation. Based on the paper 'Real-Time Fluid Dynamics for Games' included in this repository. Please refer to the paper for a description of the parameters.

![eye-candy](images/eye-candy.png)

## Build Instructions
As this project uses CMake, simply use your favourite CLI/IDE/code editor and compile the `SolverExec` target. A modern C++ and CUDA compilers are needed. All required libraries and resources are provided in this repository

## Implementation Details
- Each pixel in the window is modelled as a cell with separate floating point RGB components
- Shared memory is heavily utilised to accelerate computation within thread blocks
- Simulations are done in CUDA memory buffers and their results are copied over to OpenGL textures for display
- Exposure tonemapping is carried out to map from 32-bit floating point values to 8-bit sRGB colour values
