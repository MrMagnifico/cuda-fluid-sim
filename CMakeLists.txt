cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CudaFluidSolver C CXX CUDA)

# Enable CUDA debug symbols if compiling in debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
endif()

# Disable GLM CUDA warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20012")

# Set this before including framework such that it knows to use the OpenGL4.5 version of GLAD
if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/framework")
	# Create framework library and include CMake scripts (compiler warnings, sanitizers and static analyzers).
	add_subdirectory("framework")
else()
	# During development the framework lives in parent folder.
	add_subdirectory("../../../framework/" "${CMAKE_BINARY_DIR}/framework/")
endif()

# Additional source files
add_library(FluidLib "")
enable_sanitizers(FluidLib)
include(${CMAKE_CURRENT_LIST_DIR}/src/CMakeLists.txt)
target_include_directories(FluidLib PUBLIC "${CMAKE_CURRENT_LIST_DIR}/src/")
target_link_libraries(FluidLib PUBLIC CGFramework)
set_property(TARGET FluidLib PROPERTY CUDA_ARCHITECTURES native)

# Main executable
add_executable(SolverExec "src/main.cu")
enable_sanitizers(SolverExec)
target_link_libraries(SolverExec PRIVATE FluidLib)
set_property(TARGET SolverExec PROPERTY CUDA_ARCHITECTURES native)

# Preprocessor definitions for paths
target_compile_definitions(FluidLib PUBLIC "-DSHADERS_DIR=\"${CMAKE_CURRENT_LIST_DIR}/shaders/\"")
