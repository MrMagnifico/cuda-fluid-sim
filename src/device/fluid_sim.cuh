#ifndef _FLUID_SIM_CUH_
#define _FLUID_SIM_CUH_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec2.hpp>
DISABLE_WARNINGS_POP()

#include <render/config.h>

// Dictates how boundary edges will be dealt with
enum BoundaryStrategy { Conserve = 0,   // Use same sign as interior neighbour
                        Reverse };      // Reverse sign of interior neighbour

struct GlobalIndexing {
    unsigned int idX, idY, offset;                              // Coordinates and index into global field memory of current thread
    unsigned int verticalStride;                                // There are this many elements per row of the grid (includes ghost cells)
    unsigned int leftOffset, rightOffset, upOffset, downOffset; // Indices into global memory of axial neighbours
};

struct StatusFlags {
    bool validThread;       // Thread is actually handling a cell within field bounds
    bool handlingInterior;  // Thread is handling interior cell
};


// Allows for templating dynamically sized shared memory
template<typename T>
__device__ T* shared_memory_proxy() {
    extern __shared__ unsigned char memory[]; // TODO: Possible source of error due to lack of __align__
    return reinterpret_cast<T*>(memory);
}

/**
 * Add contribution of source cells to density field
 * 
 * @param densities Field of densities
 * @param sources Field of density-producing sources. Each entry represents how much density is produced for each time step
 * @param field_extents Number of non-ghost cells in each axis of the fields
 * @param num_cells Total number of cells in density field (INCLUDING ghost cells)
 * @param sim_params Simulation parameters
*/
template<typename T>
__global__ void add_sources(T* densities, T* sources, uint2 field_extents, SimulationParams sim_params);

/**
 * Compute diffusion of densities across field of cells
 * 
 * @param old_field The values of the field at time t-1. Assumed to have ghost cells
 * @param new_field Field to be populated with computed values (time t). Assumed to have ghost cells
 * @param field_extents Number of non-ghost cells in each axis of the fields
 * @param num_cells Total number of cells in density field (INCLUDING ghost cells)
 * @param bs Strategy for handling boundaries (See BoundaryStrategy)
 * @param sim_params Simulation parameters
*/
template<typename T>
__global__ void diffuse(T* old_field, T* new_field, uint2 field_extents, BoundaryStrategy bs, SimulationParams sim_params);


/**
 * Compute effect of vector field on density field
 * 
 * @param old_field The values of the field at time t-1. Assumed to have ghost cells
 * @param new_field Field to be populated with computed values (time t). Assumed to have ghost cells
 * @param velocity_field Field describing velocity vectors at each cell of grid (time t). Assumed to have ghost cells
 * @param field_extents Number of non-ghost cells in each axis of the fields
 * @param num_cells Total number of cells in density field (INCLUDING ghost cells)
 * @param bs Strategy for handling boundaries (See BoundaryStrategy)
 * @param sim_params Simulation parameters
*/
template<typename FieldT, typename VelocityT>
__global__ void advect(FieldT* old_field, FieldT* new_field, VelocityT* velocity_field, uint2 field_extents,
                       BoundaryStrategy bs, SimulationParams sim_params);

__global__ void project(glm::vec2* velocities, float* gradient_field, float* projection_field, uint2 field_extents,
                        SimulationParams sim_params);

/**
 * Handle the boundaries of a field. Can handle both global and shared memory fields
 * by passing offset and axial neighbour indices corresponding to shared memory.
 * 
 * @param field Field to be operated on
 * @param field_extents Number of non-ghost cells in each axis of the fields
 * @param bs Strategy for handling boundaries (See BoundaryStrategy)
 * @param tidX Global x-axis coordinate of thread in grid
 * @param tidY Global y-axis coordinate of thread in grid
 * @param offset Index of thread's field cell
 * @param leftIdx Index of thread's left neighbour's cell
 * @param rightIdx Index of thread's right neighbour's cell
 * @param upIdx Index of thread's top neighbour's cell
 * @param downIdx Index of thread's bottom neighbour's cell
*/
template<typename T>
__device__ void handle_boundary(T* field, uint2 field_extents, BoundaryStrategy bs,
                                unsigned int tidX, unsigned int tidY, unsigned int offset,
                                unsigned int leftIdx, unsigned int rightIdx, unsigned int upIdx, unsigned int downIdx);

/**
 * Generate data for indexing thread and axial neighbours in field memory
 * 
 * @param field_extents Number of non-ghost cells in each axis of the fields being operated on
 * 
 * @return See GlobalIndexing documentation
*/
__device__ GlobalIndexing generate_global_indices(uint2 field_extents);

/**
 * Generate flags for determing cell operation
 * 
 * @param globalIdX X-coordinate of the calling thread in the overall grid
 * @param globalIdY Y-coordinate of the calling thread in the overall grid
 * @param field_extents Number of non-ghost cells in each axis of the fields being operated on
 * 
 * @return See StatusFlags documentation
*/
__device__ StatusFlags generate_status_flags(unsigned int globalIdX, unsigned int globalIdY, uint2 field_extents);


#endif
